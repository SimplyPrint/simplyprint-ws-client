"""The device-side half of one printer client: how the printer is reached.

A :class:`DeviceDriver` owns one lifecycle (``start`` -> ``ensure_started`` ->
``suspend`` -> ``stop``), liveness (``is_connected`` tri-state,
``last_message_at``), and the single-flight credential-refresh choreography every
re-authenticating device needs. Concrete drivers — the pooled links in
:mod:`~simplyprint_ws_client.integration.drivers` and the request/response
:class:`~simplyprint_ws_client.integration.drivers.DevicePoller` — deliver device
edges by calling their client's ``on_device_connected`` /
``on_device_disconnected`` / ``on_device_message`` hooks, already on the client's
loop, so a brand never writes thread-hop or re-emit plumbing again.

The base :class:`~simplyprint_ws_client.integration.client.PrinterClient` owns
WHEN drivers run: it starts every declared driver in ``init``, sweeps
``ensure_started`` each tick (a driver whose config wasn't ready yet retries for
free), suspends on ``halt``, and stops on ``teardown``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Optional,
    TypeVar,
    Union,
)

import yarl

from simplyprint_ws_client.wire import mqtt as mqtt_front_door
from simplyprint_ws_client.wire import websocket as ws_front_door
from simplyprint_ws_client.wire.errors import FatalError
from simplyprint_ws_client.wire.events import Connected, Disconnected, MessageReceived
from simplyprint_ws_client.wire.lease import Lease, MqttLease, WsLease
from simplyprint_ws_client.wire.messages import MqttMessage, WsMessage
from simplyprint_ws_client.wire.options import ConnectionOptions

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.client import PrinterClient

TLease = TypeVar("TLease", bound=Lease)

__all__ = [
    "DeviceAuthError",
    "DeviceDriver",
    "DevicePoller",
    "LeaseDriver",
    "MqttDriver",
    "WsDriver",
]


class DeviceAuthError(Exception):
    """The device rejected our credentials (session/token expired).

    Raise it from ``poll_device()`` (or anywhere a driver surfaces it) to trigger
    the single-flight ``refresh_device_credentials`` -> restart choreography.
    """


class DeviceDriver(ABC):
    """One client's supervised attachment to its physical device.

    Lifecycle contract (the base printer client drives it):

    * :meth:`start` — idempotent and tolerant: a device whose config is not ready
      yet (no host, no credentials) logs and returns; the per-tick
      :meth:`ensure_started` sweep retries for free.
    * :meth:`suspend` — the client left scheduling temporarily (``halt``).
    * :meth:`stop` — final teardown. Idempotent.
    * :meth:`restart` — stop + start; re-resolves URLs/sessions, which is why
      link parameters are callables.

    Liveness: ``is_connected`` is tri-state (``None`` = never connected yet) and
    ``last_message_at`` is a monotonic timestamp of the last inbound sign of
    life — both fed by the concrete driver.

    Credential refresh: :meth:`request_credential_refresh` is single-flight; it
    awaits the client's ``refresh_device_credentials(driver)`` on the client's
    loop and restarts the driver on ``True``.
    """

    def __init__(self, client: "PrinterClient", *, name: str = "device") -> None:
        self.client = client
        self.name = name
        self.is_connected: Optional[bool] = None
        self.last_message_at: Optional[float] = None
        self._refresh_lock = threading.Lock()
        self._refreshing = False

    @abstractmethod
    def start(self) -> None:
        """Begin reaching the device. Idempotent; 'not ready yet' is tolerated."""

    def ensure_started(self) -> None:
        """Cheap per-tick retry; the default just calls the idempotent start."""
        self.start()

    def suspend(self) -> None:
        """The client is temporarily out of scheduling. Default: full stop."""
        self.stop()

    @abstractmethod
    def stop(self) -> None:
        """Tear the attachment down. Idempotent."""

    def restart(self) -> None:
        """Stop and start again, re-resolving URLs/sessions."""
        self.stop()
        self.start()

    def request_credential_refresh(self) -> None:
        """Run the client's credential refresh exactly once, then restart.

        Safe to call from any thread and repeatedly: concurrent requests
        coalesce into the one in flight (the anycubic-style lock+flag, owned
        here once).
        """
        with self._refresh_lock:
            if self._refreshing:
                return
            self._refreshing = True
        self.client.submit_to_loop(self._run_credential_refresh())

    async def _run_credential_refresh(self) -> None:
        refreshed = False
        try:
            refreshed = await self.client.refresh_device_credentials(self)
        except Exception:  # noqa: BLE001 -- a failed refresh must not kill the loop
            self.client.logger.warning(
                "device credential refresh failed", exc_info=True
            )
        finally:
            with self._refresh_lock:
                self._refreshing = False
        if refreshed:
            self.client.logger.info(
                "device credentials refreshed; restarting %s", self.name
            )
            self.restart()


#: Resolves the device endpoint at (re)start time -- a callable so a restart
#: after a credential refresh picks up the just-updated config.
UrlFactory = Callable[[], Union[str, yarl.URL]]


class LeaseDriver(DeviceDriver, Generic[TLease]):
    """A driver over a pooled lease; concrete drivers pick the front door.

    Generic over its lease type, so a concrete driver's ``lease`` carries the
    full protocol API (``MqttDriver(...).lease.subscribe_soon`` resolves in an
    IDE without casts).
    """

    def __init__(
        self,
        client: "PrinterClient",
        url: UrlFactory,
        *,
        name: str = "device",
        options: Optional[ConnectionOptions] = None,
    ) -> None:
        super().__init__(client, name=name)
        self._url = url
        self._options = options
        self.lease: Optional[TLease] = None

    def _connect(self, url: Union[str, yarl.URL], options: ConnectionOptions) -> TLease:
        raise NotImplementedError

    def _on_lease_acquired(self, lease: TLease) -> None:
        """Post-connect per-protocol setup (e.g. topic subscriptions)."""

    @property
    def connected(self) -> bool:
        """The wire is up AND the device has shown signs of life."""
        lease = self.lease
        return lease is not None and lease.connected and self.is_connected is True

    def start(self) -> None:
        if self.lease is not None and not self.lease.closed:
            return
        try:
            loop = self.client.event_loop
        except RuntimeError:
            loop = None
        if loop is None or not loop.is_running():
            self.client.logger.debug(
                "cannot start %s link yet: no running event loop", self.name
            )
            return
        try:
            url = yarl.URL(str(self._url()))
        except Exception as error:  # noqa: BLE001 -- config not ready yet; tick retries
            self.client.logger.debug("cannot start %s link yet: %s", self.name, error)
            return

        options = self._options or ConnectionOptions()
        if options.provider is None:
            # The lease's courier must deliver on the client's loop.
            options = replace(options, provider=self.client)

        try:
            lease = self._connect(url, options)
        except Exception as error:  # noqa: BLE001 -- bad URL/params; tick retries
            self.client.logger.debug("cannot start %s link yet: %s", self.name, error)
            return

        self.lease = lease
        lease.event_bus.on(MessageReceived, self._on_wire_message)
        lease.event_bus.on(Connected, self._on_wire_connected)
        lease.event_bus.on(Disconnected, self._on_wire_disconnected)
        self._on_lease_acquired(lease)
        if lease.connected:
            # Attached to an already-live shared wire: deliver the edge now.
            lease.create_task(self._on_wire_connected(None))

    def stop(self) -> None:
        lease = self.lease
        self.lease = None
        if lease is None:
            return
        lease.close_soon()

    def send_soon(self, message: object) -> bool:
        """Schedule a send if the wire is up; ``False`` (and no raise) if not."""
        lease = self.lease
        if lease is None or not lease.connected:
            return False
        return lease.send_soon(message)

    async def send(self, message: object) -> None:
        """Send on the live wire (raises if the link is down)."""
        lease = self.lease
        if lease is None:
            raise ConnectionError(f"{self.name} link is not started")
        await lease.send(message)

    async def _on_wire_connected(self, _event: object) -> None:
        self.is_connected = True
        await self.client.on_device_connected(self)

    async def _on_wire_disconnected(self, event: Disconnected) -> None:
        self.is_connected = False
        await self.client.on_device_disconnected(self, reason=event.code)
        if isinstance(event.code, FatalError):
            self.request_credential_refresh()

    async def _on_wire_message(self, event: MessageReceived) -> None:
        self.last_message_at = time.monotonic()
        await self.client.on_device_message(self._payload(event.message), self)

    @staticmethod
    def _payload(message: object) -> object:
        return message


class WsDriver(LeaseDriver[WsLease]):
    """A 1:1 WebSocket attachment: every frame is this client's.

    Inbound frames reach ``on_device_message`` as their raw payload
    (``str``/``bytes``) — the unwrap four brands wrote defensively is owned here.
    """

    def _connect(
        self, url: Union[str, yarl.URL], options: ConnectionOptions
    ) -> WsLease:
        return ws_front_door.connect(url, options=options)

    @staticmethod
    def _payload(message: object) -> object:
        return message.payload if isinstance(message, WsMessage) else message


class MqttDriver(LeaseDriver[MqttLease]):
    """A broker attachment: topics multiplexed over one shared socket.

    ``topics`` resolves at (re)start so a restart re-subscribes against the
    fresh config. Inbound messages reach ``on_device_message`` as
    :class:`~simplyprint_ws_client.wire.messages.MqttMessage` (topic +
    payload — the topic is routing information the client needs).
    """

    def __init__(
        self,
        client: "PrinterClient",
        url: UrlFactory,
        *,
        topics: Callable[[], Iterable[str]] = tuple,
        impl: str = "paho",
        name: str = "device",
        options: Optional[ConnectionOptions] = None,
    ) -> None:
        super().__init__(client, url, name=name, options=options)
        self._topics = topics
        self._impl = impl

    def _connect(
        self, url: Union[str, yarl.URL], options: ConnectionOptions
    ) -> MqttLease:
        return mqtt_front_door.connect(url, impl=self._impl, options=options)

    def _on_lease_acquired(self, lease: TLease) -> None:
        for topic in self._topics():
            lease.subscribe_soon(topic)

    def publish_soon(self, message: MqttMessage) -> bool:
        """Schedule a publish if the wire is up; ``False`` if not."""
        return self.send_soon(message)


class DevicePoller(DeviceDriver):
    """Drives ``poll_device()`` on an interval and owns the edge bookkeeping."""

    def __init__(
        self,
        client: "PrinterClient",
        *,
        interval: float,
        poll: Optional[Callable[[], Awaitable[None]]] = None,
        offline_after: float = 300.0,
        failure_backoff: float = 10.0,
        name: str = "device",
    ) -> None:
        super().__init__(client, name=name)
        self.interval = interval
        self.offline_after = offline_after
        self.failure_backoff = failure_backoff
        self._poll = poll
        self._task: Optional[asyncio.Task] = None

    @property
    def connected(self) -> bool:
        return self.is_connected is True

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = self.client.event_loop
        except RuntimeError:
            loop = None
        if loop is None or not loop.is_running():
            self.client.logger.debug(
                "cannot start %s poller yet: no running event loop", self.name
            )
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._task = loop.create_task(self._run())
        else:
            loop.call_soon_threadsafe(self._start_on_loop, loop)

    def _start_on_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run())

    def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()

    async def _run(self) -> None:
        started = time.monotonic()
        while True:
            failed = False
            try:
                await (self._poll or self.client.poll_device)()
            except asyncio.CancelledError:
                raise
            except DeviceAuthError:
                failed = True
                self.request_credential_refresh()
            except Exception:  # noqa: BLE001 -- supervised: a bad poll backs off
                failed = True
                self.client.logger.debug("%s poll failed", self.name, exc_info=True)
            else:
                self.last_message_at = time.monotonic()
                if self.is_connected is not True:
                    self.is_connected = True
                    await self.client.on_device_connected(self)

            # Silence (since the last sign of life, or since start for a device
            # never reached) flips the edge exactly once until contact resumes.
            last_life = (
                self.last_message_at if self.last_message_at is not None else started
            )
            if (
                self.is_connected is not False
                and time.monotonic() - last_life >= self.offline_after
            ):
                self.is_connected = False
                await self.client.on_device_disconnected(self, reason=None)

            await asyncio.sleep(self.failure_backoff if failed else self.interval)
