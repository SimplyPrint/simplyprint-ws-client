"""The device-side half of one printer client: how the printer is reached.

A :class:`DeviceDriver` owns one lifecycle (``start`` -> ``ensure_started`` ->
``close``), one typed device session, and the single-flight credential-refresh
choreography every re-authenticating device needs. Concrete drivers — the pooled links in
:mod:`~simplyprint_ws_client.integration.drivers` and the request/response
:class:`~simplyprint_ws_client.integration.drivers.DevicePoller` — deliver device
edges by calling their client's ``on_device_connected`` /
``on_device_disconnected`` / ``on_device_message`` hooks, already on the client's
loop, so a brand never writes thread-hop or re-emit plumbing again.

The base :class:`~simplyprint_ws_client.integration.client.PrinterClient` owns
WHEN drivers run: it starts every declared driver in ``init``, sweeps
``ensure_started`` each tick (a driver whose config wasn't ready yet retries for
free), and closes them on ``teardown``. Drivers run for the client's whole scheduled
lifetime -- they are what produce device reachability, so they are never gated
on the client's allocation flag.
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Optional,
    TypeVar,
    Union,
)

import yarl

from simplyprint_ws_client.wire import mqtt as mqtt_front_door
from simplyprint_ws_client.wire import websocket as ws_front_door
from simplyprint_ws_client.wire.errors import AuthenticationError, TransientError
from simplyprint_ws_client.wire.events import (
    ActivityTimeout,
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.lease import Lease, MqttLease, WsLease
from simplyprint_ws_client.wire.messages import MqttMessage, WsMessage
from simplyprint_ws_client.wire.options import ConnectionOptions

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.client import PrinterClient

TLease = TypeVar("TLease", bound=Lease)
#: The inbound payload shape a lease driver hands to ``on_device_message``:
#: ``str``/``bytes`` for a WebSocket frame, :class:`MqttMessage` for a broker.
TPayload = TypeVar("TPayload")

__all__ = [
    "DeviceAuthError",
    "DeviceDriver",
    "DevicePoller",
    "DeviceReachability",
    "DeviceSession",
    "DeviceSource",
    "LeaseDriver",
    "MqttDriver",
    "WsDriver",
]


class DeviceAuthError(Exception):
    """The device rejected our credentials (session/token expired).

    Raise it from ``poll_device()`` (or anywhere a driver surfaces it) to trigger
    the single-flight ``refresh_device_credentials`` -> restart choreography.
    """


class DeviceReachability(Enum):
    """What one driver currently knows about its physical device."""

    NEVER_SEEN = "never_seen"
    UP = "up"
    DOWN = "down"
    STOPPED = "stopped"


@dataclass(frozen=True)
class DeviceSource:
    """The concrete lease generation that produced a session observation."""

    lease_id: int
    wire_generation: int


@dataclass(frozen=True)
class DeviceSession:
    """The complete liveness record for one driver.

    ``generation`` advances once per continuous reachable period.
    ``observed_at`` is the last activity while UP and the first observation
    time while DOWN.
    """

    generation: int = 0
    reachability: DeviceReachability = DeviceReachability.NEVER_SEEN
    source: Optional[DeviceSource] = None
    observed_at: float = field(default_factory=time.monotonic)
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("device session generation cannot be negative")
        if self.reason is not None and self.reachability is not DeviceReachability.DOWN:
            raise ValueError("only a down session can carry a reason")


@dataclass(frozen=True)
class _StartFailure:
    """One continuous inability to construct a link."""

    since: float
    reason: str


class DeviceDriver(ABC):
    """One client's supervised attachment to its physical device.

    Lifecycle contract (the base printer client drives it):

    * :meth:`start` — idempotent and tolerant: a device whose config is not ready
      yet (no host, no credentials) logs and returns; the per-tick
      :meth:`ensure_started` sweep retries for free.
    * :meth:`stop` — stop the current attachment. Idempotent.
    * :meth:`close` — final, awaited teardown; the session becomes ``STOPPED``.
    * :meth:`restart` — stop + start; re-resolves URLs/sessions, which is why
      link parameters are callables.

    Liveness is the immutable :attr:`session` record. Repeating an edge is a
    no-op; reconnecting advances its generation. Device reachability is kept
    separate from device-reported printer and job state.

    Credential refresh: :meth:`request_credential_refresh` is single-flight; it
    awaits the client's ``refresh_device_credentials(driver)`` on the client's
    loop and restarts the driver on ``True``.
    """

    #: The driver's default name when the brand passes none. Concrete drivers
    #: override it per kind (``mqtt``/``ws``/``poll``); the name labels log
    #: lines AND names the per-printer log file the wire logs land in
    #: (``<log_dir>/<uid>/<name>.log``).
    default_name: ClassVar[str] = "device"

    def __init__(
        self,
        client: "PrinterClient",
        *,
        name: Optional[str] = None,
    ) -> None:
        self.client = client
        self.name = name or type(self).default_name
        self.session = DeviceSession()
        self._refresh_lock = threading.Lock()
        self._refreshing = False

    @property
    def connected(self) -> bool:
        return self.session.reachability is DeviceReachability.UP

    async def set_reachability(
        self,
        reachability: DeviceReachability,
        *,
        reason: Optional[object] = None,
        source: Optional[DeviceSource] = None,
    ) -> bool:
        """Apply one observed UP/DOWN edge and return whether it was new.

        This is the public path for protocol-level availability reports too;
        brands never invoke client edge hooks directly. ``STOPPED`` is owned by
        :meth:`close`, and ``NEVER_SEEN`` is only the initial state.
        """
        if reachability not in (DeviceReachability.UP, DeviceReachability.DOWN):
            raise ValueError("only UP and DOWN are observable reachability edges")

        current = self.session
        if current.reachability is DeviceReachability.STOPPED:
            return False
        source = source or current.source
        if current.reachability is reachability and current.source == source:
            return False
        if (
            reachability is DeviceReachability.DOWN
            and current.source is not None
            and source != current.source
        ):
            return False

        now = time.monotonic()

        if reachability is DeviceReachability.UP:
            session = DeviceSession(
                generation=current.generation + 1,
                reachability=reachability,
                source=source,
                observed_at=now,
            )
            self.session = session
            await self.client.on_device_connected(self)
            return True

        session = DeviceSession(
            generation=current.generation,
            reachability=reachability,
            source=source,
            observed_at=now,
            reason=str(reason) if reason is not None else None,
        )
        self.session = session
        await self.client.on_device_disconnected(self, reason=reason)
        return True

    def note_device_message(self) -> bool:
        """Record an inbound sign of life without fabricating another edge."""
        if self.session.reachability is not DeviceReachability.UP:
            return False
        self.session = replace(self.session, observed_at=time.monotonic())
        return True

    def _close_session(self) -> None:
        current = self.session
        self.session = DeviceSession(
            generation=current.generation,
            reachability=DeviceReachability.STOPPED,
            source=current.source,
            observed_at=time.monotonic(),
        )

    @abstractmethod
    def start(self) -> None:
        """Begin reaching the device. Idempotent; 'not ready yet' is tolerated."""

    def ensure_started(self) -> None:
        """Cheap per-tick retry; the default just calls the idempotent start."""
        self.start()

    def ensure_current(self) -> None:
        """Re-resolve against the (possibly edited) config; restart if stale.

        Called by the base printer client whenever the config changes, so an
        edited host/credential takes effect without a process restart. The
        default is a no-op (a poller reads its config per poll); URL-bound
        drivers compare and restart only when the endpoint actually moved.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear the attachment down. Idempotent."""

    async def close(self) -> None:
        """Stop final work and make the session terminal."""
        self._close_session()
        self.stop()

    def restart(self) -> None:
        """Restart through a real down edge, then re-resolve the attachment."""
        self.client.submit_to_loop(self._restart())

    async def _restart(self) -> None:
        await self.set_reachability(DeviceReachability.DOWN, reason="driver restart")
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


class LeaseDriver(DeviceDriver, Generic[TLease, TPayload]):
    """A driver over a pooled lease; concrete drivers pick the front door.

    Generic over its lease type, so a concrete driver's ``lease`` carries the
    full protocol API (``MqttDriver(...).lease.subscribe`` resolves in an
    IDE without casts), and over its inbound payload type, so ``message_payload`` and
    the wire-message handler agree on what ``on_device_message`` receives.
    """

    def __init__(
        self,
        client: "PrinterClient",
        url: UrlFactory,
        *,
        name: Optional[str] = None,
        options: Optional[ConnectionOptions] = None,
    ) -> None:
        super().__init__(client, name=name)
        self._url = url
        self._options = options
        self.lease: Optional[TLease] = None
        # restart() is single-flight: concurrent requests coalesce into the
        # in-flight sequence, which loops once more (re-resolving the URL) so
        # the newest config always wins.
        self._restart_lock = threading.Lock()
        self._restarting = False
        self._restart_again = False
        self._stopped = False
        self._start_failure: Optional[_StartFailure] = None

    @abstractmethod
    def acquire_lease(
        self, url: Union[str, yarl.URL], options: ConnectionOptions
    ) -> TLease:
        raise NotImplementedError

    def configure_lease(self, lease: TLease) -> None:
        """Post-connect per-protocol setup (e.g. topic subscriptions)."""

    @property
    def connected(self) -> bool:
        """The wire is up AND the device has shown signs of life."""
        lease = self.lease
        return lease is not None and lease.connected and super().connected

    def start(self) -> None:
        if self.session.reachability is DeviceReachability.STOPPED:
            return
        self._stopped = False
        lease = self.lease
        if lease is not None and not lease.closed:
            if not lease.transport.supervising():
                self.client.logger.debug(
                    "%s link supervision stopped; restarting it", self.name
                )
                lease.transport.start()
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
        except DeviceAuthError as error:
            # The factory is pure; it signals "credentials must be minted"
            # and the client's refresh hook does the (off-loop) work.
            self.client.logger.debug("%s link needs credentials: %s", self.name, error)
            self._count_start_failure(error)
            self.request_credential_refresh()
            return
        except Exception as error:  # noqa: BLE001 -- config not ready yet; tick retries
            self.client.logger.debug("cannot start %s link yet: %s", self.name, error)
            self._count_start_failure(error)
            return

        options = self._options or ConnectionOptions()
        if options.provider is None:
            # The lease's courier must deliver on the client's loop.
            options = replace(options, provider=self.client)
        if options.logger is None:
            # Wire lifecycle logs land in this printer's own log files
            # (``printers.<uid>.<name>`` -> ``<uid>/<name>.log``).
            options = replace(options, logger=self.client.logger.getChild(self.name))

        try:
            lease = self.acquire_lease(url, options)
        except Exception as error:  # noqa: BLE001 -- bad URL/params; tick retries
            self.client.logger.debug("cannot start %s link yet: %s", self.name, error)
            self._count_start_failure(error)
            return

        self.lease = lease
        self._start_failure = None
        lease.event_bus.on(MessageReceived, partial(self.wire_message, lease))
        lease.event_bus.on(Connected, partial(self.wire_connected, lease))
        lease.event_bus.on(Disconnected, partial(self.wire_disconnected, lease))
        lease.event_bus.on(ActivityTimeout, partial(self.wire_inactive, lease))
        self.configure_lease(lease)
        if lease.connected:
            # Attached to an already-live shared wire: deliver the edge now.
            lease.create_task(self.wire_connected(lease, Connected(lease.generation)))

    #: A short startup-ordering window before an unconstructable link becomes
    #: a real DOWN observation. Time, rather than scheduler call count, defines
    #: the bound.
    START_FAILURE_DOWN_AFTER = 30.0

    def _count_start_failure(self, error: Exception) -> None:
        """Track one continuous start failure and publish it after the bound."""
        now = time.monotonic()
        reason = str(error) or type(error).__name__
        failure = self._start_failure
        if failure is None:
            failure = self._start_failure = _StartFailure(now, reason)
        elif failure.reason != reason:
            failure = self._start_failure = _StartFailure(failure.since, reason)
        if (
            now - failure.since < self.START_FAILURE_DOWN_AFTER
            or self.session.reachability is DeviceReachability.DOWN
        ):
            return
        coro = self.set_reachability(
            DeviceReachability.DOWN, reason=f"link could not start: {failure.reason}"
        )
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            loop = self.client.event_loop
        except RuntimeError:
            loop = None
        if loop is not None and running is loop:
            loop.create_task(coro)
        else:
            self.client.submit_to_loop(coro)

    def ensure_current(self) -> None:
        """Restart iff the URL factory now resolves somewhere else.

        A config edit that doesn't move the endpoint (a renamed printer, a
        toggled flag) must not bounce a healthy wire; one that does (new host,
        rotated credentials) reconnects immediately.
        """
        lease = self.lease
        if lease is None or lease.closed:
            return  # not started; start()/the tick sweep own that path
        try:
            url = yarl.URL(str(self._url()))
        except Exception:  # noqa: BLE001 -- config now incomplete; keep the live wire
            return
        if url != lease.url:
            self.restart()

    def stop(self) -> None:
        self._stopped = True
        lease = self.lease
        self.lease = None
        if lease is None:
            return
        lease.close_soon()

    async def close(self) -> None:
        """Make the session terminal, then release and await the lease."""
        self._stopped = True
        self._close_session()
        lease, self.lease = self.lease, None
        if lease is not None:
            await lease.close()

    def restart(self) -> None:
        """Stop and start again -- properly sequenced.

        The old lease is closed and AWAITED before the new one is built: its
        refcount release stops the shared transport (a plain stop+start would
        re-lease the still-open endpoint and never actually bounce the wire),
        and its handlers detach before the new lease exists (so a stale
        ``Disconnected`` can never clobber the fresh link). When the endpoint
        is not moving, the shared wire is additionally kicked via ``trip`` --
        sibling leases may hold the refcount above zero, and a stuck-but-open
        connection must still reconnect. Sync-callable from any thread;
        single-flight with coalescing.
        """
        with self._restart_lock:
            if self._restarting:
                self._restart_again = True
                return
            self._restarting = True
        self.client.submit_to_loop(self._restart_sequence())

    async def _restart_sequence(self) -> None:
        try:
            while True:
                try:
                    await self.set_reachability(
                        DeviceReachability.DOWN, reason="driver restart"
                    )
                    lease, self.lease = self.lease, None
                    if lease is not None and not lease.closed:
                        await self._bounce(lease)
                    if not self._stopped:
                        self.start()
                except Exception:  # noqa: BLE001 -- a failed bounce must not wedge restarts
                    self.client.logger.warning(
                        "%s restart failed", self.name, exc_info=True
                    )
                with self._restart_lock:
                    if not self._restart_again:
                        return
                    self._restart_again = False
        finally:
            with self._restart_lock:
                self._restarting = False

    async def _bounce(self, lease: TLease) -> None:
        """Tear one lease fully down, kicking the shared wire when the
        endpoint is not moving (same URL: siblings keep the transport leased,
        so close/reopen alone would hand back the same stuck wire)."""
        try:
            same_url = yarl.URL(str(self._url())) == lease.url
        except Exception:  # noqa: BLE001 -- config in flux counts as "moved"
            same_url = False
        if same_url:
            lease.transport.trip(
                lease.transport.generation, TransientError("driver restart")
            )
        await lease.close()

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

    def _wire_source(self, lease: TLease, event: object) -> Optional[DeviceSource]:
        return (
            DeviceSource(id(lease), event.generation)
            if lease is self.lease
            and isinstance(
                event, (ActivityTimeout, Connected, Disconnected, MessageReceived)
            )
            and event.generation == lease.generation
            else None
        )

    async def set_reachability(
        self,
        reachability: DeviceReachability,
        *,
        reason: Optional[object] = None,
        source: Optional[DeviceSource] = None,
    ) -> bool:
        lease = self.lease
        if source is None and lease is not None and not lease.closed:
            source = DeviceSource(id(lease), lease.generation)
        return await super().set_reachability(
            reachability, reason=reason, source=source
        )

    def note_device_message(self) -> bool:
        lease = self.lease
        if lease is not None:
            lease.note_activity()
        return super().note_device_message()

    async def wire_connected(self, lease: TLease, event: Connected) -> None:
        source = self._wire_source(lease, event)
        if source is not None:
            await self.set_reachability(DeviceReachability.UP, source=source)

    async def wire_disconnected(self, lease: TLease, event: Disconnected) -> None:
        source = self._wire_source(lease, event)
        if source is None:
            return
        await self.set_reachability(
            DeviceReachability.DOWN, reason=event.code, source=source
        )
        if isinstance(event.code, AuthenticationError):
            self.request_credential_refresh()

    async def wire_inactive(self, lease: TLease, event: ActivityTimeout) -> None:
        source = self._wire_source(lease, event)
        if source is not None and lease.last_activity == event.last_activity:
            await self.set_reachability(
                DeviceReachability.DOWN, reason=event.code, source=source
            )

    async def wire_message(self, lease: TLease, event: MessageReceived) -> None:
        source = self._wire_source(lease, event)
        if source is None:
            return
        await self.client.on_device_message(self.message_payload(event.message), self)

    @staticmethod
    def message_payload(message: object) -> TPayload:
        return message


class WsDriver(LeaseDriver[WsLease, Union[str, bytes]]):
    """A 1:1 WebSocket attachment: every frame is this client's.

    Inbound frames reach ``on_device_message`` as their raw payload
    (``str``/``bytes``) — the unwrap four brands wrote defensively is owned here.
    """

    default_name = "ws"

    def acquire_lease(
        self, url: Union[str, yarl.URL], options: ConnectionOptions
    ) -> WsLease:
        pool = ws_front_door.pool_for(
            self.client.context.websocket_pools,
            options.provider,
            options.wire_keepalive,
        )
        return ws_front_door.connect(url, pool=pool, options=options)

    @staticmethod
    def message_payload(message: object) -> Union[str, bytes]:
        return message.payload if isinstance(message, WsMessage) else message


class MqttDriver(LeaseDriver[MqttLease, MqttMessage]):
    """A broker attachment: topics multiplexed over one shared socket.

    ``topics`` resolves at (re)start so a restart re-subscribes against the
    fresh config. Inbound messages reach ``on_device_message`` as
    :class:`~simplyprint_ws_client.wire.messages.MqttMessage` (topic +
    payload — the topic is routing information the client needs).
    """

    default_name = "mqtt"

    def __init__(
        self,
        client: "PrinterClient",
        url: UrlFactory,
        *,
        topics: Callable[[], Iterable[str]] = tuple,
        name: Optional[str] = None,
        options: Optional[ConnectionOptions] = None,
    ) -> None:
        super().__init__(client, url, name=name, options=options)
        self._topics = topics

    def acquire_lease(
        self, url: Union[str, yarl.URL], options: ConnectionOptions
    ) -> MqttLease:
        pool = mqtt_front_door.pool_for(
            self.client.context.mqtt_pools,
            options.provider,
            options.wire_keepalive,
        )
        return mqtt_front_door.connect(
            url, pool=pool, options=options, topics=self._topics()
        )

    async def wire_connected(self, lease: MqttLease, event: Connected) -> None:
        if self._wire_source(lease, event) is not None:
            await self.client.on_device_transport_connected(self)

    def publish_soon(self, message: MqttMessage) -> bool:
        """Schedule a publish if the wire is up; ``False`` if not."""
        return self.send_soon(message)


class DevicePoller(DeviceDriver):
    """Drives ``poll_device()`` on an interval and owns the edge bookkeeping."""

    default_name = "poll"

    def __init__(
        self,
        client: "PrinterClient",
        *,
        interval: float,
        poll: Optional[Callable[[], Awaitable[None]]] = None,
        unreachable_after: float = 300.0,
        failure_backoff: float = 10.0,
        poll_timeout: float = 30.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(client, name=name)
        self.interval = interval
        self.unreachable_after = unreachable_after
        self.failure_backoff = failure_backoff
        #: Hard bound on one ``poll_device()`` call. A poll stuck on a dead
        #: socket must count as a failure and keep the silence clock running --
        #: an unbounded poll would freeze the loop and the DOWN edge with it.
        self.poll_timeout = poll_timeout
        self._poll = poll
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self.session.reachability is DeviceReachability.STOPPED:
            return
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

    async def close(self) -> None:
        task = self._task
        self._close_session()
        self.stop()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        started = time.monotonic()
        while True:
            failed = False
            try:
                await asyncio.wait_for(
                    (self._poll or self.client.poll_device)(), self.poll_timeout
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                failed = True
                self.client.logger.debug(
                    "%s poll timed out after %.0fs", self.name, self.poll_timeout
                )
            except DeviceAuthError:
                failed = True
                self.request_credential_refresh()
            except Exception:  # noqa: BLE001 -- supervised: a bad poll backs off
                failed = True
                self.client.logger.debug("%s poll failed", self.name, exc_info=True)
            else:
                await self.set_reachability(DeviceReachability.UP)
                self.note_device_message()

            # Silence (since the last sign of life, or since start for a device
            # never reached) flips the edge exactly once until contact resumes.
            last_life = (
                self.session.observed_at
                if self.session.reachability is DeviceReachability.UP
                else started
            )
            if (
                self.session.reachability is not DeviceReachability.DOWN
                and time.monotonic() - last_life >= self.unreachable_after
            ):
                await self.set_reachability(DeviceReachability.DOWN)

            await asyncio.sleep(self.failure_backoff if failed else self.interval)
