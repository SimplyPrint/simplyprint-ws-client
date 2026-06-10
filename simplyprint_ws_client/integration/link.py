"""Pooled, self-healing transport leases bridged onto a printer client.

This is the class four integrations used to hand-copy (~100 lines each): lease
the shared pool, bind the wire events, hop them onto the client, guard sends
while the link is down, keep the tri-state liveness, restart with fresh
credentials. ``url`` (and ``topics``) are callables so a
:meth:`~simplyprint_ws_client.integration.driver.DeviceDriver.restart` after a
credential refresh re-resolves them from the (just updated) config.

The lease's courier already delivers events on the client's loop, so the link
calls the client hooks directly — there is no second event bus and no per-brand
event classes.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import yarl

from simplyprint_ws_client.common.wire.errors import FatalError
from simplyprint_ws_client.common.wire.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.common.wire.messages import MqttMessage, WsMessage
from simplyprint_ws_client.common.wire import mqtt as mqtt_front_door
from simplyprint_ws_client.common.wire import websocket as ws_front_door
from simplyprint_ws_client.common.wire.lease import Lease, MqttLease, WsLease
from simplyprint_ws_client.common.wire.options import ConnectionOptions
from simplyprint_ws_client.integration.driver import DeviceDriver

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.client import PrinterClient

__all__ = ["DeviceLink", "WsDeviceLink", "MqttDeviceLink"]

#: Resolves the device endpoint at (re)start time -- a callable so a restart
#: after a credential refresh picks up the just-updated config.
UrlFactory = Callable[[], Union[str, yarl.URL]]


class DeviceLink(DeviceDriver):
    """The shared lease-bridge lifecycle; concrete links pick the front door."""

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
        self.lease: Optional[Lease] = None

    def _connect(self, url: Union[str, yarl.URL], options: ConnectionOptions) -> Lease:
        raise NotImplementedError

    def _on_lease_acquired(self, lease: Lease) -> None:
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


class WsDeviceLink(DeviceLink):
    """A 1:1 WebSocket attachment: every frame is this client's.

    Inbound frames reach ``on_device_message`` as their raw payload
    (``str``/``bytes``) — the unwrap four brands wrote defensively is owned here.
    """

    lease: Optional[WsLease]

    def _connect(
        self, url: Union[str, yarl.URL], options: ConnectionOptions
    ) -> WsLease:
        return ws_front_door.connect(url, options=options)

    @staticmethod
    def _payload(message: object) -> object:
        return message.payload if isinstance(message, WsMessage) else message


class MqttDeviceLink(DeviceLink):
    """A broker attachment: topics multiplexed over one shared socket.

    ``topics`` resolves at (re)start so a restart re-subscribes against the
    fresh config. Inbound messages reach ``on_device_message`` as
    :class:`~simplyprint_ws_client.common.wire.messages.MqttMessage` (topic +
    payload — the topic is routing information the client needs).
    """

    lease: Optional[MqttLease]

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

    def _on_lease_acquired(self, lease: Lease) -> None:
        for topic in self._topics():
            lease.subscribe_soon(topic)

    def publish_soon(self, message: MqttMessage) -> bool:
        """Schedule a publish if the wire is up; ``False`` if not."""
        return self.send_soon(message)
