"""Shared skeleton for the always-on multicast listener backends.

The UDP multicast (SSDP) and mDNS backends are the same machine: bind one
multicast socket on the harness's shared loop, join the group, hand datagrams
to a parsing protocol that caches mapped devices in a TTL dict and fans out the
spec's event. This base owns that skeleton once -- socket setup, the
fixed-port bind retry, group join, the run template, the device cache and the
``stop`` hop -- while each backend supplies its bind policy, group-join policy
and send loop. No brand knowledge lives here; everything comes from the spec.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import struct
import sys
from abc import abstractmethod

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable

#: How long a discovered device lingers in the cache after it was last seen.
DEVICE_TTL = 300


class MulticastListenerBase(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop]
):
    """One always-on multicast listener for a single brand spec."""

    #: Label used in the listening/stopping log lines.
    listen_label = "discovery"
    #: Label used in the bind-retry warning.
    port_label = "discovery"

    def __init__(self, spec, event_bus: EventBus) -> None:
        AsyncStoppable.__init__(self)
        EventLoopProvider.__init__(self)

        self.spec = spec
        self.event_bus = event_bus
        self.logger = logging.getLogger("discovery")
        self.devices = ExpiringDict(ttl=DEVICE_TTL)
        # blocking=True -> the async ``emit`` (the wrapped callable is awaited in
        # the backend loop); identical to the per-brand services this replaces.
        self._emit = event_bus.emit_wrap(spec.event_type, blocking=True)

    def get_devices(self) -> list:
        return self.devices.values()

    @abstractmethod
    def protocol_factory(self) -> asyncio.DatagramProtocol: ...

    @abstractmethod
    async def bind_socket(self, sock: socket.socket) -> None:
        """Bind the socket (fixed announcement port or ephemeral, per backend)."""

    @abstractmethod
    def join_group(self, sock: socket.socket) -> None:
        """Join the multicast group with the backend's failure policy."""

    @abstractmethod
    async def run_transport(self, transport: asyncio.DatagramTransport) -> None:
        """The backend's long-lived send/wait loop (until stopped)."""

    def _make_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if sys.platform != "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if self.spec.multicast_ttl is not None:
            sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.spec.multicast_ttl
            )
        sock.setblocking(False)
        return sock

    async def bind_fixed_port(self, sock: socket.socket) -> None:
        """Bind the spec's fixed port, retrying while it is briefly in use
        (e.g. a previous instance still tearing down)."""
        while not self.is_stopped():
            try:
                sock.bind(("", self.spec.port))
                return
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    self.logger.warning(
                        "%s port %s in use - retrying in 5s",
                        self.port_label,
                        self.spec.port,
                    )
                    await self.wait(5)
                else:
                    raise

    def join_multicast_group(self, sock: socket.socket) -> None:
        """Issue the IP_ADD_MEMBERSHIP join (raises ``OSError`` on failure)."""
        group = socket.inet_aton(self.spec.group)
        mreq = struct.pack("4sL", group, socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    async def run(self) -> None:
        self.use_running_loop()

        sock = self._make_socket()
        transport: asyncio.DatagramTransport | None = None
        try:
            await self.bind_socket(sock)

            if self.is_stopped():
                return

            self.join_group(sock)

            transport, _ = await self.event_loop.create_datagram_endpoint(
                self.protocol_factory, sock=sock
            )
            self.logger.info(
                "%s listening for %s on %s:%s",
                self.listen_label,
                self.spec.brand,
                self.spec.group,
                self.spec.port,
            )
            await self.run_transport(transport)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EPERM, 10013}:
                raise
            if not self.is_stopped():
                self.logger.warning(
                    "%s backend %s disabled: OS denied UDP multicast socket "
                    "access on port %s; continuing without this listener",
                    self.listen_label,
                    self.spec.brand,
                    self.spec.port,
                )
                self.logger.debug("multicast socket access denied", exc_info=True)
                await self.wait()
        finally:
            if transport is None:
                sock.close()
            else:
                transport.close()

        self.logger.info("%s for %s stopping", self.listen_label, self.spec.brand)

    def stop(self) -> None:
        if self.is_stopped():
            return

        if self.event_loop_is_running():
            self.event_loop.call_soon_threadsafe(super().stop)
        else:
            super().stop()
