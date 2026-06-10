"""Generic UDP multicast discovery backend.

Drives one brand's :class:`~simplyprint_ws_client.device.discovery.spec.MulticastSpec`:
bind a multicast socket, parse each datagram with the shared
:class:`SSDPRequestParser`, map it to a device record, cache it with a TTL, and
emit the spec's event. The backend holds no brand knowledge -- group/port,
optional search payload, header mapping and cache key all come from the spec.

A *passive* spec (no ``search_payload``) only listens for announcements; an
*active* spec additionally sends periodic searches. :meth:`run` is a long-lived
coroutine; the harness runs every backend's ``run()`` concurrently on one shared
loop (see :class:`DiscoveryServiceHost`), so no backend owns a thread of its own.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import struct
import time

from simplyprint_ws_client.common.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable

from simplyprint_ws_client.device.discovery.spec import MulticastSpec
from simplyprint_ws_client.device.discovery.ssdp import SSDPRequestParser
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict

_DEVICE_TTL = 300


class _MulticastProtocol(asyncio.DatagramProtocol):
    """Parses datagrams, caches mapped devices, and fans out the spec's event."""

    def __init__(self, spec: MulticastSpec, devices: ExpiringDict, emit, logger):
        self._spec = spec
        self._devices = devices
        self._emit = emit  # async callable(record) | None
        self._logger = logger

    def datagram_received(self, data: bytes, addr) -> None:
        request = SSDPRequestParser.parse(data)
        if request is None:
            return

        asyncio.create_task(self._handle(request, addr))

    async def _handle(self, request, addr) -> None:
        try:
            record = self._spec.mapper(request, addr)
        except Exception:
            self._logger.debug(
                "discovery mapper failed for %s", self._spec.brand, exc_info=True
            )
            return

        if record is None:
            return

        self._devices[self._spec.key(record)] = record

        if self._emit is not None:
            await self._emit(record)

    def error_received(self, exc) -> None:
        if isinstance(exc, OSError) and exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return
        self._logger.error("multicast error for %s", self._spec.brand, exc_info=exc)

    def connection_lost(self, exc) -> None:
        if exc:
            self._logger.error(
                "multicast connection lost for %s", self._spec.brand, exc_info=exc
            )


class MulticastDiscoveryBackend(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop]
):
    """One always-on multicast listener (passive or active) for a single brand."""

    def __init__(self, spec: MulticastSpec, event_bus: EventBus) -> None:
        AsyncStoppable.__init__(self)
        EventLoopProvider.__init__(self)

        self.spec = spec
        self.event_bus = event_bus
        self.logger = logging.getLogger("discovery")
        self.devices = ExpiringDict(ttl=_DEVICE_TTL)
        # blocking=True -> the async ``emit`` (the wrapped callable is awaited in
        # the backend loop); identical to the per-brand services this replaces.
        self._emit = event_bus.emit_wrap(spec.event_type, blocking=True)

    def get_devices(self) -> list:
        return self.devices.values()

    def _protocol_factory(self) -> _MulticastProtocol:
        return _MulticastProtocol(self.spec, self.devices, self._emit, self.logger)

    def _make_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        if hasattr(socket, "SO_REUSEADDR"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if self.spec.multicast_ttl is not None:
            sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.spec.multicast_ttl
            )
        sock.setblocking(False)
        return sock

    async def _bind(self, sock: socket.socket) -> None:
        # Active specs search from an ephemeral port; passive specs bind the fixed
        # announcement port and retry while it is briefly in use (e.g. a previous
        # instance still tearing down).
        if self.spec.is_active:
            sock.bind(("", 0))
            return

        while not self.is_stopped():
            try:
                sock.bind(("", self.spec.port))
                return
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    self.logger.warning(
                        "discovery port %s in use - retrying in 5s", self.spec.port
                    )
                    await self.wait(5)
                else:
                    raise

    def _join_group(self, sock: socket.socket) -> None:
        group = socket.inet_aton(self.spec.group)
        mreq = struct.pack("4sL", group, socket.INADDR_ANY)
        if self.spec.is_active:
            # An active spec still works via unicast search responses if the join
            # fails (e.g. no multicast route); don't let that kill the listener.
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            except OSError:
                self.logger.warning(
                    "could not join multicast group for %s - passive announcements disabled",
                    self.spec.brand,
                )
        else:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    async def run(self) -> None:
        self.use_running_loop()

        sock = self._make_socket()
        await self._bind(sock)

        if self.is_stopped():
            sock.close()
            return

        self._join_group(sock)

        transport, _ = await self.event_loop.create_datagram_endpoint(
            self._protocol_factory, sock=sock
        )
        self.logger.info(
            "discovery listening for %s on %s:%s",
            self.spec.brand,
            self.spec.group,
            self.spec.port,
        )

        try:
            if self.spec.is_active:
                while not self.is_stopped():
                    transport.sendto(
                        self.spec.search_payload, (self.spec.group, self.spec.port)
                    )
                    await self.wait(self.spec.search_interval)
            else:
                await self.wait()
        finally:
            transport.close()

        self.logger.info("discovery for %s stopping", self.spec.brand)

    def stop(self) -> None:
        if self.is_stopped():
            return

        if self.event_loop_is_running():
            self.event_loop.call_soon_threadsafe(super().stop)
        else:
            super().stop()


def scan_blocking(spec: MulticastSpec, timeout: float = 5.0) -> list:
    """One-shot synchronous active scan for a standalone CLI (no DiscoveryService).

    Sends the spec's search payload once, collects unicast responses for
    ``timeout`` seconds, and returns the deduplicated mapped devices. Intended for
    per-brand setup CLIs that run in their own process; the always-on listener in
    the unified process uses :class:`MulticastDiscoveryBackend` instead.
    """
    logger = logging.getLogger("discovery")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if spec.multicast_ttl is not None:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, spec.multicast_ttl)
    sock.settimeout(1.0)

    if spec.search_payload:
        sock.sendto(spec.search_payload, (spec.group, spec.port))

    devices: dict = {}
    deadline = time.monotonic() + timeout

    try:
        while time.monotonic() < deadline:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            request = SSDPRequestParser.parse(data)
            if request is None:
                continue

            try:
                record = spec.mapper(request, addr)
            except Exception:
                logger.debug(
                    "discovery mapper failed for %s", spec.brand, exc_info=True
                )
                continue

            if record is not None:
                devices[spec.key(record)] = record
    finally:
        sock.close()

    return list(devices.values())
