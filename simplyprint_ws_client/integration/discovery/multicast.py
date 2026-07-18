"""Generic UDP multicast discovery backend.

Drives one brand's :class:`~simplyprint_ws_client.integration.discovery.spec.MulticastSpec`:
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
import time

from simplyprint_ws_client.common.asyncio.bounded_dispatch import (
    BoundedAsyncDispatcher,
)
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.integration.discovery.multicast_base import (
    MulticastListenerBase,
)
from simplyprint_ws_client.integration.discovery.spec import MulticastSpec
from simplyprint_ws_client.integration.discovery.ssdp import SSDPRequestParser


class _MulticastProtocol(asyncio.DatagramProtocol):
    """Parses datagrams, caches mapped devices, and fans out the spec's event."""

    _QUEUE_MAXSIZE = 128
    _WORKERS = 2
    _IDLE_TIMEOUT = 1.0

    def __init__(self, spec: MulticastSpec, devices: ExpiringDict, emit, logger):
        self._spec = spec
        self._devices = devices
        self._emit = emit  # async callable(record) | None
        self._logger = logger
        self._dispatch = BoundedAsyncDispatcher(
            self._handle,
            workers=self._WORKERS,
            maxsize=self._QUEUE_MAXSIZE,
            idle_timeout=self._IDLE_TIMEOUT,
            on_overflow=self._on_overflow,
            on_error=self._on_dispatch_error,
        )

    def connection_made(self, _transport) -> None:
        self._dispatch.open()

    def datagram_received(self, data: bytes, addr) -> None:
        request = SSDPRequestParser.parse(data)
        if request is None:
            return

        self._dispatch.submit((request, addr))

    def _on_overflow(self) -> None:
        self._logger.warning(
            "discovery queue full for %s; dropping datagrams",
            self._spec.brand,
        )

    def _on_dispatch_error(self, exc: Exception) -> None:
        self._logger.error(
            "discovery handler failed for %s", self._spec.brand, exc_info=exc
        )

    async def _handle(self, item) -> None:
        request, addr = item
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
        self._dispatch.close()
        if exc:
            self._logger.error(
                "multicast connection lost for %s", self._spec.brand, exc_info=exc
            )


class MulticastDiscoveryBackend(MulticastListenerBase):
    """One always-on multicast listener (passive or active) for a single brand."""

    spec: MulticastSpec

    def protocol_factory(self) -> _MulticastProtocol:
        return _MulticastProtocol(self.spec, self.devices, self._emit, self.logger)

    async def bind_socket(self, sock: socket.socket) -> None:
        # Active specs search from an ephemeral port; passive specs bind the fixed
        # announcement port and retry while it is briefly in use.
        if self.spec.is_active:
            sock.bind(("", 0))
            return

        await self.bind_fixed_port(sock)

    def join_group(self, sock: socket.socket) -> None:
        if self.spec.is_active:
            # An active spec still works via unicast search responses if the join
            # fails (e.g. no multicast route); don't let that kill the listener.
            try:
                self.join_multicast_group(sock)
            except OSError:
                self.logger.warning(
                    "could not join multicast group for %s - passive announcements disabled",
                    self.spec.brand,
                )
        else:
            self.join_multicast_group(sock)

    async def run_transport(self, transport: asyncio.DatagramTransport) -> None:
        if self.spec.is_active:
            while not self.is_stopped():
                transport.sendto(
                    self.spec.search_payload, (self.spec.group, self.spec.port)
                )
                await self.wait(self.spec.search_interval)
        else:
            await self.wait()


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
