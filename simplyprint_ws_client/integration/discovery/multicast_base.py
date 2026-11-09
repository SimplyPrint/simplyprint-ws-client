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
import contextlib
import errno
import logging
import socket
import sys
from abc import abstractmethod
from typing import Callable, Iterable, Optional, Tuple

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable
from simplyprint_ws_client.integration.discovery.netif import multicast_interfaces

#: How long a discovered device lingers in the cache after it was last seen.
DEVICE_TTL = 300

#: How often a listener compares the interfaces it joined with the machine's: a VPN,
#: a second adapter or another Wi-Fi network after sleep shows up after start.
MEMBERSHIP_REFRESH = 30

#: "Already a member on this interface" (Windows reports its own code).
_ALREADY_JOINED = {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)}

_ANY_INTERFACE = socket.inet_aton("0.0.0.0")


def send_from_interfaces(
    sock: socket.socket,
    payload: bytes,
    address: Tuple[str, int],
    interfaces: Iterable[str],
    logger: logging.Logger,
) -> None:
    """Send one copy of ``payload`` out of each interface (by its IPv4 address), or
    one copy out of the OS default interface when there are none. A failure on one
    interface (a full buffer, an adapter going away) never stops the others."""
    targets: list[Optional[str]] = sorted(interfaces) or [None]
    for interface in targets:
        try:
            if interface is not None:
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_MULTICAST_IF,
                    socket.inet_aton(interface),
                )
            sock.sendto(payload, address)
        except OSError:
            logger.debug(
                "multicast send to %s via %s failed",
                address,
                interface or "the default interface",
                exc_info=True,
            )


class GroupMembership:
    """One socket's membership of a multicast group, on every IPv4 interface.

    ``IP_ADD_MEMBERSHIP`` with ``INADDR_ANY`` lets the OS pick a single interface,
    the default route's. When that is a VPN, a Hyper-V/WSL adapter or the other of
    two connections, announcements on the network the printers are on are never
    delivered. Joining by each interface's address hears them all; ``INADDR_ANY``
    stays the fallback for a machine whose interfaces can't be joined or listed.
    """

    def __init__(
        self,
        group: str,
        logger: logging.Logger,
        interfaces: Callable[[], Iterable[str]] = multicast_interfaces,
    ) -> None:
        self._group = socket.inet_aton(group)
        self._group_name = group
        self._logger = logger
        self._interfaces = interfaces
        #: Interface addresses this socket is a member on.
        self.joined: set[str] = set()
        #: Joined through the OS default interface (the fallback).
        self.default_joined = False

    def refresh(self, sock: socket.socket) -> bool:
        """Join every current interface not joined yet and forget vanished ones (the
        OS drops a membership with its interface). Returns whether that changed the
        joined set. Raises ``OSError`` only when nothing, not even the default
        interface, could be joined."""
        current = set(self._interfaces())
        before = set(self.joined)
        for address in sorted(current - self.joined):
            try:
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_ADD_MEMBERSHIP,
                    self._group + socket.inet_aton(address),
                )
            except OSError as exc:
                if exc.errno not in _ALREADY_JOINED:
                    self._logger.debug(
                        "could not join %s on %s: %s", self._group_name, address, exc
                    )
                    continue
            self.joined.add(address)
        self.joined &= current

        if not self.joined and not self.default_joined:
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                self._group + _ANY_INTERFACE,
            )
            self.default_joined = True

        return self.joined != before

    def send(
        self, sock: socket.socket, payload: bytes, address: Tuple[str, int]
    ) -> None:
        """Send to the group out of every joined interface."""
        send_from_interfaces(sock, payload, address, self.joined, self._logger)


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
        self._sock: socket.socket | None = None
        self._membership: GroupMembership | None = None
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
        """Join the group on every interface (see :class:`GroupMembership`); raises
        ``OSError`` when no interface, not even the default, could be joined. The
        listener keeps the membership current while it runs."""
        self._membership = GroupMembership(self.spec.group, self.logger)
        self._membership.refresh(sock)

    def send_to_group(
        self, transport: asyncio.DatagramTransport, payload: bytes
    ) -> None:
        """Send ``payload`` to the spec's group out of every joined interface. Before
        the listener runs (or without a membership) it goes through ``transport``."""
        address = (self.spec.group, self.spec.port)
        if self._sock is None or self._membership is None:
            transport.sendto(payload, address)
            return
        self._membership.send(self._sock, payload, address)

    async def _keep_membership_current(self, sock: socket.socket) -> None:
        while not await self.wait(MEMBERSHIP_REFRESH):
            if self._membership is None:
                continue
            try:
                changed = self._membership.refresh(sock)
            except OSError:
                self.logger.debug(
                    "%s for %s: no interface could join %s",
                    self.listen_label,
                    self.spec.brand,
                    self.spec.group,
                    exc_info=True,
                )
                continue
            if changed:
                self.logger.info(
                    "%s for %s now on %s",
                    self.listen_label,
                    self.spec.brand,
                    ", ".join(sorted(self._membership.joined))
                    or "the default interface",
                )

    async def run(self) -> None:
        self.use_running_loop()

        sock = self._make_socket()
        transport: asyncio.DatagramTransport | None = None
        try:
            await self.bind_socket(sock)

            if self.is_stopped():
                return

            self.join_group(sock)
            self._sock = sock

            transport, _ = await self.event_loop.create_datagram_endpoint(
                self.protocol_factory, sock=sock
            )
            interfaces = (
                ", ".join(sorted(self._membership.joined))
                if self._membership is not None and self._membership.joined
                else "the default interface"
            )
            self.logger.info(
                "%s listening for %s on %s:%s (%s)",
                self.listen_label,
                self.spec.brand,
                self.spec.group,
                self.spec.port,
                interfaces,
            )
            keeper = self.event_loop.create_task(self._keep_membership_current(sock))
            try:
                await self.run_transport(transport)
            finally:
                keeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keeper
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
            self._sock = None
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
