"""Brand-owned always-on datagram listeners, hosted sans-I/O.

A brand that needs a long-lived listener the shared SSDP/mDNS backends do not
cover (a raw UDP broadcast, a proprietary multicast announcement) writes a
:class:`ListenerProtocol`: a state machine that takes events and returns
actions. It owns no socket and no clock. The host owns both:
:class:`ListenerDiscoveryBackend` binds the socket the
:class:`~simplyprint_ws_client.integration.discovery.spec.ListenerSpec`
describes, on the discovery host's shared loop, and feeds the protocol.

Events (host -> protocol): :class:`ListenerStarted` once the socket is up,
:class:`DatagramReceived` per datagram, :class:`ListenerTick` when a requested
wake-up fires. Actions (protocol -> host): :class:`SendTo` (a datagram from the
listener's socket: unicast, broadcast or multicast), :class:`WakeAfter` (one
pending timer; a later request replaces it), :class:`Sighted` (cache a record
under the spec's key and publish it like any passive listener) and
:class:`Forget` (drop a cached key).

The protocol is built fresh for every run (the host relaunches a crashed
listener), and a protocol that raises loses only the event it was handling.
:class:`ListenerSession` is the I/O-free half: tests drive it with a fake send
and a fake timer.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from simplyprint_ws_client.common.asyncio.bounded_dispatch import (
    BoundedAsyncDispatcher,
)
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.integration.discovery.multicast_base import (
    MulticastListenerBase,
)

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.discovery.spec import ListenerSpec

__all__ = [
    "DatagramReceived",
    "Forget",
    "ListenerAction",
    "ListenerDiscoveryBackend",
    "ListenerEvent",
    "ListenerProtocol",
    "ListenerSession",
    "ListenerStarted",
    "ListenerTick",
    "SendTo",
    "Sighted",
    "WakeAfter",
]


# -- events --------------------------------------------------------------------


@dataclass(frozen=True)
class ListenerStarted:
    """The listener's socket is bound; ``now`` is the host loop's clock."""

    now: float


@dataclass(frozen=True)
class ListenerTick:
    """The wake-up a :class:`WakeAfter` asked for has fired."""

    now: float


@dataclass(frozen=True)
class DatagramReceived:
    """One datagram reached the listener's socket, raw."""

    data: bytes
    source: Tuple[str, int]
    now: float


ListenerEvent = Union[ListenerStarted, ListenerTick, DatagramReceived]


# -- actions -------------------------------------------------------------------


@dataclass(frozen=True)
class SendTo:
    """Send ``data`` from the listener's socket to ``host:port``."""

    data: bytes
    host: str
    port: int


@dataclass(frozen=True)
class WakeAfter:
    """Deliver a :class:`ListenerTick` in ``seconds``; replaces a pending one."""

    seconds: float


@dataclass(frozen=True)
class Sighted:
    """A device was seen: cache ``record`` under the spec's key and publish it."""

    record: Any


@dataclass(frozen=True)
class Forget:
    """Drop the cached record under ``key`` (a device said goodbye)."""

    key: str


ListenerAction = Union[SendTo, WakeAfter, Sighted, Forget]


class ListenerProtocol(Protocol):
    """A sans-I/O listener: one event in, the actions it causes out."""

    def handle(self, event: ListenerEvent) -> Iterable[ListenerAction]: ...


# -- the I/O-free session ------------------------------------------------------


class ListenerSession:
    """Feeds one protocol instance and performs its actions through callables.

    ``send(data, (host, port))`` writes a datagram, ``wake(seconds)`` arms the
    single timer (``None`` disarms it) and must call :meth:`tick` when it
    fires, ``publish(record)`` hands a sighting on after it is cached, and
    ``clock()`` is the time stamped on each event.
    """

    def __init__(
        self,
        spec: "ListenerSpec",
        devices: ExpiringDict,
        *,
        send: Callable[[bytes, Tuple[str, int]], None],
        wake: Callable[[Optional[float]], None],
        publish: Callable[[Any], None],
        clock: Callable[[], float],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.spec = spec
        self.devices = devices
        self._send = send
        self._wake = wake
        self._publish = publish
        self._clock = clock
        self.logger = logger or logging.getLogger("discovery")
        self.protocol: ListenerProtocol = spec.protocol()

    def start(self) -> None:
        self._feed(ListenerStarted(self._clock()))

    def received(self, data: bytes, source: Tuple[Any, ...]) -> None:
        self._feed(DatagramReceived(bytes(data), (source[0], source[1]), self._clock()))

    def tick(self) -> None:
        self._feed(ListenerTick(self._clock()))

    def stop(self) -> None:
        self._wake(None)

    def _feed(self, event: ListenerEvent) -> None:
        try:
            actions = list(self.protocol.handle(event))
        except Exception:
            self.logger.debug(
                "listener protocol failed for %s", self.spec.brand, exc_info=True
            )
            return
        for action in actions:
            self._perform(action)

    def _perform(self, action: ListenerAction) -> None:
        if isinstance(action, SendTo):
            try:
                self._send(action.data, (action.host, action.port))
            except OSError:
                self.logger.debug(
                    "listener send failed for %s", self.spec.brand, exc_info=True
                )
        elif isinstance(action, WakeAfter):
            self._wake(max(0.0, float(action.seconds)))
        elif isinstance(action, Sighted):
            try:
                key = self.spec.key(action.record)
            except Exception:
                self.logger.debug(
                    "listener key failed for %s", self.spec.brand, exc_info=True
                )
                return
            self.devices[key] = action.record
            self._publish(action.record)
        elif isinstance(action, Forget):
            try:
                del self.devices[action.key]
            except KeyError:
                pass
        else:
            self.logger.warning(
                "listener for %s returned an unknown action: %r",
                self.spec.brand,
                action,
            )


# -- the hosted backend --------------------------------------------------------


class _ListenerDatagramProtocol(asyncio.DatagramProtocol):
    """Bridges the socket to a :class:`ListenerSession` on the host loop."""

    _QUEUE_MAXSIZE = 128
    _WORKERS = 2
    _IDLE_TIMEOUT = 1.0

    def __init__(self, backend: "ListenerDiscoveryBackend") -> None:
        self._backend = backend
        self._session: Optional[ListenerSession] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        self._dispatch = BoundedAsyncDispatcher(
            backend._publish,
            workers=self._WORKERS,
            maxsize=self._QUEUE_MAXSIZE,
            idle_timeout=self._IDLE_TIMEOUT,
            on_overflow=self._on_overflow,
            on_error=self._on_dispatch_error,
        )

    def connection_made(self, transport) -> None:
        self._dispatch.open()
        loop = asyncio.get_running_loop()
        self._session = ListenerSession(
            self._backend.spec,
            self._backend.devices,
            send=transport.sendto,
            wake=self._wake,
            publish=self._dispatch.submit,
            clock=loop.time,
            logger=self._backend.logger,
        )
        self._session.start()

    def _wake(self, seconds: Optional[float]) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if seconds is None or self._session is None:
            return
        self._timer = asyncio.get_running_loop().call_later(seconds, self._fire)

    def _fire(self) -> None:
        self._timer = None
        if self._session is not None:
            self._session.tick()

    def datagram_received(self, data: bytes, addr) -> None:
        if self._session is not None:
            self._session.received(data, addr)

    def _on_overflow(self) -> None:
        self._backend.logger.warning(
            "listener queue full for %s; dropping sightings", self._backend.spec.brand
        )

    def _on_dispatch_error(self, exc: Exception) -> None:
        self._backend.logger.error(
            "listener publish failed for %s", self._backend.spec.brand, exc_info=exc
        )

    def error_received(self, exc) -> None:
        if isinstance(exc, OSError) and exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return
        self._backend.logger.error(
            "listener error for %s", self._backend.spec.brand, exc_info=exc
        )

    def connection_lost(self, exc) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.stop()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._dispatch.close()
        if exc:
            self._backend.logger.error(
                "listener connection lost for %s",
                self._backend.spec.brand,
                exc_info=exc,
            )


class ListenerDiscoveryBackend(MulticastListenerBase):
    """One brand-owned always-on listener, run by the discovery host.

    Binds ``spec.port`` (``0``: an ephemeral port, for a listener that only
    hears replies to its own sends), joins ``spec.group`` when set, and allows
    broadcast sends when ``spec.broadcast``. Caches sightings with the same TTL
    as the other passive listeners, so ``DiscoveryService.snapshot`` reads it.
    """

    spec: "ListenerSpec"

    listen_label = "listener discovery"
    port_label = "listener"

    def _make_socket(self) -> socket.socket:
        sock = super()._make_socket()
        if self.spec.broadcast:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        return sock

    def protocol_factory(self) -> _ListenerDatagramProtocol:
        return _ListenerDatagramProtocol(self)

    async def bind_socket(self, sock: socket.socket) -> None:
        if not self.spec.port:
            sock.bind(("", 0))
            return
        await self.bind_fixed_port(sock)

    def join_group(self, sock: socket.socket) -> None:
        if self.spec.group is None:
            return
        try:
            self.join_multicast_group(sock)
        except OSError:
            self.logger.warning(
                "could not join multicast group for %s - group traffic disabled",
                self.spec.brand,
            )

    async def run_transport(self, transport: asyncio.DatagramTransport) -> None:
        await self.wait()
