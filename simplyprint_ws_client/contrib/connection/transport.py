"""The transport contract: ask for a link to an endpoint, get a handle you drive
by *events* -- never a thread.

This module is the **contract surface** a wire implements and a consumer programs
against; the pooling/routing machinery that fulfils it lives in :mod:`.pool`, the
event vocabulary in :mod:`.events`, the cross-thread hop in :mod:`.loop`. A
**transport** is a supervised, self-healing link to ONE endpoint (an MQTT broker,
a WebSocket host, ...). It owns the whole reliability story -- connect, reconnect
with backoff, liveness, crash recovery, state -- so no consumer writes that twice
and no consumer ever touches a thread.

Two separations are deliberate:

* **Where the work runs is the transport's private business.** paho drives its
  own network thread; a websocket-client wire supervises a daemon thread; an
  asyncio wire runs on the harness loop. None of that leaks across the seam: a
  consumer only subscribes to :class:`TransportEvent` s and calls ``send``. That
  is what lets one brand's code sit on a threaded paho transport *or* an async
  WebSocket transport unchanged -- the event surface is the single language. It
  is also what "the harness owns the threads" means: the library may spawn and
  supervise threads inside a transport; an integration never does.

* **Sync vs async is a real, kept distinction -- not a wart to paper over.**
  :class:`AsyncTransport` is the ideal (``await send``) for asyncio-native wires
  (``websockets``/aiohttp). :class:`Transport` is the synchronous form for wires
  that are inherently thread-driven (paho), where wrapping them in async buys
  nothing and costs clarity and performance. Both emit the *same* events, so a
  consumer that only listens is agnostic to which it got.

**Pooling is a capability layered on top** (see :class:`Pool` / :class:`AsyncPool`,
implemented in :mod:`.pool`): the same transport contract is handed out *shared by
endpoint* (many printers, one broker socket -- the performance win) or *dedicated*
(the backend's 1:1 socket). A dedicated transport is just a pool of one. The front
door is :meth:`Pool.connect`: an endpoint in, a :class:`Lease` out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Hashable, Optional, TypeVar

from simplyprint_ws_client.events import EventBus

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    ConnectionSuspect,
    Disconnected,
    MessageReceived,
    StateChanged,
    TransportEvent,
)
from simplyprint_ws_client.contrib.connection.loop import CoroFactory
from simplyprint_ws_client.contrib.connection.state import ConnectionState

__all__ = [
    # event vocabulary (re-exported from .events for one import site)
    "TransportEvent",
    "Connected",
    "Disconnected",
    "MessageReceived",
    "ConnectionSuspect",
    "StateChanged",
    # transport contract
    "BaseTransport",
    "Transport",
    "AsyncTransport",
    # lease contract
    "BaseLease",
    "Lease",
    "AsyncLease",
    "LeaseHandler",
    "Unsubscribe",
    # pool contract
    "Pool",
    "AsyncPool",
]

TParams = TypeVar("TParams", bound=Hashable)


class BaseTransport(ABC, Generic[TParams]):
    """A supervised link to one endpoint, driven by events.

    Subclasses own the wire and the reliability loop (connect, reconnect with
    backoff, liveness, crash recovery) and publish :class:`TransportEvent` s on
    :attr:`events`. The execution model -- a library thread, a supervised daemon
    thread, an asyncio task on the harness loop -- is the subclass's private
    business and must never leak to consumers.

    ``params`` is the endpoint identity and MUST be hashable: it is the key a
    :class:`Pool` shares transports by.
    """

    #: Hashable endpoint identity (broker tuple, ws url, ...). The pool key.
    params: TParams
    #: Where consumers subscribe (``Connected`` / ``Disconnected`` /
    #: ``MessageReceived`` / ``ConnectionSuspect`` / ``StateChanged``).
    events: EventBus[TransportEvent]
    #: Last-known reachability; also published via :class:`StateChanged`.
    state: ConnectionState
    #: Monotonic connection epoch, read-only to consumers. Changes once per
    #: (re)connect so a consumer can tell one live link from the next (e.g. to
    #: drop work queued for a link that has since dropped). A self-healing wire
    #: bumps it on each connect; a :class:`~.reconnect.ReconnectingTransport` wire bumps it
    #: once per dropped attempt. Default 0 (never connected).
    generation: int = 0

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the underlying wire currently holds a live connection."""

    @abstractmethod
    def start(self) -> None:
        """Begin supervising the link (connect, then auto-reconnect). Idempotent.

        For an async transport this schedules the loop task on the harness event
        loop; for a sync transport it starts the library/supervisor thread. Either
        way it returns immediately -- readiness is reported via :class:`Connected`.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear the link down permanently and release resources. Idempotent."""

    def subscribe(self, topic: str) -> None:
        """Register interest in ``topic`` -- topic-routed wires only.

        A 1:1 wire (WebSocket) has no topics and keeps this no-op; a broker
        transport (MQTT) overrides it to (re)assert the subscription on the
        shared socket. Concrete (not abstract) so a lease can always call it."""

    def unsubscribe(self, topic: str) -> None:
        """Drop interest in ``topic`` -- the no-op/override mirror of :meth:`subscribe`."""

    def _emit(self, event: TransportEvent) -> None:
        """Publish ``event`` on :attr:`events`.

        Keyed by the event's *type*, so a consumer subscribes with
        ``transport.events.on(Connected, handler)`` and ``handler`` receives the
        typed event instance. Every transport publishes through this one seam so
        the event surface stays identical across the sync and async families.
        """
        self.events.emit_sync(type(event), event)


class Transport(BaseTransport[TParams]):
    """A **synchronous** transport for inherently thread-driven wires (paho,
    websocket-client).

    It owns and supervises its own thread(s) internally; the caller only
    ``send`` s and listens on :attr:`events`. ``send`` is best-effort and never
    blocks on reconnect.
    """

    @abstractmethod
    def send(self, payload: Any) -> bool:
        """Send ``payload`` over the wire. Returns ``False`` if the link is down."""


class AsyncTransport(BaseTransport[TParams]):
    """An **asynchronous** transport for asyncio-native wires (``websockets``,
    aiohttp), driven on the harness event loop."""

    @abstractmethod
    async def send(self, payload: Any) -> None:
        """Send ``payload`` over the wire; raises if the link is unusable."""


TTransport = TypeVar("TTransport", bound=BaseTransport)


class Pool(ABC, Generic[TParams, TTransport]):
    """Pooling for synchronous transports.

    Hands out ONE :class:`Transport` per endpoint (keyed by ``params``), shared
    by every client that resolves to the same endpoint, and owns the
    sharing/routing/keepalive/registration lifecycle. A *dedicated* link is just
    a pool entry with a single client.

    The front door is :meth:`connect`: it hands back a :class:`Lease`
    -- a per-client view that ``send`` s, subscribes to *its* messages, and
    ``close`` s its own hold. The implementation is :class:`.pool.TransportPool`.
    """

    @abstractmethod
    def connect(self, params: TParams, *, route: Optional[Hashable] = None) -> "Lease":
        """Lease the shared transport for ``params`` as a :class:`Lease`.

        ``route`` (e.g. an MQTT topic) scopes which inbound messages this lease
        receives; ``None`` (the dedicated/1:1 case) receives them all.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear down the pool: close every transport and release resources."""

    @abstractmethod
    def submit_to_loop(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        """Run a coroutine on the pool loop from a transport thread, coalesced by
        ``coalesce_key`` (the one sanctioned cross-thread hop; see :class:`.loop.LoopBridge`)."""

    @abstractmethod
    def call_on_loop(self, fn: Callable[[], None]) -> None:
        """Run a sync callable on the pool loop -- event-bus delivery from a worker
        or transport thread, without a per-call future."""


class AsyncPool(ABC, Generic[TParams, TTransport]):
    """Pooling for asynchronous transports -- the async sibling of :class:`Pool`
    (implemented by :class:`.pool.AsyncTransportPool`)."""

    @abstractmethod
    async def connect(
        self, params: TParams, *, route: Optional[Hashable] = None
    ) -> "AsyncLease":
        """Lease the shared async transport for ``params`` as an
        :class:`AsyncLease`. See :meth:`Pool.connect`."""


#: A handler for a connection event. May be sync, or return a coroutine that is
#: scheduled on the pool loop. Receives the event payload: ``on_message`` ->
#: the wire message; ``on_connected`` -> nothing; ``on_disconnected`` -> the
#: :class:`Disconnected` event.
LeaseHandler = Callable[..., Any]
Unsubscribe = Callable[[], None]


class BaseLease(ABC, Generic[TParams]):
    """A per-client lease over a (possibly shared) transport.

    A consumer never touches the transport's raw event bus or its thread; it
    drives THIS handle: ``send``, subscribe to *its own* messages via
    :meth:`on_message`, and :meth:`close` its hold. The pool routes a shared
    transport's messages to the right lease(s) by ``route`` -- so a client no
    longer sees (and filters) every other client's traffic.
    """

    params: TParams

    @abstractmethod
    def on_message(self, handler: LeaseHandler) -> Unsubscribe:
        """Receive this lease's inbound messages. Returns an unsubscribe callable."""

    @abstractmethod
    def on_connected(self, handler: LeaseHandler) -> Unsubscribe:
        """Called when the shared link comes up. Returns an unsubscribe callable."""

    @abstractmethod
    def on_disconnected(self, handler: LeaseHandler) -> Unsubscribe:
        """Called when the shared link drops. Returns an unsubscribe callable."""

    @abstractmethod
    def on_suspect(self, handler: LeaseHandler) -> Unsubscribe:
        """Called on a suspect connect (repeated failure / CONNACK reject). The
        handler receives the :class:`ConnectionSuspect` event. Returns an unsubscribe."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the underlying shared transport currently holds a live link."""

    @abstractmethod
    def subscribe(self, topic: str) -> None:
        """Register interest in ``topic`` on a shared transport (no-op on a 1:1)."""

    @abstractmethod
    def submit_to_loop(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        """Run async work on the pool's event loop from a transport thread.

        The one sanctioned cross-thread coroutine hop -- the pool owns the loop,
        so the caller never reaches for one. ``coalesce_key`` makes a second
        submission while one is in flight a no-op (replacing hand-rolled
        in-flight guards). Returns immediately; failures are logged by the pool.
        """


class Lease(BaseLease[TParams]):
    """A **synchronous** lease (paho family)."""

    @abstractmethod
    def send(self, payload: Any) -> bool:
        """Send ``payload`` over the shared link. ``False`` if the link is down."""

    @abstractmethod
    def close(self) -> None:
        """Release this lease; the pool tears the transport down with the last."""


class AsyncLease(BaseLease[TParams]):
    """An **asynchronous** lease (``websockets`` / aiomqtt family)."""

    @abstractmethod
    async def send(self, payload: Any) -> None:
        """Send ``payload`` over the shared link; raises if unusable."""

    @abstractmethod
    async def close(self) -> None:
        """Release this lease; the pool tears the transport down with the last."""
