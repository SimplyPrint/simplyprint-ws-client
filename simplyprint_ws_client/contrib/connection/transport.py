"""The transport contract: ask for a link to an endpoint, get a handle you drive
by *events* -- never a thread.

This is the heart of the connection subsystem. A **transport** is a supervised,
self-healing link to ONE endpoint (an MQTT broker, a WebSocket host, ...). It
owns the whole reliability story -- connect, reconnect with backoff, liveness,
crash recovery, state -- so no consumer writes that twice and no consumer ever
touches a thread.

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

**Pooling is a capability layered on top** (see :class:`Pool` /
:class:`AsyncPool`): the same transport contract is handed out *shared by
endpoint* (many printers, one broker socket -- the performance win) or
*dedicated* (the backend's 1:1 socket). A dedicated transport is just a pool of
one.

The front door is :func:`connect`: "connect me to MQTT / connect me to WS" in,
a transport interface out.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Hashable,
    List,
    Optional,
    Set,
    TypeVar,
)

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.state import ConnectionState

__all__ = [
    "TransportEvent",
    "Connected",
    "Disconnected",
    "MessageReceived",
    "ConnectSuspect",
    "StateChanged",
    "BaseTransport",
    "Transport",
    "AsyncTransport",
    "BaseConnection",
    "Connection",
    "AsyncConnection",
    "ConsumerLoop",
    "TransportRouter",
    "Pool",
    "AsyncPool",
    "Wire",
]


@dataclass(frozen=True)
class TransportEvent:
    """Base of every event a transport publishes on its ``events`` bus."""


@dataclass(frozen=True)
class Connected(TransportEvent):
    """The link is up (first connect or a recovery)."""


@dataclass(frozen=True)
class Disconnected(TransportEvent):
    """The link went down.

    ``transient`` marks a drop the transport is already recovering from on its
    own -- a consumer can tolerate it until the failures pile up, rather than
    treating every blip as a hard disconnect.
    """

    reason: str = ""
    transient: bool = False


@dataclass(frozen=True)
class MessageReceived(TransportEvent):
    """An inbound message.

    ``payload`` is wire-shaped -- text for a WebSocket, the broker message for
    MQTT. The transport does not parse brand protocol; routing a pooled
    endpoint's message to the right client is the :class:`Pool`'s job.
    """

    payload: Any


@dataclass(frozen=True)
class ConnectSuspect(TransportEvent):
    """Repeated connect failures: the endpoint may be unreachable. Advisory --
    the transport keeps retrying regardless."""

    error: Optional[BaseException] = None


@dataclass(frozen=True)
class StateChanged(TransportEvent):
    """The transport's :class:`ConnectionState` changed."""

    state: ConnectionState


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
    #: ``MessageReceived`` / ``ConnectSuspect`` / ``StateChanged``).
    events: EventBus[TransportEvent]
    #: Last-known reachability; also published via :class:`StateChanged`.
    state: ConnectionState

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
    a pool entry with a single client. This is the capability today's
    ``ConnectionManager`` provides, named for what it is.

    The front door is :meth:`connect`: it hands back a :class:`Connection` lease
    -- a per-client view that ``send`` s, subscribes to *its* messages, and
    ``close`` s its own hold. :meth:`acquire` / :meth:`release` are the lower-level
    shared-transport mechanism the lease is built on.
    """

    @abstractmethod
    def connect(
        self, params: TParams, *, route: Optional[Hashable] = None
    ) -> Connection:
        """Lease the shared transport for ``params`` as a :class:`Connection`.

        ``route`` (e.g. an MQTT topic) scopes which inbound messages this lease
        receives; ``None`` (the dedicated/1:1 case) receives them all.
        """

    @abstractmethod
    def acquire(self, params: TParams) -> TTransport:
        """Reuse or create the shared transport for ``params``."""

    @abstractmethod
    def release(self, params: TParams) -> None:
        """Drop a client's hold; tear the transport down when the last leaves."""


class AsyncPool(ABC, Generic[TParams, TTransport]):
    """Pooling for asynchronous transports -- the async sibling of :class:`Pool`."""

    @abstractmethod
    async def connect(
        self, params: TParams, *, route: Optional[Hashable] = None
    ) -> AsyncConnection:
        """Lease the shared async transport for ``params`` as an
        :class:`AsyncConnection`. See :meth:`Pool.connect`."""

    @abstractmethod
    async def acquire(self, params: TParams) -> TTransport:
        """Reuse or create the shared async transport for ``params``."""

    @abstractmethod
    async def release(self, params: TParams) -> None:
        """Drop a client's hold; tear the transport down when the last leaves."""


class Wire(str, Enum):
    """The wire protocols the connection subsystem can hand a transport for."""

    MQTT = "mqtt"
    WEBSOCKET = "websocket"


_logger = logging.getLogger("connection.lease")

#: A handler for a connection event. May be sync, or return a coroutine that is
#: scheduled on the consumer loop. Receives the event payload: ``on_message`` ->
#: the wire message; ``on_connected`` -> nothing; ``on_disconnected`` -> the
#: :class:`Disconnected` event.
ConnectionHandler = Callable[..., Any]
Unsubscribe = Callable[[], None]
CoroFactory = Callable[[], Coroutine[Any, Any, Any]]


class BaseConnection(ABC, Generic[TParams]):
    """A per-client lease over a (possibly shared) transport.

    A consumer never touches the transport's raw event bus or its thread; it
    drives THIS handle: ``send``, subscribe to *its own* messages via
    :meth:`on_message`, and :meth:`close` its hold. The pool routes a shared
    transport's messages to the right lease(s) by ``route`` -- so a client no
    longer sees (and filters) every other client's traffic.
    """

    params: TParams

    @abstractmethod
    def on_message(self, handler: ConnectionHandler) -> Unsubscribe:
        """Receive this lease's inbound messages. Returns an unsubscribe callable."""

    @abstractmethod
    def on_connected(self, handler: ConnectionHandler) -> Unsubscribe:
        """Called when the shared link comes up. Returns an unsubscribe callable."""

    @abstractmethod
    def on_disconnected(self, handler: ConnectionHandler) -> Unsubscribe:
        """Called when the shared link drops. Returns an unsubscribe callable."""

    @abstractmethod
    def on_suspect(self, handler: ConnectionHandler) -> Unsubscribe:
        """Called on a suspect connect (repeated failure / CONNACK reject). The
        handler receives the :class:`ConnectSuspect` event. Returns an unsubscribe."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the underlying shared transport currently holds a live link."""

    @abstractmethod
    def subscribe(self, topic: str) -> None:
        """Register interest in ``topic`` on a shared transport (no-op on a 1:1)."""

    @abstractmethod
    def submit_to_consumer(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        """Run async work on the consumer loop from a transport thread.

        The one sanctioned cross-thread coroutine hop -- the pool owns the loop,
        so the caller never reaches for one. ``coalesce_key`` makes a second
        submission while one is in flight a no-op (replacing hand-rolled
        in-flight guards). Returns immediately; failures are logged by the pool.
        """


class Connection(BaseConnection[TParams]):
    """A **synchronous** lease (paho family)."""

    @abstractmethod
    def send(self, payload: Any) -> bool:
        """Send ``payload`` over the shared link. ``False`` if the link is down."""

    @abstractmethod
    def close(self) -> None:
        """Release this lease; the pool tears the transport down with the last."""


class AsyncConnection(BaseConnection[TParams]):
    """An **asynchronous** lease (``websockets`` / aiomqtt family)."""

    @abstractmethod
    async def send(self, payload: Any) -> None:
        """Send ``payload`` over the shared link; raises if unusable."""

    @abstractmethod
    async def close(self) -> None:
        """Release this lease; the pool tears the transport down with the last."""


class ConsumerLoop:
    """The cross-thread coroutine bridge behind :meth:`BaseConnection.submit_to_consumer`.

    A pool owns one (bound to the consumer's loop via an :class:`EventLoopProvider`)
    and hands it to every lease. This is the single, formal replacement for every
    ad-hoc ``getattr(client, "submit_to_loop")`` reach-in: the loop is owned here,
    not discovered from a client.
    """

    def __init__(
        self,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._provider = provider or EventLoopProvider.default()
        self._logger = logger or logging.getLogger("connection.consumer")
        self._inflight: Set[Hashable] = set()
        self._lock = threading.Lock()

    def submit(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        if coalesce_key is not None:
            with self._lock:
                if coalesce_key in self._inflight:
                    return
                self._inflight.add(coalesce_key)

        try:
            loop = self._provider.event_loop
        except RuntimeError:
            self._logger.warning("submit_to_consumer: no consumer loop available")
            self._discard(coalesce_key)
            return

        coro = coro_factory()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            coro.close()  # loop closed; don't leak an un-awaited coroutine
            self._discard(coalesce_key)
            return

        future.add_done_callback(lambda f: self._on_done(f, coalesce_key))

    def call(self, fn: Callable[[], None]) -> None:
        """Run synchronous work on the consumer loop.

        If the caller is already on the target loop the function runs inline;
        otherwise it is scheduled with ``call_soon_threadsafe``. This is the sync
        companion to :meth:`submit` for event-bus delivery from scheduler/worker
        threads.
        """
        try:
            loop = self._provider.event_loop
        except RuntimeError:
            self._logger.warning("call_to_consumer: no consumer loop available")
            return

        try:
            if asyncio.get_running_loop() is loop:
                self._safe_call(fn)
                return
        except RuntimeError:
            pass

        try:
            loop.call_soon_threadsafe(self._safe_call, fn)
        except RuntimeError:
            self._logger.warning("call_to_consumer: consumer loop is closed")

    def _safe_call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            self._logger.exception("call_to_consumer work failed")

    def _on_done(
        self, future: "concurrent.futures.Future", key: Optional[Hashable]
    ) -> None:
        self._discard(key)
        try:
            exc = future.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            self._logger.warning("submit_to_consumer work failed: %r", exc)

    def _discard(self, key: Optional[Hashable]) -> None:
        if key is None:
            return
        with self._lock:
            self._inflight.discard(key)


class TransportRouter:
    """Subscribes ONCE to a shared transport's event bus and fans each event to
    the right lease(s): messages by ``route``, connect/disconnect to all leases.

    This is what keeps "one broker socket, many printers" from devolving into
    "every client sees every message and filters it." The transport's bus has
    exactly one listener per event type -- this router -- which does the routing.

    An async transport runs on the consumer loop already, so the router's
    :meth:`dispatch` is subscribed directly (zero hops). A **sync** transport
    emits on its own thread; pass a :class:`Courier` to :meth:`attach` and the
    router instead subscribes ``courier.post`` -- the courier carries each event
    onto the loop and calls :meth:`dispatch` there.
    """

    def __init__(
        self,
        transport: "BaseTransport",
        *,
        topic_of: Optional[Callable[[Any], Optional[str]]] = None,
        matcher: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self._transport = transport
        self._topic_of = topic_of
        self._matcher = matcher
        self._leases: List["_LeaseCore"] = []
        self._lock = threading.RLock()
        self._message_courier: Optional[Any] = None
        self._lifecycle_courier: Optional[Any] = None
        self._message_target: Optional[Callable[[TransportEvent], None]] = None
        self._lifecycle_target: Optional[Callable[[TransportEvent], None]] = None

    def attach(
        self,
        courier: Optional[Any] = None,
        *,
        message_courier: Optional[Any] = None,
        lifecycle_courier: Optional[Any] = None,
    ) -> None:
        """Subscribe the router to transport events.

        Sync transports should pass separate couriers so telemetry can be bounded
        while lifecycle events remain lossless. ``courier`` is kept for the async
        direct path and simple tests.
        """
        if self._message_target is not None or self._lifecycle_target is not None:
            return
        if courier is not None:
            if message_courier is not None or lifecycle_courier is not None:
                raise ValueError("courier cannot be combined with split couriers")
            message_courier = courier
            lifecycle_courier = courier

        self._message_courier = message_courier
        self._lifecycle_courier = lifecycle_courier
        self._message_target = (
            message_courier.post if message_courier is not None else self.dispatch
        )
        self._lifecycle_target = (
            lifecycle_courier.post if lifecycle_courier is not None else self.dispatch
        )
        self._transport.events.on(MessageReceived, self._message_target)
        for event_type in (Connected, Disconnected, ConnectSuspect):
            self._transport.events.on(event_type, self._lifecycle_target)

    def detach(self) -> None:
        if self._message_target is None or self._lifecycle_target is None:
            return
        self._transport.events.off(MessageReceived, self._message_target)
        for event_type in (Connected, Disconnected, ConnectSuspect):
            self._transport.events.off(event_type, self._lifecycle_target)

        seen = set()
        for courier in (self._message_courier, self._lifecycle_courier):
            if courier is None or id(courier) in seen:
                continue
            seen.add(id(courier))
            courier.close(drain=False)
        with self._lock:
            self._leases.clear()
        self._message_target = None
        self._lifecycle_target = None
        self._message_courier = None
        self._lifecycle_courier = None

    def dispatch(self, event: TransportEvent) -> None:
        """Fan one event to the right lease(s). Runs on the consumer loop."""
        if isinstance(event, MessageReceived):
            self._on_message(event)
        elif isinstance(event, Connected):
            self._on_connected(event)
        elif isinstance(event, Disconnected):
            self._on_disconnected(event)
        elif isinstance(event, ConnectSuspect):
            self._on_suspect(event)

    def add(self, lease: "_LeaseCore") -> None:
        with self._lock:
            self._leases.append(lease)

    def remove(self, lease: "_LeaseCore") -> None:
        with self._lock:
            try:
                self._leases.remove(lease)
            except ValueError:
                pass

    def _snapshot(self) -> List["_LeaseCore"]:
        with self._lock:
            return list(self._leases)

    def match(self, route: Optional[Hashable], topic: Optional[str]) -> bool:
        if route is None:
            return True  # dedicated/1:1: every message is ours
        if topic is None:
            return False
        if route == topic:
            return True
        if self._matcher is not None and isinstance(route, str):
            return self._matcher(route, topic)
        return False

    def _on_message(self, event: MessageReceived) -> None:
        topic = self._topic_of(event.payload) if self._topic_of is not None else None
        for lease in self._snapshot():
            if self.match(lease._route, topic):
                lease._deliver_message(event.payload)

    def _on_connected(self, _event: Connected) -> None:
        for lease in self._snapshot():
            lease._deliver_connected()

    def _on_disconnected(self, event: Disconnected) -> None:
        for lease in self._snapshot():
            lease._deliver_disconnected(event)

    def _on_suspect(self, event: ConnectSuspect) -> None:
        for lease in self._snapshot():
            lease._deliver_suspect(event)


class _LeaseCore(Generic[TParams]):
    """Shared lease behaviour for both the sync and async families."""

    def __init__(
        self,
        *,
        params: TParams,
        transport: "BaseTransport",
        router: TransportRouter,
        consumer: ConsumerLoop,
        route: Optional[Hashable],
    ) -> None:
        self.params = params
        self._transport = transport
        self._router = router
        self._consumer = consumer
        self._route = route
        self._closed = False
        self._subscriptions: Set[str] = set()
        self._tasks: Set[asyncio.Task] = set()
        self._msg_handlers: List[ConnectionHandler] = []
        self._connected_handlers: List[ConnectionHandler] = []
        self._disconnected_handlers: List[ConnectionHandler] = []
        self._suspect_handlers: List[ConnectionHandler] = []

    def on_message(self, handler: ConnectionHandler) -> Unsubscribe:
        self._msg_handlers.append(handler)
        return lambda: self._remove(self._msg_handlers, handler)

    def on_connected(self, handler: ConnectionHandler) -> Unsubscribe:
        self._connected_handlers.append(handler)
        return lambda: self._remove(self._connected_handlers, handler)

    def on_disconnected(self, handler: ConnectionHandler) -> Unsubscribe:
        self._disconnected_handlers.append(handler)
        return lambda: self._remove(self._disconnected_handlers, handler)

    def on_suspect(self, handler: ConnectionHandler) -> Unsubscribe:
        self._suspect_handlers.append(handler)
        return lambda: self._remove(self._suspect_handlers, handler)

    @staticmethod
    def _remove(handlers: List[ConnectionHandler], handler: ConnectionHandler) -> None:
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def _deliver_message(self, payload: Any) -> None:
        self._dispatch(self._msg_handlers, payload)

    def _deliver_connected(self) -> None:
        self._dispatch(self._connected_handlers)

    def _deliver_disconnected(self, event: Disconnected) -> None:
        self._dispatch(self._disconnected_handlers, event)

    def _deliver_suspect(self, event: ConnectSuspect) -> None:
        self._dispatch(self._suspect_handlers, event)

    def _dispatch(self, handlers: List[ConnectionHandler], *args: Any) -> None:
        if self._closed:
            return
        for handler in list(handlers):
            try:
                result = handler(*args)
            except Exception:  # noqa: BLE001 -- a bad handler must not break routing
                _logger.exception("connection handler failed")
                continue
            if asyncio.iscoroutine(result):
                try:
                    task = asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    result.close()
                    continue
                self._tasks.add(task)
                task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            _logger.warning("async connection handler failed: %r", exc)

    def _close_core(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        for task in list(self._tasks):
            self._cancel_task(task)
        self._tasks.clear()
        self._release_subscriptions()
        self._router.remove(self)
        return True

    @staticmethod
    def _cancel_task(task: asyncio.Task) -> None:
        loop = task.get_loop()
        try:
            if asyncio.get_running_loop() is loop:
                task.cancel()
                return
        except RuntimeError:
            pass
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            task.cancel()

    @property
    def connected(self) -> bool:
        return self._transport.connected

    def subscribe(self, topic: str) -> None:
        if topic in self._subscriptions:
            return
        sub = getattr(self._transport, "subscribe", None)
        if sub is not None:
            sub(topic)
            self._subscriptions.add(topic)

    def _release_subscriptions(self) -> None:
        unsub = getattr(self._transport, "unsubscribe", None)
        if unsub is None:
            self._subscriptions.clear()
            return
        for topic in list(self._subscriptions):
            unsub(topic)
        self._subscriptions.clear()

    def submit_to_consumer(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        self._consumer.submit(coro_factory, coalesce_key=coalesce_key)


class _SyncLease(_LeaseCore[TParams], Connection[TParams]):
    """Concrete synchronous lease handed out by a sync :class:`Pool`."""

    def __init__(self, *, pool: "Pool", **kwargs: Any) -> None:
        _LeaseCore.__init__(self, **kwargs)
        self._pool = pool

    def send(self, payload: Any) -> bool:
        return self._transport.send(payload)

    def close(self) -> None:
        if not self._close_core():
            return
        self._pool.release(self.params)


class _AsyncLease(_LeaseCore[TParams], AsyncConnection[TParams]):
    """Concrete asynchronous lease handed out by an :class:`AsyncPool`."""

    def __init__(self, *, pool: "AsyncPool", **kwargs: Any) -> None:
        _LeaseCore.__init__(self, **kwargs)
        self._pool = pool

    async def send(self, payload: Any) -> None:
        await self._transport.send(payload)

    async def close(self) -> None:
        if not self._close_core():
            return
        await self._pool.release(self.params)
