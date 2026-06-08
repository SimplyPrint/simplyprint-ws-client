"""Pooling, routing, and per-client leases -- the machinery behind the contracts.

This is where "many printers share one socket" actually lives. A :class:`Pool`
(``TransportPool`` / ``AsyncTransportPool``) keeps ONE supervised transport
per endpoint, ref-counted, and hands every client a lease (a ``Lease`` /
``AsyncLease``). A single :class:`_TransportFanout` per transport subscribes
to its event bus and broadcasts each event to every lease; the lease self-filters
by its route, so the fan-out stays dumb and no client sees another's traffic.

The two pools differ only in their lock (``threading`` vs ``asyncio``) and the
``async`` on ``connect``/``close``; the work under the lock is identical and lives
once in :class:`_PoolCommon`. A sync wire's events are born off-loop, so the
fan-out carries them across with two couriers built from a :class:`DeliveryConfig`
(bounded telemetry, lossless lifecycle); an async wire is already on the loop and
dispatches directly.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
)

from simplyprint_ws_client.shared.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    ConnectionSuspect,
    Disconnected,
    MessageReceived,
    TransportEvent,
)
from simplyprint_ws_client.contrib.connection.loop import CoroFactory, LoopBridge
from simplyprint_ws_client.contrib.connection.transport import (
    AsyncLease,
    AsyncPool,
    BaseTransport,
    Lease,
    LeaseHandler,
    Pool,
    TParams,
    TTransport,
    Unsubscribe,
)

__all__ = [
    "DeliveryConfig",
    "TransportPool",
    "AsyncTransportPool",
]

_logger = logging.getLogger("connection.lease")

#: Builds (but does not start) a transport for ``params``. Injectable for tests.
TransportFactory = Callable[[TParams], TTransport]


@dataclass(frozen=True)
class DeliveryConfig:
    """How a pooled *sync* transport's events cross from its wire thread to the loop.

    Message telemetry (``MessageReceived``) is shed under load -- a slow consumer
    must not grow an unbounded backlog -- while lifecycle events (connect /
    disconnect / suspect) are lossless: dropping a disconnect would strand a
    client ``ONLINE`` forever. Async transports already emit on the loop and
    ignore this entirely.
    """

    message_overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST
    message_maxsize: int = 1024
    lifecycle_overflow: OverflowPolicy = OverflowPolicy.UNBOUNDED
    lifecycle_maxsize: int = 1024


class _PoolCommon(Generic[TParams, TTransport]):
    """Shared refcount/fanout storage for concrete transport pools."""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger],
        transport_factory: Optional[TransportFactory[TParams, TTransport]],
        event_loop_provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]],
    ) -> None:
        self._logger = logger or logging.getLogger(type(self).__name__)
        self._transport_factory = transport_factory or self._build_transport
        self._provider = self._resolve_provider(event_loop_provider)
        self._loop_bridge = LoopBridge(
            self._provider, logger=self._logger.getChild("loop")
        )
        self._transports: Dict[TParams, TTransport] = {}
        self._refs: Dict[TParams, int] = {}
        self._fanouts: Dict[TParams, _TransportFanout] = {}

    @staticmethod
    def _resolve_provider(
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]],
    ) -> EventLoopProvider[asyncio.AbstractEventLoop]:
        if provider is not None:
            return provider
        try:
            return EventLoopProvider(loop=asyncio.get_running_loop())
        except RuntimeError:
            return EventLoopProvider.default()

    def _build_transport(self, params: TParams) -> TTransport:
        raise NotImplementedError

    def _make_fanout(self, transport: TTransport) -> "_TransportFanout":
        # Default: async wire -- events already on the loop, no courier (no
        # ``delivery``). A sync pool overrides to pass its :class:`DeliveryConfig`.
        return _TransportFanout(transport, provider=self._provider)  # type: ignore[arg-type]

    def _lease_route(self, route: Optional[Hashable]) -> Optional[Hashable]:
        return route

    def _lease_filter(
        self,
    ) -> Tuple[
        Optional[Callable[[Any], Optional[str]]],
        Optional[Callable[[str, str], bool]],
    ]:
        """The ``(topic_of, matcher)`` a lease uses to self-filter inbound messages.

        Default: a 1:1 link where every message is the lease's (no extractor, no
        matcher). A topic-routed wire (MQTT) overrides this to pull the message's
        topic and match it against the lease's route, so each lease on a shared
        socket sees only its own traffic."""
        return (None, None)

    def _after_connect(self, lease: "_LeaseCore", route: Optional[Hashable]) -> None:
        pass

    def _new_entry(self, params: TParams) -> TTransport:
        transport = self._transport_factory(params)
        self._transports[params] = transport
        self._refs[params] = 0
        return transport

    def _ensure_fanout(
        self, params: TParams, transport: TTransport
    ) -> "_TransportFanout":
        fanout = self._fanouts.get(params)
        if fanout is None:
            fanout = self._make_fanout(transport)
            self._attach_fanout(fanout)
            self._fanouts[params] = fanout
        return fanout

    def _attach_fanout(self, fanout: "_TransportFanout") -> None:
        fanout.attach()

    # -- shared lease/refcount bookkeeping (the caller holds the pool lock) --
    #
    # The sync and async pools differ only in their lock (``threading`` vs
    # ``asyncio``); the work under it is identical and does no I/O, so it lives
    # here once. Each helper returns the transports/fanouts to start or tear down
    # *outside* the lock, keeping blocking calls off the critical section.

    def _acquire(
        self,
        params: TParams,
        route: Optional[Hashable],
        lease_cls: "type",
    ) -> Tuple["_LeaseCore", Optional[TTransport]]:
        """Get-or-create the shared transport, refcount it, and build a lease.

        Returns ``(lease, start)`` where ``start`` is the transport to ``start()``
        iff it was newly created (so the caller starts it after releasing the lock).
        """
        start: Optional[TTransport] = None
        transport = self._transports.get(params)
        if transport is None:
            transport = self._new_entry(params)
            start = transport
        self._refs[params] += 1
        fanout = self._ensure_fanout(params, transport)
        topic_of, matcher = self._lease_filter()
        lease = lease_cls(
            pool=self,
            params=params,
            transport=transport,
            fanout=fanout,
            loop_bridge=self._loop_bridge,
            route=self._lease_route(route),
            topic_of=topic_of,
            matcher=matcher,
        )
        fanout.add(lease)
        return lease, start

    def _release_core(
        self, params: TParams
    ) -> Tuple[List[TTransport], List["_TransportFanout"]]:
        """Drop one ref; on the last, pop the transport+fanout for teardown."""
        if params not in self._refs:
            return [], []
        self._refs[params] -= 1
        if self._refs[params] > 0:
            return [], []
        self._refs.pop(params, None)
        transport = self._transports.pop(params, None)
        fanout = self._fanouts.pop(params, None)
        return (
            [transport] if transport is not None else [],
            [fanout] if fanout is not None else [],
        )

    def _stop_core(self) -> Tuple[List[TTransport], List["_TransportFanout"]]:
        """Snapshot + clear every entry for teardown outside the lock."""
        transports = list(self._transports.values())
        fanouts = list(self._fanouts.values())
        self._transports.clear()
        self._refs.clear()
        self._fanouts.clear()
        return transports, fanouts

    @staticmethod
    def _teardown(
        transports: List[TTransport], fanouts: List["_TransportFanout"]
    ) -> None:
        """Detach fanouts then stop transports (the order both pools used)."""
        for fanout in fanouts:
            fanout.detach()
        for transport in transports:
            transport.stop()

    def submit_to_loop(
        self,
        coro_factory: Callable[[], Any],
        *,
        coalesce_key: Optional[Hashable] = None,
    ) -> None:
        self._loop_bridge.submit(coro_factory, coalesce_key=coalesce_key)

    def call_on_loop(self, fn: Callable[[], None]) -> None:
        self._loop_bridge.call(fn)


class TransportPool(_PoolCommon[TParams, TTransport], Pool[TParams, TTransport]):
    """Refcounted pool for transports that emit events from producer threads."""

    def __init__(
        self,
        *,
        delivery: DeliveryConfig = DeliveryConfig(),
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[TransportFactory[TParams, TTransport]] = None,
        event_loop_provider: Optional[
            EventLoopProvider[asyncio.AbstractEventLoop]
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            transport_factory=transport_factory,
            event_loop_provider=event_loop_provider,
        )
        self._delivery = delivery
        self._lock = threading.Lock()

    def _make_fanout(self, transport: TTransport) -> "_TransportFanout":
        # Sync wire: events are born off-loop, so the fan-out carries them across
        # via two couriers built from this pool's :class:`DeliveryConfig`.
        return _TransportFanout(  # type: ignore[arg-type]
            transport, delivery=self._delivery, provider=self._provider
        )

    def connect(self, params: TParams, *, route: Optional[Hashable] = None) -> "Lease":
        with self._lock:
            lease, start = self._acquire(params, route, _Lease)
        self._after_connect(lease, route)
        if start is not None:
            start.start()
        return lease  # type: ignore[return-value]

    def _release(self, params: TParams) -> None:
        with self._lock:
            transports, fanouts = self._release_core(params)
        self._teardown(transports, fanouts)

    def stop(self) -> None:
        with self._lock:
            transports, fanouts = self._stop_core()
        self._teardown(transports, fanouts)


class AsyncTransportPool(
    _PoolCommon[TParams, TTransport], AsyncPool[TParams, TTransport]
):
    """Refcounted pool for transports that live on the pool asyncio loop."""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[TransportFactory[TParams, TTransport]] = None,
        event_loop_provider: Optional[
            EventLoopProvider[asyncio.AbstractEventLoop]
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            transport_factory=transport_factory,
            event_loop_provider=event_loop_provider,
        )
        self._lock = asyncio.Lock()

    async def connect(
        self, params: TParams, *, route: Optional[Hashable] = None
    ) -> "AsyncLease":
        async with self._lock:
            lease, start = self._acquire(params, route, _AsyncLease)
        self._after_connect(lease, route)
        if start is not None:
            start.start()
        return lease  # type: ignore[return-value]

    async def _release(self, params: TParams) -> None:
        async with self._lock:
            transports, fanouts = self._release_core(params)
        self._teardown(transports, fanouts)

    async def stop(self) -> None:
        async with self._lock:
            transports, fanouts = self._stop_core()
        self._teardown(transports, fanouts)


class _TransportFanout:
    """Subscribes ONCE to a shared transport's event bus and fans each event to
    every lease -- each lease then self-filters (a message whose topic does not
    match the lease's route is dropped by the lease, in :meth:`_LeaseCore.deliver`).

    Keeping the fan-out dumb is what lets "one broker socket, many printers" work
    without the fanout knowing anything about topics or routes: the transport's
    bus has exactly one listener per event type (this object), and routing is the
    lease's own business.

    An async transport runs on the pool loop already, so :meth:`dispatch` is
    subscribed directly (zero hops). A **sync** transport emits on its own thread,
    so it is built with a :class:`DeliveryConfig` and :meth:`attach` stands up two
    couriers (bounded for telemetry, lossless for lifecycle) that carry each event
    onto the loop and call :meth:`dispatch` there.
    """

    def __init__(
        self,
        transport: "BaseTransport",
        *,
        delivery: Optional[DeliveryConfig] = None,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
    ) -> None:
        self._transport = transport
        #: Set for a sync wire (events born off-loop -> courier hop); ``None`` for
        #: an async wire (events already on the loop -> direct dispatch).
        self._delivery = delivery
        self._provider = provider
        self._leases: List["_LeaseCore"] = []
        self._lock = threading.RLock()
        self._message_courier: Optional[Any] = None
        self._lifecycle_courier: Optional[Any] = None
        self._message_target: Optional[Callable[[TransportEvent], None]] = None
        self._lifecycle_target: Optional[Callable[[TransportEvent], None]] = None

    def attach(self) -> None:
        """Subscribe to the transport's event bus.

        A sync wire (``delivery`` set) gets two couriers -- bounded for message
        telemetry, lossless for lifecycle -- each carrying events off the wire
        thread onto the pool loop. An async wire (no ``delivery``) is already on
        the loop, so :meth:`dispatch` is subscribed directly.
        """
        if self._message_target is not None or self._lifecycle_target is not None:
            return
        if self._delivery is not None:
            self._message_courier = Courier(
                sink=self.dispatch,
                provider=self._provider,
                policy=self._delivery.message_overflow,
                maxsize=self._delivery.message_maxsize,
            )
            self._lifecycle_courier = Courier(
                sink=self.dispatch,
                provider=self._provider,
                policy=self._delivery.lifecycle_overflow,
                maxsize=self._delivery.lifecycle_maxsize,
            )
        self._message_target = (
            self._message_courier.post
            if self._message_courier is not None
            else self.dispatch
        )
        self._lifecycle_target = (
            self._lifecycle_courier.post
            if self._lifecycle_courier is not None
            else self.dispatch
        )
        self._transport.events.on(MessageReceived, self._message_target)
        for event_type in (Connected, Disconnected, ConnectionSuspect):
            self._transport.events.on(event_type, self._lifecycle_target)

    def detach(self) -> None:
        if self._message_target is None or self._lifecycle_target is None:
            return
        self._transport.events.off(MessageReceived, self._message_target)
        for event_type in (Connected, Disconnected, ConnectionSuspect):
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
        """Fan one event to every lease; each lease self-filters. Runs on the loop."""
        for lease in self._snapshot():
            lease.deliver(event)

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


class _LeaseCore(Generic[TParams]):
    """Shared lease behaviour for both the sync and async families."""

    def __init__(
        self,
        *,
        params: TParams,
        transport: "BaseTransport",
        fanout: _TransportFanout,
        loop_bridge: LoopBridge,
        route: Optional[Hashable],
        topic_of: Optional[Callable[[Any], Optional[str]]] = None,
        matcher: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self.params = params
        self._transport = transport
        self._fanout = fanout
        self._loop_bridge = loop_bridge
        self._route = route
        self._topic_of = topic_of
        self._matcher = matcher
        self._closed = False
        self._subscriptions: Set[str] = set()
        self._tasks: Set[asyncio.Task] = set()
        #: Handlers keyed by the transport-event type they listen for.
        self._handlers: Dict[type, List[LeaseHandler]] = {}

    def on_message(self, handler: LeaseHandler) -> Unsubscribe:
        return self._on(MessageReceived, handler)

    def on_connected(self, handler: LeaseHandler) -> Unsubscribe:
        return self._on(Connected, handler)

    def on_disconnected(self, handler: LeaseHandler) -> Unsubscribe:
        return self._on(Disconnected, handler)

    def on_suspect(self, handler: LeaseHandler) -> Unsubscribe:
        return self._on(ConnectionSuspect, handler)

    def _on(self, event_type: type, handler: LeaseHandler) -> Unsubscribe:
        self._handlers.setdefault(event_type, []).append(handler)
        return lambda: self._remove(event_type, handler)

    def _remove(self, event_type: type, handler: LeaseHandler) -> None:
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def deliver(self, event: TransportEvent) -> None:
        """Fan one transport event to this lease's handlers (runs on the pool loop).

        The lease self-filters: a :class:`MessageReceived` whose topic does not
        match this lease's ``route`` is dropped here, so the fanout can stay a
        dumb broadcast. ``Connected`` handlers are called with no args; the others
        receive the event."""
        if self._closed:
            return
        handlers = self._handlers.get(type(event))
        if not handlers:
            return
        if isinstance(event, MessageReceived):
            if not self._matches(event.payload):
                return
            self._dispatch(handlers, event.payload)
        elif isinstance(event, Connected):
            self._dispatch(handlers)
        else:
            self._dispatch(handlers, event)

    def _matches(self, payload: Any) -> bool:
        """Whether an inbound message belongs to this lease's route.

        ``route is None`` (the dedicated/1:1 case) takes every message; otherwise
        the topic is extracted with ``topic_of`` and compared to the route,
        falling back to the wire's ``matcher`` (e.g. MQTT ``#`` wildcards)."""
        route = self._route
        if route is None:
            return True
        topic = self._topic_of(payload) if self._topic_of is not None else None
        if topic is None:
            return False
        if route == topic:
            return True
        if self._matcher is not None and isinstance(route, str):
            return self._matcher(route, topic)
        return False

    def _dispatch(self, handlers: List[LeaseHandler], *args: Any) -> None:
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
        self._fanout.remove(self)
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
        self._transport.subscribe(topic)  # no-op on a 1:1 wire
        self._subscriptions.add(topic)

    def _release_subscriptions(self) -> None:
        for topic in list(self._subscriptions):
            self._transport.unsubscribe(topic)
        self._subscriptions.clear()

    def submit_to_loop(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        self._loop_bridge.submit(coro_factory, coalesce_key=coalesce_key)


class _Lease(_LeaseCore[TParams], Lease[TParams]):
    """Concrete synchronous lease handed out by a sync :class:`Pool`."""

    def __init__(self, *, pool: "Pool", **kwargs: Any) -> None:
        _LeaseCore.__init__(self, **kwargs)
        self._pool = pool

    def send(self, payload: Any) -> bool:
        return self._transport.send(payload)

    def close(self) -> None:
        if not self._close_core():
            return
        self._pool._release(self.params)


class _AsyncLease(_LeaseCore[TParams], AsyncLease[TParams]):
    """Concrete asynchronous lease handed out by an :class:`AsyncPool`."""

    def __init__(self, *, pool: "AsyncPool", **kwargs: Any) -> None:
        _LeaseCore.__init__(self, **kwargs)
        self._pool = pool

    async def send(self, payload: Any) -> None:
        await self._transport.send(payload)

    async def close(self) -> None:
        if not self._close_core():
            return
        await self._pool._release(self.params)
