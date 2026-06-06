"""Threaded WebSocket transport + pool (websocket-client) -- the WS sync family.

The WS sibling of :mod:`.sync_mqtt`. WebSocket printers are 1:1 (one socket per
host), so a "pool" here is really one transport per endpoint with a single lease,
but it goes through the same :class:`Pool` / :class:`Connection` contract as MQTT
so brands share the manager. The proven blocking ``run_forever`` wire lives in
:class:`~.threaded_ws.ThreadedWebSocketTransport`; here we wrap it as a
:class:`Transport` whose callbacks become :class:`TransportEvent` s, and a
:class:`Courier` carries them off the wire's daemon thread onto the consumer loop.

The wire is imported lazily, so importing this module never drags
``websocket-client`` in.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Hashable, NamedTuple, Optional

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.manager import PooledConnectionManager
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    Connected,
    Connection,
    ConsumerLoop,
    Disconnected,
    MessageReceived,
    Pool,
    StateChanged,
    Transport,
    TransportEvent,
    TransportRouter,
    _SyncLease,
)


class WsParams(NamedTuple):
    """Hashable identity of a websocket endpoint -- one per printer host."""

    url: str

    def __str__(self) -> str:
        return self.url


#: Builds the blocking websocket-client wire for ``params``. Injectable for tests.
WsWireFactory = Callable[[WsParams, logging.Logger], Any]


def _default_wire_factory(params: WsParams, logger: logging.Logger):
    from simplyprint_ws_client.contrib.connection.threaded_ws import (
        ThreadedWebSocketTransport,  # lazy: importing this module must not need websocket-client
    )

    # Callbacks are wired by the transport after construction.
    return ThreadedWebSocketTransport(
        params.url, logger=logger, on_message=lambda _m: None
    )


class ThreadedWsTransport(Transport[WsParams]):
    """A supervised websocket-client link wrapped as a :class:`Transport`.

    The wire owns the one unavoidable blocking ``run_forever`` daemon thread and
    its reconnect; we translate its callbacks into :class:`TransportEvent` s,
    emitted on that thread (the pool's courier crosses them to the loop).
    """

    def __init__(
        self,
        params: WsParams,
        *,
        logger: Optional[logging.Logger] = None,
        wire_factory: WsWireFactory = _default_wire_factory,
        app_ping: Optional[Callable[["ThreadedWsTransport"], None]] = None,
        app_ping_interval: float = 30.0,
    ) -> None:
        self.params = params
        self.events: EventBus[TransportEvent] = EventBus()
        self.state = ConnectionState.OFFLINE
        self._logger = logger or logging.getLogger("threaded_ws")
        self._app_ping = app_ping
        self._wire = wire_factory(params, self._logger)
        # Re-point the wire's callbacks at our event surface.
        self._wire._on_message = self._on_message
        self._wire._on_connected = self._on_connected
        self._wire._on_disconnected = self._on_disconnected
        # Optional brand keep-warm fired by the wire's ping thread every
        # ``app_ping_interval`` s, unconditionally (e.g. a camera-stream re-arm).
        if app_ping is not None:
            self._wire._app_ping = self._run_app_ping
            self._wire._app_ping_interval = app_ping_interval

    def _run_app_ping(self) -> None:
        if self._app_ping is not None:
            self._app_ping(self)

    @property
    def connected(self) -> bool:
        return bool(self._wire.connected)

    def start(self) -> None:
        self._wire.start()

    def stop(self) -> None:
        self._wire.stop()
        self._set_state(ConnectionState.OFFLINE)

    def subscribe(self, topic: str) -> None:
        pass  # WebSocket is 1:1 -- no topics.

    def send(self, payload: Any) -> bool:
        return bool(self._wire.send(payload))

    def _set_state(self, state: ConnectionState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    def _on_message(self, message: str) -> None:
        self._emit(MessageReceived(message))

    def _on_connected(self) -> None:
        self._set_state(ConnectionState.ONLINE)
        self._emit(Connected())

    def _on_disconnected(self, reason: str = "") -> None:
        self._set_state(ConnectionState.OFFLINE)
        self._emit(Disconnected(reason=reason, transient=True))


#: Builds a :class:`ThreadedWsTransport` for ``params``. Injectable for tests.
WsTransportFactory = Callable[[WsParams], ThreadedWsTransport]


class WsPool(Pool[WsParams, ThreadedWsTransport]):
    """Pooling for the threaded WS transport -- the WS sibling of ``MqttPool``.

    One transport (and one lease) per endpoint, ref-counted, plus the courier that
    carries the wire-thread events onto the consumer loop. There is no topic
    routing: a WS link is 1:1, so the single lease receives every message.
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[WsTransportFactory] = None,
        event_loop_provider: Optional[EventLoopProvider] = None,
        message_overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        message_maxsize: int = 1024,
        lifecycle_overflow: OverflowPolicy = OverflowPolicy.UNBOUNDED,
        lifecycle_maxsize: int = 1024,
        app_ping: Optional[Callable[[ThreadedWsTransport], None]] = None,
        app_ping_interval: float = 30.0,
    ) -> None:
        self._logger = logger or logging.getLogger("ws_pool")
        self._transport_factory = transport_factory or self._build_transport
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._consumer = ConsumerLoop(
            self._provider, logger=self._logger.getChild("consumer")
        )
        self._message_overflow = message_overflow
        self._message_maxsize = message_maxsize
        self._lifecycle_overflow = lifecycle_overflow
        self._lifecycle_maxsize = lifecycle_maxsize
        self._app_ping = app_ping
        self._app_ping_interval = app_ping_interval
        self._transports: Dict[WsParams, ThreadedWsTransport] = {}
        self._refs: Dict[WsParams, int] = {}
        self._routers: Dict[WsParams, TransportRouter] = {}
        self._lock = threading.Lock()

    def _build_transport(self, params: WsParams) -> ThreadedWsTransport:
        return ThreadedWsTransport(
            params,
            logger=self._logger.getChild(str(params)),
            app_ping=self._app_ping,
            app_ping_interval=self._app_ping_interval,
        )

    def connect(
        self, params: WsParams, *, route: Optional[Hashable] = None
    ) -> Connection:
        transport = self.acquire(params)
        with self._lock:
            router = self._routers.get(params)
            if router is None:
                router = TransportRouter(transport)  # 1:1: no topic_of/matcher
                message_courier: Courier = Courier(
                    sink=router.dispatch,
                    provider=self._provider,
                    policy=self._message_overflow,
                    maxsize=self._message_maxsize,
                )
                lifecycle_courier: Courier = Courier(
                    sink=router.dispatch,
                    provider=self._provider,
                    policy=self._lifecycle_overflow,
                    maxsize=self._lifecycle_maxsize,
                )
                router.attach(
                    message_courier=message_courier,
                    lifecycle_courier=lifecycle_courier,
                )
                self._routers[params] = router
        lease = _SyncLease(
            pool=self,
            params=params,
            transport=transport,
            router=router,
            consumer=self._consumer,
            route=None,  # 1:1: the single lease receives every message
        )
        router.add(lease)
        return lease

    def acquire(self, params: WsParams) -> ThreadedWsTransport:
        with self._lock:
            transport = self._transports.get(params)
            if transport is None:
                transport = self._transport_factory(params)
                self._transports[params] = transport
                self._refs[params] = 0
                transport.start()
            self._refs[params] += 1
            return transport

    def release(self, params: WsParams) -> None:
        transport = None
        router = None
        with self._lock:
            if params not in self._refs:
                return
            self._refs[params] -= 1
            if self._refs[params] <= 0:
                self._refs.pop(params, None)
                transport = self._transports.pop(params, None)
                router = self._routers.pop(params, None)
        if router is not None:
            router.detach()
        if transport is not None:
            transport.stop()

    def submit_to_consumer(
        self,
        coro_factory: Callable[[], Any],
        *,
        coalesce_key: Optional[Hashable] = None,
    ) -> None:
        self._consumer.submit(coro_factory, coalesce_key=coalesce_key)

    def call_to_consumer(self, fn: Callable[[], None]) -> None:
        self._consumer.call(fn)

    def stop(self) -> None:
        with self._lock:
            transports = list(self._transports.values())
            routers = list(self._routers.values())
            self._transports.clear()
            self._refs.clear()
            self._routers.clear()
        for router in routers:
            router.detach()
        for transport in transports:
            transport.stop()


class PooledWsConnectionManager(PooledConnectionManager):
    """A :class:`PooledConnectionManager` backed by the threaded :class:`WsPool`.

    The base for brand WS managers: a subclass sets the event types and
    ``params_factory``; the wire + dispatch (run_forever -> Courier -> loop) and
    the 1:1 lease come from :class:`WsPool`. WS links are 1:1, so the lease route
    is ``None`` (every message is the one client's).
    """

    def __init__(
        self,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        logger: Optional[logging.Logger] = None,
        message_overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        message_maxsize: int = 1024,
        lifecycle_overflow: OverflowPolicy = OverflowPolicy.UNBOUNDED,
        lifecycle_maxsize: int = 1024,
        **kwargs: object,
    ) -> None:
        self._event_loop_provider = event_loop_provider
        self._pool_logger = logger
        self._message_overflow = message_overflow
        self._message_maxsize = message_maxsize
        self._lifecycle_overflow = lifecycle_overflow
        self._lifecycle_maxsize = lifecycle_maxsize
        super().__init__(**kwargs)

    def _make_pool(self) -> WsPool:
        return WsPool(
            event_loop_provider=self._event_loop_provider,
            logger=self._pool_logger,
            message_overflow=self._message_overflow,
            message_maxsize=self._message_maxsize,
            lifecycle_overflow=self._lifecycle_overflow,
            lifecycle_maxsize=self._lifecycle_maxsize,
            app_ping=self._transport_app_ping(),
            app_ping_interval=self.app_ping_interval,
        )

    #: Interval (s) of the optional transport keep-warm; see ``_transport_app_ping``.
    app_ping_interval: float = 30.0

    def _transport_app_ping(self) -> Optional[Callable[[ThreadedWsTransport], None]]:
        """Optional brand keep-warm fired on every WS link, unconditionally, every
        ``app_ping_interval`` s by the wire's ping thread.

        Distinct from :meth:`_send_keepalive` (which only pokes *idle* clients):
        some printers drop a side-channel (e.g. a camera stream) unless a command is
        re-issued on a fixed cadence regardless of traffic. The callback receives the
        transport and sends through it; it knows no client. Default: no keep-warm."""
        return None

    def _route_for(self, client) -> Optional[Hashable]:
        return None  # 1:1 socket -- every message on the link is this client's
