"""Threaded WebSocket transport + pool (websocket-client) -- the WS sync family.

The WS sibling of the MQTT sync family. WebSocket printers are 1:1 (one socket per
host), so a "pool" here is really one transport per endpoint with a single lease,
but it goes through the same :class:`Pool` / :class:`Lease` contract as MQTT
so brands share the manager. The proven blocking ``run_forever`` wire lives in
:class:`~.threaded_impl.ThreadedImpl`; here we wrap it as a
:class:`Transport` whose callbacks become :class:`TransportEvent` s, and a
:class:`Courier` carries them off the wire's daemon thread onto the pool loop.

The wire is imported lazily, so importing this module never drags
``websocket-client`` in.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Hashable, Optional

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.manager import PooledConnectionManager
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    Disconnected,
    MessageReceived,
    StateChanged,
    TransportEvent,
)
from simplyprint_ws_client.contrib.connection.pool import (
    DeliveryConfig,
    TransportPool,
)
from simplyprint_ws_client.contrib.connection.transport import Transport
from simplyprint_ws_client.contrib.connection.websocket.common import WsParams


#: Builds the blocking websocket-client wire for ``params``. Injectable for tests.
WireFactory = Callable[[WsParams, logging.Logger], Any]


def _default_wire_factory(params: WsParams, logger: logging.Logger):
    from simplyprint_ws_client.contrib.connection.websocket.threaded_impl import (
        ThreadedImpl,  # lazy: importing this module must not need websocket-client
    )

    # Callbacks are wired by the transport after construction.
    return ThreadedImpl(params.url, logger=logger, on_message=lambda _m: None)


class WebSocketTransport(Transport[WsParams]):
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
        wire_factory: WireFactory = _default_wire_factory,
        app_ping: Optional[Callable[["WebSocketTransport"], None]] = None,
        app_ping_interval: float = 30.0,
    ) -> None:
        self.params = params
        self.events: EventBus[TransportEvent] = EventBus()
        self.state = ConnectionState.OFFLINE
        self.generation = 0
        self._logger = logger or logging.getLogger("websocket.sync")
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

    def send(self, payload: Any) -> bool:
        return bool(self._wire.send(payload))

    def _set_state(self, state: ConnectionState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    def _on_message(self, message: str) -> None:
        self._emit(MessageReceived(message))

    def _on_connected(self) -> None:
        self.generation += 1  # ws-client self-heals; a new connect == a new generation
        self._set_state(ConnectionState.ONLINE)
        self._emit(Connected())

    def _on_disconnected(self, reason: str = "") -> None:
        self._set_state(ConnectionState.OFFLINE)
        self._emit(Disconnected(reason=reason, transient=True))


#: Builds a :class:`WebSocketTransport` for ``params``. Injectable for tests.
WebSocketTransportFactory = Callable[[WsParams], WebSocketTransport]


class WebSocketPool(TransportPool[WsParams, WebSocketTransport]):
    """Pooling for the threaded WS transport -- the WS sibling of ``MqttPool``.

    One transport (and one lease) per endpoint, ref-counted, plus the courier that
    carries the wire-thread events onto the pool loop. There is no topic
    routing: a WS link is 1:1, so the single lease receives every message.
    """

    def __init__(
        self,
        *,
        delivery: DeliveryConfig = DeliveryConfig(),
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[WebSocketTransportFactory] = None,
        event_loop_provider: Optional[EventLoopProvider] = None,
        app_ping: Optional[Callable[[WebSocketTransport], None]] = None,
        app_ping_interval: float = 30.0,
    ) -> None:
        self._app_ping = app_ping
        self._app_ping_interval = app_ping_interval
        super().__init__(
            delivery=delivery,
            logger=logger or logging.getLogger("websocket.sync.pool"),
            transport_factory=transport_factory,
            event_loop_provider=event_loop_provider,
        )

    def _build_transport(self, params: WsParams) -> WebSocketTransport:
        return WebSocketTransport(
            params,
            logger=self._logger.getChild(str(params)),
            app_ping=self._app_ping,
            app_ping_interval=self._app_ping_interval,
        )

    def _lease_route(self, route: Optional[Hashable]) -> Optional[Hashable]:
        return None  # 1:1 socket -- every message on the link is this client's


class WebSocketConnectionManager(PooledConnectionManager):
    """A :class:`PooledConnectionManager` backed by the threaded :class:`WebSocketPool`.

    The base for brand WS managers: a subclass sets the event types and
    ``params_factory``; the wire + dispatch (run_forever -> Courier -> loop) and
    the 1:1 lease come from :class:`WebSocketPool`. WS links are 1:1, so the lease route
    is ``None`` (every message is the one client's).
    """

    def __init__(
        self,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        logger: Optional[logging.Logger] = None,
        delivery: DeliveryConfig = DeliveryConfig(),
        **kwargs: object,
    ) -> None:
        self._event_loop_provider = event_loop_provider
        self._pool_logger = logger
        self._delivery = delivery
        super().__init__(**kwargs)

    def _make_pool(self) -> WebSocketPool:
        return WebSocketPool(
            event_loop_provider=self._event_loop_provider,
            logger=self._pool_logger,
            delivery=self._delivery,
            app_ping=self._transport_app_ping(),
            app_ping_interval=self.app_ping_interval,
        )

    #: Interval (s) of the optional transport keep-warm; see ``_transport_app_ping``.
    app_ping_interval: float = 30.0

    def _transport_app_ping(self) -> Optional[Callable[[WebSocketTransport], None]]:
        """Optional brand keep-warm fired on every WS link, unconditionally, every
        ``app_ping_interval`` s by the wire's ping thread.

        Distinct from :meth:`_send_keepalive` (which only pokes *idle* clients):
        some printers drop a side-channel (e.g. a camera stream) unless a command is
        re-issued on a fixed cadence regardless of traffic. The callback receives the
        transport and sends through it; it knows no client. Default: no keep-warm."""
        return None

    def _route_for(self, client) -> Optional[Hashable]:
        return None  # 1:1 socket -- every message on the link is this client's
