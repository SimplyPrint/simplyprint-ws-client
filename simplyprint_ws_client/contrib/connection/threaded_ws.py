"""Shared threaded WebSocket transport (websocket-client).

Unlike MQTT, WebSocket printers are *not* pooled -- each printer owns one
``WebSocketApp`` on its own daemon thread. So the shared piece here is a small
transport object that brand clients *compose* (hold one) rather than subclass:
it owns the socket, the supervised connect/reconnect loop, an optional
app-level ping, and :class:`ConnectionState` tracking. The brand client keeps
its own message parsing, protocol and event emission and just wires three
callbacks (``on_message`` / ``on_connected`` / ``on_disconnected``).

This unifies brands that previously each grew their own retry thread or manual
ping thread onto one supervised, stop-aware implementation.

Note the deliberate name split from the async :class:`WebSocketTransport` ABC in
:mod:`.transport`: that one is the asyncio backend transport; this one is the
threaded, websocket-client-based printer transport. They share neither code nor
execution model -- only the WebSocket wire.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Hashable, NamedTuple, Optional, Type

import websocket

from simplyprint_ws_client.shared.utils.backoff import Backoff, ConstantBackoff

from .pool import ClientBucket, Connection, ConnectionManager, TClient
from .state import ConnectionState


class ThreadedWebSocketTransport:
    """Owns a ``WebSocketApp`` running on a supervised daemon thread.

    The supervisor thread runs ``run_forever`` (which itself auto-reconnects on
    drops) and, should it ever return, waits ``backoff`` and re-enters -- so a
    failed *initial* connect is retried too. ``stop()`` wakes it immediately.
    """

    #: Defaults shared by every brand today.
    DEFAULT_PING_INTERVAL = 5
    DEFAULT_PING_TIMEOUT = 2
    DEFAULT_RECONNECT = 5

    def __init__(
        self,
        url: str,
        *,
        logger,
        on_message: Callable[[str], None],
        on_connected: Optional[Callable[[], None]] = None,
        on_disconnected: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        ping_interval: int = DEFAULT_PING_INTERVAL,
        ping_timeout: int = DEFAULT_PING_TIMEOUT,
        reconnect: int = DEFAULT_RECONNECT,
        app_ping: Optional[Callable[[], None]] = None,
        app_ping_interval: float = 30.0,
        header: Optional[Any] = None,
        backoff: Optional[Backoff] = None,
    ) -> None:
        self.url = url
        self.logger = logger
        self._on_message = on_message
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_error = on_error
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._reconnect = reconnect
        self._app_ping = app_ping
        self._app_ping_interval = app_ping_interval
        self._header = header
        self._backoff = backoff or ConstantBackoff()

        self.state: ConnectionState = ConnectionState.OFFLINE
        self.wsapp: Optional[websocket.WebSocketApp] = None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._supervisor_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None

    @property
    def connected(self) -> bool:
        return (
            self._supervisor_thread is not None
            and self._supervisor_thread.is_alive()
            and self.state is ConnectionState.ONLINE
        )

    def start(self) -> None:
        with self._lock:
            if (
                self._supervisor_thread is not None
                and self._supervisor_thread.is_alive()
            ):
                return

            self._stop.clear()
            self._supervisor_thread = threading.Thread(
                target=self._supervise, daemon=True
            )
            self._supervisor_thread.start()

            if self._app_ping is not None and (
                self._ping_thread is None or not self._ping_thread.is_alive()
            ):
                self._ping_thread = threading.Thread(
                    target=self._ping_loop, daemon=True
                )
                self._ping_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            wsapp = self.wsapp
        if wsapp is not None:
            try:
                wsapp.close()
            except Exception as e:
                self.logger.warning("Failed to close WebSocket: %s", e)

    def send(self, data: str) -> bool:
        wsapp = self.wsapp
        if wsapp is None or wsapp.sock is None:
            self.logger.error("WebSocket is not connected")
            return False
        try:
            wsapp.send(data)
            return True
        except Exception as e:
            self.logger.error("Failed to send over WebSocket: %s", e)
            return False

    def _supervise(self) -> None:
        while not self._stop.is_set():
            try:
                self.wsapp = websocket.WebSocketApp(
                    self.url,
                    on_message=self._handle_message,
                    on_open=self._handle_open,
                    on_close=self._handle_close,
                    on_error=self._handle_error,
                    header=self._header,
                )
                # Blocks until the connection ends; reconnects internally on drops.
                self.wsapp.run_forever(
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    reconnect=self._reconnect,
                )
            except BaseException as e:  # noqa: BLE001 - keep the supervisor alive
                self.logger.warning("WebSocket %s error: %s", self.url, e)

            self.state = ConnectionState.OFFLINE

            # run_forever returned (closed / fatal). Back off, then retry unless stopped.
            if self._stop.wait(self._backoff.delay()):
                break

    def _ping_loop(self) -> None:
        while not self._stop.wait(self._app_ping_interval):
            if not self.connected or self._app_ping is None:
                continue
            try:
                self._app_ping()
            except Exception as e:
                self.logger.error("Error sending app-level ping: %s", e)

    def _handle_open(self, _wsapp) -> None:
        self.state = ConnectionState.ONLINE
        self.logger.debug("WebSocket open: %s", self.url)
        if self._on_connected is not None:
            self._on_connected()

    def _handle_message(self, _wsapp, message: str) -> None:
        self._on_message(message)

    def _handle_close(self, _wsapp, close_status_code, close_msg) -> None:
        self.state = ConnectionState.OFFLINE
        reason = (
            f"{close_status_code}: {close_msg}"
            if close_msg
            else str(close_status_code or "")
        )
        self.logger.debug("WebSocket closed: %s", reason)
        if self._on_disconnected is not None:
            self._on_disconnected(reason)

    def _handle_error(self, _wsapp, error) -> None:
        self.logger.error("%s", error)
        if self._on_error is not None:
            self._on_error(error)


class WsConnectionParams(NamedTuple):
    """Hashable identity of a websocket endpoint -- one per printer host."""

    url: str

    def __str__(self) -> str:
        return self.url


class WsConnection(Connection[WsConnectionParams]):
    """A websocket-client connection, pooled like any other :class:`Connection`.

    Composes the proven :class:`ThreadedWebSocketTransport` (which owns the one
    unavoidable blocking ``run_forever`` daemon thread) and adapts its callbacks
    onto the pool lifecycle: ``on_connected`` -> :meth:`handle_connected`,
    ``on_message`` -> fan ``message_event`` to every client on this connection,
    ``on_disconnected`` -> :meth:`handle_disconnected`. The blocking thread is
    owned here, by the pool -- never by a brand client.

    The optional ``app_ping`` is the brand keep-warm callback the transport fires
    on its own cadence (e.g. a device that needs a periodic poke); leave it
    ``None`` to rely on the WebSocket protocol ping alone.
    """

    def __init__(
        self,
        bucket: ClientBucket,
        params: WsConnectionParams,
        *,
        connected_event: Hashable,
        disconnected_event: Hashable,
        message_event: Hashable,
        app_ping: Optional[Callable[[], None]] = None,
        app_ping_interval: float = 30.0,
        header: Optional[Any] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            bucket,
            params,
            connected_event=connected_event,
            disconnected_event=disconnected_event,
            **kwargs,
        )

        self.message_event = message_event
        self.transport = ThreadedWebSocketTransport(
            params.url,
            logger=self.logger,
            on_message=self._on_message,
            on_connected=self.handle_connected,
            on_disconnected=self._on_disconnected,
            app_ping=app_ping,
            app_ping_interval=app_ping_interval,
            header=header,
        )
        self.transport.start()

    @property
    def connected(self) -> bool:
        return self.transport.connected

    def send(self, data: str) -> bool:
        """Write a text frame over this connection."""
        return self.transport.send(data)

    def _on_message(self, message: str) -> None:
        self.emit_sync_all(self.message_event, message)

    def _on_disconnected(self, reason: str = "") -> None:
        self.handle_disconnected(reason=reason)

    def stop(self) -> None:
        super().stop()
        self.transport.stop()


class WsConnectionManager(ConnectionManager[TClient, WsConnectionParams]):
    """Connection manager for websocket-client transports.

    Concrete brand managers set the event types + ``params_factory`` (as class
    attributes) and override :attr:`connection_class` for a brand-specific
    :class:`WsConnection`. WebSocket printers are 1:1 (one client per host), so
    there is no topic routing and no subscribe step -- the manager just owns the
    pooled connection's lifecycle and event fan-out.
    """

    connection_class: Type[WsConnection] = WsConnection

    def _create_connection(self, params: WsConnectionParams) -> WsConnection:
        return self.connection_class(
            bucket=self.bucket,
            params=params,
            connected_event=self.connected_event,
            disconnected_event=self.disconnected_event,
            message_event=self.message_event,
            parent_stoppable=self,
        )
