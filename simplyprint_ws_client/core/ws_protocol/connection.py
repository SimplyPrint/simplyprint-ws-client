__all__ = ["Connection", "ConnectionHint", "ConnectionMode"]

import asyncio
import logging
from enum import Enum
from typing import Any, Hashable, Optional, final

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from yarl import URL

from .events import (
    ConnectionEvent,
    ConnectionEstablishedEvent,
    ConnectionIncomingEvent,
    ConnectionLostEvent,
    ConnectionOutgoingEvent,
    ConnectionSuspectEvent,
)
from .messages import ClientMsg, ClientMsgType, ServerMsg
from ..config import PrinterConfig
from ...events import EventBus
from ...shared.asyncio.courier import Courier, OverflowPolicy
from ...shared.asyncio.event_loop_provider import EventLoopProvider
from ...shared.sp.url_builder import SimplyPrintURL
from ...shared.utils.backoff import ConstantBackoff
from ...shared.utils.bounded_variable import BoundedInterval
from ...shared.utils.stoppable import AsyncStoppable
from ...contrib.logging import printer_logger
from ...contrib.connection.events import (
    Connected,
    ConnectionSuspect,
    Disconnected,
    MessageReceived,
)
from ...contrib.connection.reconnect import Link, ReconnectingTransport
from ...contrib.connection.websocket.base import (
    WebSocket,
    WebSocketError,
    WebSocketFactory,
)
from ...contrib.connection.websocket.websockets_impl import WebsocketsImpl


class ConnectionMode(Enum):
    MULTI = "mp"
    SINGLE = "p"


class ConnectionHint:
    mode: ConnectionMode = ConnectionMode.SINGLE
    config: PrinterConfig = PrinterConfig.get_blank()

    def __init__(
        self,
        mode: Optional[ConnectionMode] = None,
        config: Optional[PrinterConfig] = None,
    ):
        self.mode = mode or self.mode
        self.config = config or self.config

    @property
    def ws_url(self) -> URL:
        return (
            SimplyPrintURL().ws_url
            / self.mode.value
            / str(self.config.id)
            / str(self.config.token)
        )


# WebSocket transport parameters, passed to ``WebSocket.connect``.
# Mirrors the previous aiohttp settings: a 30s ping heartbeat, unbounded message
# size, a 60s connect timeout and a 10s close timeout.
TransportParams = {
    "ping_interval": 30,
    "ping_timeout": 30,
    "open_timeout": 60,
    "close_timeout": 10,
    "max_size": None,
}

# Errors we treat as a closed connection when sending.
WsConnectionErrors = (
    OSError,
    ConnectionError,  # Technically a subset of OSError, but more specific.
    asyncio.TimeoutError,
    asyncio.CancelledError,
    WebSocketError,  # Base of WebSocketClosed -- covers both.
)

# How often a stuck-connecting socket is flagged ``suspect`` (every Nth failure),
# and how long after connect a silent server gets before the link is recycled.
WsSuspectConnectionBoundedInterval = BoundedInterval[int](7, 1)
WsFirstMessageTimeout = 30.0


class _ServerLink(Link):
    """One backend connection attempt -- the raw :class:`WebSocket` socket adapted
    to the engine's neutral :class:`Link`.

    ``open`` builds a fresh socket via the connection's factory and publishes it as
    :attr:`Connection.transport`, so the version-targeted :meth:`Connection.send`
    and :attr:`Connection.connected` read the live socket directly; ``recv`` yields
    raw text the connection parses into a :class:`ServerMsg`; ``close`` tears the
    socket down and clears the published handle. The link owns no reconnect /
    backoff / version logic -- that is the :class:`~...contrib.connection.reconnect.ReconnectingTransport`
    engine's job.
    """

    def __init__(self, connection: "Connection") -> None:
        self._conn = connection
        self._socket: Optional[WebSocket] = None

    async def open(self) -> None:
        socket = self._conn.transport_factory(self._conn.logger)
        await socket.connect(str(self._conn.url), **TransportParams)
        self._socket = socket
        self._conn.transport = socket

    async def recv(self) -> Optional[str]:
        assert self._socket is not None
        return await self._socket.recv()

    async def send(self, payload: Any) -> None:
        assert self._socket is not None
        await self._socket.send(payload)

    async def close(self) -> None:
        socket, self._socket = self._socket, None
        # Only retract the published handle if it is still ours: a freshly-opened
        # attempt may have already replaced it (resume races a winding-down drop).
        if self._conn.transport is socket:
            self._conn.transport = None
        if socket is not None:
            await socket.close()

    @property
    def is_open(self) -> bool:
        return self._socket is not None and self._socket.is_open


@final
class Connection(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop], Hashable
):
    """The link to the SimplyPrint server: the connection *protocol* over a shared
    reconnection engine.

    The reconnect/backoff/liveness/generation loop is no longer hand-rolled here --
    it is a :class:`~...contrib.connection.reconnect.ReconnectingTransport` transport driving
    a fresh :class:`_ServerLink` per attempt. ``Connection`` keeps only what is
    genuinely SimplyPrint-specific: the URL, version-targeted ``send`` + JSON
    (de)serialization, ``ServerMsg`` parsing, per-printer log routing, and the
    ordered lifetime/message event delivery on its own UNBOUNDED :class:`Courier`.

    It maps the engine's brand-free :class:`TransportEvent` s onto the protocol's
    events: ``Connected`` -> :class:`ConnectionEstablishedEvent` (``v``); a parsed
    ``MessageReceived`` -> :class:`ConnectionIncomingEvent` (``msg``, ``v``);
    ``Disconnected`` -> :class:`ConnectionLostEvent` (``v``) **then** ``v += 1``
    (the single version-bump site -- structurally one bump per ended attempt, so
    the historic double-increment cannot recur); ``ConnectionSuspect`` ->
    :class:`ConnectionSuspectEvent`. ``connect`` / ``disconnect`` start / stop the
    engine; ``stop`` (from :class:`AsyncStoppable`) tears it down permanently.

    Attributes:
        v: Connection generation, incremented on every ended attempt to invalidate
            messages targeted at a since-dropped link.
        transport: The live underlying socket (published by the engine's link,
            ``None`` while down) -- what ``send`` / ``connected`` read.
        hint: Connection hint from which the URL is derived.
    """

    v: int
    transport: Optional[WebSocket]
    transport_factory: WebSocketFactory
    hint: ConnectionHint
    logger: logging.Logger

    event_bus: EventBus[ConnectionEvent]

    _engine: Optional[ReconnectingTransport]

    def __init__(
        self,
        transport_factory: WebSocketFactory = WebsocketsImpl,
        hint: Optional[ConnectionHint] = None,
        logger: logging.Logger = logging.getLogger("ws"),
        **kwargs,
    ):
        AsyncStoppable.__init__(self, **kwargs)
        EventLoopProvider.__init__(self, **kwargs)

        self.v = 0
        self.transport = None
        self.transport_factory = transport_factory
        self.hint = hint or ConnectionHint()
        self.logger = logger

        self.event_bus = EventBus[ConnectionEvent]()
        self.event_bus.on(ConnectionOutgoingEvent, self.send)

        # Lifetime/message events used to go out via `event_bus.emit_task`, which
        # allocates a concurrent.futures.Future per event even though these run on
        # the loop already. The courier delivers them with the same ordered,
        # deferred, single-loop semantics -- no per-event Future, coalesced
        # wakeups -- reusing the bus's own loop provider so resolution is
        # identical. UNBOUNDED: dropping a lifetime event would desync `v`.
        self._event_courier: Courier = Courier(
            sink=self._emit_event,
            is_async_sink=True,
            provider=self.event_bus.event_loop_provider,
            policy=OverflowPolicy.UNBOUNDED,
        )

        self._engine = None

    async def _emit_event(self, item) -> None:
        """Courier sink: re-emit one ``(event, args)`` on the event bus, in order."""
        event, args = item
        await self.event_bus.emit(event, *args)

    def _post(self, event: object, *args: object) -> None:
        """Queue a lifetime/message event for ordered delivery on the loop.

        Drop-in for the old ``event_bus.emit_task(event, *args)`` at every site,
        but coalesced and without a per-event future.
        """
        self._event_courier.post((event, args))

    def __hash__(self):
        return hash(id(self))

    def __await__(self):
        task = self.loop_task
        if task is None:

            async def _done() -> None:
                return None

            return _done().__await__()
        return task.__await__()

    @property
    def url(self) -> URL:
        return self.hint.ws_url

    @property
    def connected(self):
        """This has nothing to do with our `Connection` state and everything to do
        with the real, physical underlying connection state."""
        return self.transport is not None and self.transport.is_open

    @property
    def running(self):
        """Whether the connection loop is running."""
        task = self.loop_task
        return task is not None and not task.done()

    @property
    def loop_task(self) -> Optional[asyncio.Task]:
        """The engine's supervision task, while running -- the handle the scheduler
        awaits at teardown. ``None`` before the first ``connect`` / after ``stop``."""
        return self._engine.task if self._engine is not None else None

    def _build_engine(self) -> ReconnectingTransport:
        """Build (once) the reconnection engine and wire its events to ours.

        Built lazily on first ``connect`` so module-level knobs
        (``WsFirstMessageTimeout``) are read at connect time, the way the previous
        loop read them -- which is what the patch-based tests rely on.
        """
        engine: ReconnectingTransport = ReconnectingTransport(
            str(self.url),
            lambda: _ServerLink(self),
            logger=self.logger,
            backoff=ConstantBackoff(),
            suspect_after=WsSuspectConnectionBoundedInterval,
            first_message_timeout=WsFirstMessageTimeout,
        )
        engine.events.on(Connected, self._on_connected)
        engine.events.on(Disconnected, self._on_disconnected)
        engine.events.on(MessageReceived, self._on_message)
        engine.events.on(ConnectionSuspect, self._on_suspect)
        return engine

    def _on_connected(self, _event: Connected) -> None:
        self._post(ConnectionEstablishedEvent(self.v))
        self.logger.info("Connected to %s", self.url)

    def _on_disconnected(self, _event: Disconnected) -> None:
        # The single version-bump site: one ended attempt -> one Lost(v) -> v += 1,
        # preserving the Established(g)...Lost(g) pairing the consumer relies on.
        self.logger.debug("Emitting ConnectionLostEvent.")
        self._post(ConnectionLostEvent(self.v))
        self.v += 1

    def _on_message(self, event: MessageReceived) -> None:
        data = event.payload
        if data is None:
            return
        try:
            msg = ServerMsg.model_validate_json(data).root
        except ValidationError as e:
            self.logger.error("Invalid message: %s", data, exc_info=e)
            return
        self._message_logger(msg).debug("received %s", msg)
        self._post(ConnectionIncomingEvent, msg, self.v)

    def _on_suspect(self, event: ConnectionSuspect) -> None:
        self._post(ConnectionSuspectEvent, event.error)

    def _message_logger(self, msg) -> logging.Logger:
        """Route a WS message's log line to the printer it belongs to.

        A printer-specific message goes to that printer's ``ws`` log
        (``<unique_id>/ws.log``); a genuinely global message stays on the
        connection's ``ws`` logger (the system scope). In MULTI mode each message
        carries ``for_client`` -- the printer's unique_id, or ``None`` when
        global; in SINGLE mode every message belongs to the connection's printer.
        """
        if self.hint.mode == ConnectionMode.MULTI:
            client_id = getattr(msg, "for_client", None)
        else:
            client_id = self.hint.config.unique_id
        if client_id:
            return printer_logger(str(client_id), "ws")
        return self.logger

    async def send(
        self, msg: ClientMsg[ClientMsgType], v: Optional[int] = None
    ) -> None:
        """
        :param msg: Message to send.
        :param v: Optional version to target.
        """

        # We drop messages if we are not connected.
        if not self.connected:
            self.logger.warning("Dropped message %s, not connected.", msg)
            return

        # Optionally, specify a connection version the message was targeted for.
        # Messages with a different version will not be sent.
        if v is not None and self.v != v:
            self.logger.warning(
                "Dropped message %s, version mismatch. %d != %d", msg, self.v, v
            )
            return

        try:
            data = msg.model_dump_json()
            await self.transport.send(data)
            self._message_logger(msg).debug(
                "sent %s", data if len(data) < 1024 else msg.msg_type()
            )

        except (PydanticSerializationError, UnicodeError) as e:
            # Serialization error.
            self.logger.error("Serialization error.", exc_info=e)

        except WsConnectionErrors:
            # The link dropped mid-send; the engine's recv loop sees the same drop
            # and reconnects on its own -- nothing to poke here.
            pass

    async def connect(self, hint: Optional[ConnectionHint] = None):
        """Start (or resume) the connection: spin up the reconnection engine.

        Idempotent -- ``start`` is a no-op while the engine is already supervising,
        and re-creates the task after a ``disconnect`` (the resume path).
        """
        self.hint = hint or self.hint

        if self.is_stopped():
            return

        if self._engine is None:
            self._engine = self._build_engine()

        self._engine.start()

    async def disconnect(self):
        """Pause (disconnect) the connection: stop the engine, no reconnection
        attempts until ``connect`` is called again."""
        if self._engine is not None:
            self._engine.stop()

    def stop(self):
        """Permanently tear the connection down and drain pending events."""
        super().stop()
        if self._engine is not None:
            self._engine.stop()
        self._event_courier.close(drain=True)
