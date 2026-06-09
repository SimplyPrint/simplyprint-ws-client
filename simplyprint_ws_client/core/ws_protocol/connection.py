from __future__ import annotations

__all__ = ["Connection", "ConnectionHint", "ConnectionMode"]

import asyncio
import logging
from enum import Enum
from typing import Hashable, Optional, final

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
from .backend import (
    BackendError,
    BackendFactory,
    BackendTransport,
    WebsocketsBackend,
)
from ..config import PrinterConfig
from ...events import EventBus
from ...shared.asyncio.event_loop_provider import EventLoopProvider
from ...shared.sp.url_builder import SimplyPrintURL
from ...shared.utils.backoff import ConstantBackoff
from ...shared.utils.bounded_variable import BoundedInterval
from ...shared.utils.stoppable import AsyncStoppable
from ...contrib.logging import printer_logger
from ...contrib.connection.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from ...contrib.connection.policy import RetryPolicy
from ...contrib.connection.reconnect import Reconnecting
from ...contrib.connection.transport import TransientError, WsTransport


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


# WebSocket transport parameters, passed to ``BackendTransport.connect``.
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
    BackendError,
)

# How often a stuck-connecting backend is flagged ``suspect`` (every Nth failure),
# and how long after connect a silent server gets before the link is recycled.
WsSuspectConnectionBoundedInterval = BoundedInterval[int](7, 1)
WsFirstMessageTimeout = 30.0


class ServerTransport(WsTransport, Reconnecting):
    """The SimplyPrint backend transport.

    This is the transport the protocol :class:`Connection` drives. It is a
    :class:`~simplyprint_ws_client.contrib.connection.reconnect.Reconnecting` transport that
    fills the four wire hooks (``open`` / ``recv`` / ``write`` / ``aclose``) on
    *itself* -- no separate link object -- holding the raw :class:`BackendTransport`
    as a plain attribute. The reconnect base owns the whole reliability story
    (connect, reconnect, backoff, generation, lifecycle events); this class adds only
    the two SimplyPrint-specific socket behaviours the brand-free base deliberately
    omits:

    * a fresh socket per attempt, built via the connection's
      :data:`~simplyprint_ws_client.core.ws_protocol.backend.BackendFactory` and
      published as :attr:`Connection.transport` so the version-targeted
      :meth:`Connection.send` and :attr:`Connection.connected` read the live socket
      directly;
    * a *first-message* deadline -- the attempt is dropped (and reconnected) if no
      real message arrives within :attr:`first_message_timeout` of connecting.

    Parsing, the version counter, and event translation stay in :class:`Connection`.
    """

    def __init__(self, connection: "Connection") -> None:
        super().__init__(
            connection.url,
            RetryPolicy(backoff=ConstantBackoff()),
            provider=connection,
            logger=connection.logger,
        )
        self.connection = connection
        self.first_message_timeout: Optional[float] = WsFirstMessageTimeout
        #: The live raw backend for this attempt, or ``None`` while down.
        self.backend: Optional[BackendTransport] = None

    async def open(self) -> None:
        """Build a fresh socket via the factory, connect it, and publish it as the
        connection's live :attr:`~Connection.transport`."""
        backend = self.connection.backend_factory(self.connection.logger)
        await backend.connect(str(self.connection.url), **TransportParams)
        self.backend = backend
        self.connection.backend = backend

    async def recv(self) -> Optional[str]:
        """Return the next inbound text frame; a dropped socket ends the attempt."""
        backend = self.backend
        if backend is None:
            raise TransientError("websocket not connected")
        return await backend.recv()

    async def write(self, message: object) -> None:
        """Put one frame on the live socket."""
        backend = self.backend
        if backend is None:
            raise TransientError("websocket not connected")
        await backend.send(message)

    async def aclose(self) -> None:
        """Tear the backend down and retract the published handle. Never raises."""
        backend, self.backend = self.backend, None
        # Only retract the published handle if it is still ours: a freshly-opened
        # attempt may have already replaced it (resume races a winding-down drop).
        if self.connection.backend is backend:
            self.connection.backend = None
        if backend is not None:
            try:
                await backend.close()
            except Exception:  # noqa: BLE001 -- aclose must never break supervision
                self.logger.debug("websocket %s close failed", self.url, exc_info=True)

    async def consume(self) -> None:
        """Stream inbound frames, with a deadline on the *first* real message.

        A liveness deadline applies only until the first message arrives; if none
        does in time the attempt is dropped (and reconnected) like any other. After
        the first message the stream falls back to the brand-free base behaviour.
        """
        if self.first_message_timeout is not None:
            await asyncio.wait_for(self.await_first(), self.first_message_timeout)
        await super().consume()

    async def await_first(self) -> None:
        """Block until one real message arrives, emitting it like the base loop."""
        while not self.stopped:
            message = await self.recv()
            if message is not None:
                await self.events.emit(MessageReceived(self.generation, message))
                return


@final
class Connection(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop], Hashable
):
    """The SimplyPrint server protocol over a reconnecting transport.

    The reconnect/backoff/liveness/generation loop is no longer hand-rolled here --
    it is a :class:`ServerTransport` riding the brand-free
    :class:`~simplyprint_ws_client.contrib.connection.reconnect.Reconnecting` loop.
    ``Connection`` keeps only what is genuinely SimplyPrint-specific: the URL,
    version-targeted ``send`` + JSON (de)serialization, ``ServerMsg`` parsing,
    per-printer log routing, and the suspect-after-N-failures advisory.

    It maps the transport's brand-free events onto the protocol's events:
    :class:`~simplyprint_ws_client.contrib.connection.events.Connected` ->
    :class:`ConnectionEstablishedEvent` (``v``); a parsed
    :class:`~simplyprint_ws_client.contrib.connection.events.MessageReceived` ->
    :class:`ConnectionIncomingEvent` (``msg``, ``v``);
    :class:`~simplyprint_ws_client.contrib.connection.events.Disconnected` ->
    :class:`ConnectionLostEvent` (``v``) **then** ``v += 1`` (the single version-bump
    site -- structurally one bump per ended attempt, so the historic
    double-increment cannot recur). A run of failed attempts (a ``Disconnected`` that
    carried an error code) raises :class:`ConnectionSuspectEvent` every Nth.
    ``connect`` / ``disconnect`` start / stop the transport; ``stop`` (from
    :class:`AsyncStoppable`) tears it down permanently.

    Attributes:
        v: Connection generation, incremented on every ended attempt to invalidate
            messages targeted at a since-dropped link.
        backend: The live underlying connection (published by the transport,
            ``None`` while down) -- what ``send`` / ``connected`` read.
        hint: Connection hint from which the URL is derived.
    """

    v: int
    backend: Optional[BackendTransport]
    backend_factory: BackendFactory
    hint: ConnectionHint
    logger: logging.Logger

    event_bus: EventBus[ConnectionEvent]

    _transport: Optional[ServerTransport]

    def __init__(
        self,
        backend_factory: BackendFactory = WebsocketsBackend,
        hint: Optional[ConnectionHint] = None,
        logger: logging.Logger = logging.getLogger("ws"),
        **kwargs,
    ):
        AsyncStoppable.__init__(self, **kwargs)
        EventLoopProvider.__init__(self, **kwargs)

        self.v = 0
        self.backend = None
        self.backend_factory = backend_factory
        self.hint = hint or ConnectionHint()
        self.logger = logger

        self.event_bus = EventBus[ConnectionEvent]()
        self.event_bus.on(ConnectionOutgoingEvent, self.send)

        # Tracks runs of failed connection attempts so a stuck endpoint raises a
        # periodic ConnectionSuspectEvent (every Nth failure); reset on a connect.
        self._suspect = WsSuspectConnectionBoundedInterval.create_variable(0)

        self._transport = None

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
        return self.backend is not None and self.backend.is_open

    @property
    def running(self):
        """Whether the connection loop is running."""
        task = self.loop_task
        return task is not None and not task.done()

    @property
    def loop_task(self) -> Optional[asyncio.Task]:
        """The transport's supervision task, while running -- the handle the scheduler
        awaits at teardown. ``None`` before the first ``connect`` / after ``stop``."""
        return self._transport.task if self._transport is not None else None

    def _build_transport(self) -> ServerTransport:
        """Build (once) the reconnecting transport and wire its events to ours.

        Built lazily on first ``connect`` so module-level knobs
        (``WsFirstMessageTimeout``) are read at connect time, the way the previous
        loop read them -- which is what the patch-based tests rely on.
        """
        transport = ServerTransport(self)
        transport.events.on(Connected, self._on_connected)
        transport.events.on(Disconnected, self._on_disconnected)
        transport.events.on(MessageReceived, self._on_message)
        return transport

    async def _on_connected(self, _event: Connected) -> None:
        self._suspect.reset()
        await self.event_bus.emit(ConnectionEstablishedEvent(self.v))
        self.logger.info("Connected to %s", self.url)

    async def _on_disconnected(self, event: Disconnected) -> None:
        # The single version-bump site: one ended attempt -> one Lost(v) -> v += 1,
        # preserving the Established(g)...Lost(g) pairing the consumer relies on.
        self.logger.debug("Emitting ConnectionLostEvent.")
        await self.event_bus.emit(ConnectionLostEvent(self.v))
        self.v += 1

        # An attempt that ended in an error (a dropped/failed socket) feeds the
        # suspect run; every Nth such failure raises an advisory ConnectionSuspectEvent.
        if event.code is not None and self._suspect.guard_until_bound():
            await self.event_bus.emit(ConnectionSuspectEvent, event.code)

    async def _on_message(self, event: MessageReceived) -> None:
        data = event.message
        if data is None:
            return
        try:
            msg = ServerMsg.model_validate_json(data).root
        except ValidationError as e:
            self.logger.error("Invalid message: %s", data, exc_info=e)
            return
        self._message_logger(msg).debug("received %s", msg)
        await self.event_bus.emit(ConnectionIncomingEvent, msg, self.v)

    def _message_logger(self, msg) -> logging.Logger:
        """Route a WS message's log line to the printer it belongs to.

        A printer-specific message goes to that printer's ``ws`` log
        (``<unique_id>/ws.log``); a genuinely global message stays on the
        connection's ``ws`` logger (the system scope). In MULTI mode each message
        carries ``for_client`` -- the printer's unique_id, or ``None`` when
        global; in SINGLE mode every message belongs to the connection's printer.
        """
        if self.hint.mode == ConnectionMode.MULTI:
            client_id = msg.for_client
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
            await self.backend.send(data)
            self._message_logger(msg).debug(
                "sent %s", data if len(data) < 1024 else msg.msg_type()
            )

        except (PydanticSerializationError, UnicodeError) as e:
            # Serialization error.
            self.logger.error("Serialization error.", exc_info=e)

        except WsConnectionErrors:
            # The link dropped mid-send; the transport recv loop sees the same drop
            # and reconnects on its own -- nothing to poke here.
            pass

    async def connect(self, hint: Optional[ConnectionHint] = None):
        """Start (or resume) the reconnecting transport.

        Idempotent -- ``start`` is a no-op while the transport is already supervising,
        and re-creates the task after a ``disconnect`` (the resume path).
        """
        self.hint = hint or self.hint

        if self.is_stopped():
            return

        if self._transport is None:
            self._transport = self._build_transport()

        self._transport.start()

    async def disconnect(self):
        """Pause (disconnect) the connection: stop the transport, no reconnection
        attempts until ``connect`` is called again."""
        if self._transport is not None:
            await self._transport.stop()

    def stop(self):
        """Permanently tear the connection down."""
        super().stop()
        if self._transport is not None:
            # Sync teardown: flag stopped and cancel the supervision task without
            # awaiting (the async stop() is for the disconnect path). The transport's
            # own finally settles the public state to DISCONNECTED.
            self._transport.stopped = True
            task = self._transport.task
            self._transport.task = None
            if task is not None and not task.done():
                task.cancel()
