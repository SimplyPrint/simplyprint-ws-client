__all__ = ["Connection", "ConnectionHint", "ConnectionMode"]

import asyncio
import logging
from enum import Enum, auto
from typing import Optional, final, Hashable

from aiohttp import (
    ClientWebSocketResponse,
    ClientSession,
    WSMsgType,
    ClientError,
    WebSocketError,
    ClientTimeout,
    ClientWSTimeout,
    WSCloseCode,
)
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from yarl import URL

from .events import (
    ConnectionEvent,
    ConnectionIncomingEvent,
    ConnectionOutgoingEvent,
    ConnectionEstablishedEvent,
    ConnectionLostEvent,
    ConnectionSuspectEvent,
)
from .messages import ClientMsg, ServerMsg, ClientMsgType
from ..config import PrinterConfig
from ...events import EventBus
from ...shared.asyncio.continuous_task import ContinuousTask
from ...shared.asyncio.event_loop_provider import EventLoopProvider
from ...shared.sp.url_builder import SimplyPrintURL
from ...shared.utils.backoff import ConstantBackoff
from ...shared.utils.bounded_variable import BoundedInterval
from ...shared.utils.stoppable import AsyncStoppable


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


class Action(Enum):
    INTERRUPT = auto()
    PAUSE = auto()
    RESUME = auto()

    def transition(self, state: "State"):
        return {
            self.PAUSE: State.PAUSED,
            self.RESUME: State.NOT_CONNECTED,
        }.get(self, state)


class State(Enum):
    NOT_CONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    PAUSED = auto()


ConnectionTimeout = ClientTimeout(
    total=None, connect=60.0, sock_connect=60.0, sock_read=None
)

# aiohttp WebSocket parameters.
# https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession.ws_connect
WsParams = {
    "autoclose": True,
    "autoping": True,
    "heartbeat": 30,
    "max_msg_size": 0,
    "timeout": ClientWSTimeout(ws_receive=None, ws_close=10.0),
}

# Errors we treat as a closed connection.
WsConnectionErrors = (
    OSError,
    ConnectionError,  # Technically a subset of OSError, but more specific.
    ClientError,
    asyncio.TimeoutError,
    asyncio.CancelledError,
    WebSocketError,
)

WsSuspectConnectionBoundedInterval = BoundedInterval[int](7, 1)
WsFirstMessageTimeout = 30.0


@final
class Connection(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop], Hashable
):
    """Underlying connection to the SimplyPrint server. Manages the WebSocket connection and dispatches
    both event and messages to power client functionality. Receives outgoing messages, and is stateless
    by design.

    It manages one primary task that is the main `connect` loop, which is created on its first invocation.
    Additional calls to `connect` and `disconnect` modify the behavior of the initial task. It is then finally
    fully stopped (permanently) by calling `stop` from `AsyncStoppable`.

    Attributes:
        v: Version representing the connection generation, on every disconnect this value is incremented to invalidate previous versions.
        ws: WebSocket connection.
        session: HTTP session.
        hint: Connection hint from which the URL is derived.
        logger: Logger for connection events.
        event_bus: ConnectionEvent bus for lifetime events and message events.
        _state: State of connection, allows us to resume whatever action was interrupted.
        _queue: Allows us to signal the main loop to either pause, resume or re-check connectivity.
        _loop_task: Main task that manages the connection loop.
    """

    v: int
    ws: Optional[ClientWebSocketResponse]
    session: Optional[ClientSession]
    hint: ConnectionHint
    logger: logging.Logger

    event_bus: EventBus[ConnectionEvent]

    _state: State
    _queue: asyncio.Queue
    _loop_task: ContinuousTask[None]

    def __init__(
        self,
        session: Optional[ClientSession] = None,
        hint: Optional[ConnectionHint] = None,
        logger: logging.Logger = logging.getLogger("ws"),
        **kwargs,
    ):
        AsyncStoppable.__init__(self, **kwargs)
        EventLoopProvider.__init__(self, **kwargs)

        self.v = 0
        self.ws = None
        self.session = session
        self.hint = hint or ConnectionHint()
        self.logger = logger

        self.event_bus = EventBus[ConnectionEvent]()
        self.event_bus.on(ConnectionOutgoingEvent, self.send)

        self._state = State.NOT_CONNECTED
        self._queue = asyncio.Queue()
        self._loop_task = ContinuousTask(self._loop, provider=self)

    def __hash__(self):
        return hash(id(self))

    def __await__(self):
        return self._loop_task.task.__await__()

    @property
    def url(self) -> URL:
        return self.hint.ws_url

    @property
    def connected(self):
        """This has nothing to do with our `Connection` state and everything to do
        with the real, physical underlying connection state."""
        return self.ws is not None and not self.ws.closed

    @property
    def running(self):
        """Whether the connection loop is running."""
        return self._loop_task.task is not None and not self._loop_task.done()

    async def _close_ws(self, code: int = WSCloseCode.OK, message: bytes = b""):
        """
        Close WebSocket connection manually, typically used when we are paused or
        stopped and no reconnections are taking place.
        """
        if not self.connected:
            return
        await self.ws.close(code=code, message=message)
        self.ws = None
        self.logger.debug("Emitting ConnectionLostEvent due to manual close.")
        self._state = State.NOT_CONNECTED
        _ = self.event_bus.emit_task(ConnectionLostEvent(self.v))
        self.v += 1
        self.logger.info("Manually closed connection.")

    async def _loop(self):
        """Connection main loop - only run once per instance."""
        if self._loop_task.task != asyncio.current_task():
            raise RuntimeError("Connection task already running.")

        backoff = ConstantBackoff()
        suspect_bound = WsSuspectConnectionBoundedInterval.create_variable(0)
        action: Optional[Action] = None

        queue_task = ContinuousTask(self._queue.get, provider=self)
        poll_task = ContinuousTask(self.poll, provider=self)
        ws_connect_task = ContinuousTask(
            lambda: self.session.ws_connect(self.url, **WsParams), provider=self
        )
        wait_delay_task = ContinuousTask(lambda d: self.wait(d), provider=self)
        wait_stop_task = ContinuousTask(self.wait, provider=self)

        wait_first_msg_task = ContinuousTask(lambda d: self.wait(d), provider=self)
        has_first_msg = False

        # Flat main loop. Handles all states correctly.
        # First, PAUSED is excluded and handled.
        # Then NOT_CONNECTED is transformed to CONNECTING.
        # Then CONNECTING sis transformed to CONNECTED.
        # In the case where `send` sets connected = False we get an interrupt,
        # call `poll()` which will fail for us, which then sets the state to NOT_CONNECTED.
        while not self.is_stopped():
            try:
                # Deal with any actions that were queued.
                if action is not None:
                    self._state = action.transition(self._state)
                    action = None
                    continue

                # When we are paused we just have to wait until we are resumed.
                if self._state == State.PAUSED:
                    # Stop connecting and polling while paused.
                    poll_task.discard()
                    ws_connect_task.discard()
                    wait_delay_task.discard()

                    # Disconnect if we are paused and connected.
                    await self._close_ws()

                    self.logger.info("Paused.")

                    # Wait for next action.
                    await queue_task
                    continue

                # Once we are in this state we need to be connected
                # and actively poll the connection.

                # We split up connection into two phases both managing
                # a shared `ws_connect_task`.

                # Phase 1: Goal `CONNECTING`: When we are not in progress of connecting,
                # we need to start a new connection. Clear previous connection task,
                # deal with backoff and schedule the connection attempt.
                # This can be interrupted by a queue action.
                if not self.connected and self._state == State.NOT_CONNECTED:
                    ws_connect_task.discard()

                    # All connections except the first are subject to backoff.
                    if self.v != 0:
                        resumed = wait_delay_task.task is not None

                        if not resumed:
                            delay = backoff.delay()
                            self.logger.info(f"Reconnecting in {delay} seconds.")
                        else:
                            delay = None
                            self.logger.info("Reconnecting (resumed).")

                        await asyncio.wait(
                            [
                                wait_delay_task.schedule(
                                    delay
                                ),  # wait task with delay (sleep for delay seconds)
                                queue_task.schedule(),
                            ],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        # Handle more immediate actions.
                        if self.is_stopped() or queue_task.done():
                            continue

                        # Either way next time we hit this block the delay is over.
                        wait_delay_task.discard()

                    self.logger.info(f"Connecting to {self.url}")

                    self._state = State.CONNECTING

                # Phase 2: Goal `CONNECTED`: When we are in progress with connecting,
                # we need to wait for the connection to be established.
                # this can also be interrupted by a queue action.
                if not self.connected and self._state == State.CONNECTING:
                    await asyncio.wait(
                        [
                            ws_connect_task.schedule(),
                            queue_task.schedule(),
                            wait_stop_task.schedule(),
                            # wait task without any delay (wait forever for stop event)
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Queue action interrupted us, deal with it.
                    if not ws_connect_task.done():
                        continue

                    self.ws = ws_connect_task.pop().result()

                    self._state = State.CONNECTED

                    backoff.reset()
                    suspect_bound.reset()
                    # Begin waiting for first message.
                    wait_first_msg_task.discard()
                    has_first_msg = False

                    _ = self.event_bus.emit_task(ConnectionEstablishedEvent(self.v))

                    self.logger.info(f"Connected to {self.url}")

                if not self.connected:
                    raise ConnectionResetError(
                        f"Invalid connection state. Previous close code: {self.ws.close_code if self.ws is not None else None}"
                    )

                tasks = [queue_task.schedule(), poll_task.schedule()]

                # Wait for up to 30 seconds for first message, after we have connected.
                if not has_first_msg:
                    tasks.append(wait_first_msg_task.schedule(WsFirstMessageTimeout))

                # In this state we are connected and actively polling the connection.
                # Poll for messages and interrupt on queue action.
                await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Ensure all exceptions from poll task are propagated so we can detect connection closure.
                # This is allowed to fail to trigger a reconnect.
                if poll_task.done():
                    poll_task.pop().result()
                    has_first_msg = True

                # Reset connection if waiter is done without having received first message.
                if not has_first_msg and wait_first_msg_task.done():
                    await self._close_ws(
                        code=WSCloseCode.PROTOCOL_ERROR,
                        message=b"Did not receive first message in time.",
                    )

            except WsConnectionErrors as e:
                # Disconnect / No connection handling
                self._state = State.NOT_CONNECTED
                _ = self.event_bus.emit_task(ConnectionLostEvent(self.v))
                self.v += 1
                self.logger.info("%s: %s", type(e), e)

                # For repeated connection failures, we suspect the ability to connect might be compromised.
                # This could be due to loss of network connectivity, server issues, etc.
                # To allow external users to react to this, we emit a suspect event.
                if suspect_bound.guard_until_bound():
                    _ = self.event_bus.emit_task(ConnectionSuspectEvent, e)
            except Exception as e:
                self.logger.error("Other error.", exc_info=e)

            finally:
                # If an action arrived while we were polling, we need to handle it.
                if queue_task.done() and queue_task.exception() is None:
                    action = queue_task.pop().result()
                    self._queue.task_done()

        # Clean up.
        await self._close_ws()

        if self.session:
            await self.session.close()
            self.session = None

        poll_task.discard()
        ws_connect_task.discard()
        wait_delay_task.discard()
        wait_stop_task.discard()

        self.logger.info("Connection stopped.")

    async def connect(self, hint: Optional[ConnectionHint] = None):
        """Create or resume the connection loop."""
        if not self.session:
            self.session = ClientSession(timeout=ConnectionTimeout)

        self.hint = hint or self.hint

        # Task is already running.
        if self.running:
            # Resume the task if it is paused.
            if self._state == State.PAUSED:
                await self._queue.put(Action.RESUME)

            return

        # If the loop task is done, we need to pop it
        # before being able to restart it. Note this will do
        # nothing if the connection is stopped.
        if self._loop_task.done():
            self._loop_task.discard()

        # Ensure connection loop is running.
        self._loop_task.schedule()

    async def disconnect(self):
        """Pause (disconnect) the connection loop, no reconnection attempts will be made until resumed."""
        if self.running and self._state != State.PAUSED:
            await self._queue.put(Action.PAUSE)

    async def interrupt(self):
        """Issue an interrupt to the connection thread, which makes it check for connection state changes."""
        if self.running:
            await self._queue.put(Action.INTERRUPT)

    async def poll(self) -> None:
        """
        Raises:
         ConnectionResetError: Signal that the connection is closed.
        """
        message = await self.ws.receive()

        if message.type in (
            WSMsgType.CLOSE,
            WSMsgType.CLOSING,
            WSMsgType.CLOSED,
            WSMsgType.ERROR,
        ):
            raise ConnectionResetError(f"Connection closed. {message}")

        try:
            if message.type in (WSMsgType.TEXT, WSMsgType.BINARY):
                msg = ServerMsg.model_validate_json(message.data).root
                self.logger.debug("received %s", msg)
                _ = self.event_bus.emit_task(ConnectionIncomingEvent, msg, self.v)
                return

            self.logger.warning("Unhandled message: %s", message)

        except ValidationError as e:
            # Invalid message.
            self.logger.error("Invalid message: %s", message, exc_info=e)

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
            await self.interrupt()
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
            await self.ws.send_str(data)
            self.logger.debug("sent %s", data if len(data) < 1024 else msg.msg_type())

        except (PydanticSerializationError, UnicodeError) as e:
            # Serialization error.
            self.logger.error("Serialization error.", exc_info=e)

        except WsConnectionErrors:
            await self.interrupt()

    def stop(self):
        super().stop()

        if self.event_loop_is_running():
            asyncio.run_coroutine_threadsafe(self.interrupt(), self.event_loop)
