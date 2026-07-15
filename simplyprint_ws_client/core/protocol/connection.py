from __future__ import annotations

__all__ = [
    "SimplyPrintConnection",
    "ConnectionHint",
    "ConnectionMode",
    "TransportFactory",
]

import asyncio
import logging
from enum import Enum
from typing import Callable, Hashable, Optional, final

from yarl import URL

from simplyprint_ws_client.core.protocol.events import SimplyPrintConnectionEvent
from simplyprint_ws_client.core.protocol.messages import ClientMsg, ClientMsgType
from simplyprint_ws_client.core.protocol.protocol import SimplyPrintProtocol
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.transport import WsTransport
from simplyprint_ws_client.wire.websockets import Websockets
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable


class ConnectionMode(Enum):
    MULTI = "mp"
    SINGLE = "p"


class ConnectionHint:
    def __init__(
        self,
        websocket_base: URL,
        mode: Optional[ConnectionMode] = None,
        config: Optional[PrinterConfig] = None,
    ) -> None:
        self.websocket_base = websocket_base
        self.mode = mode or ConnectionMode.SINGLE
        self.config = config or PrinterConfig.get_blank()

    @property
    def ws_url(self) -> URL:
        return (
            self.websocket_base
            / "0.2"
            / self.mode.value
            / str(self.config.id)
            / str(self.config.token)
        )


TransportParams = {
    "ping_interval": 30,
    "ping_timeout": 30,
    "open_timeout": 60,
    "close_timeout": 10,
    "max_size": None,
}

WsFirstMessageTimeout = 30.0

TransportFactory = Callable[
    [URL, EventLoopProvider[asyncio.AbstractEventLoop], logging.Logger], WsTransport
]


def default_transport_factory(
    url: URL,
    provider: EventLoopProvider[asyncio.AbstractEventLoop],
    logger: logging.Logger,
) -> WsTransport:
    return Websockets(
        url,
        RetryPolicy(backoff=ConstantBackoff()),
        provider,
        connect_kwargs=TransportParams,
        first_message_timeout=WsFirstMessageTimeout,
        logger=logger,
    )


@final
class SimplyPrintConnection(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop], Hashable
):
    """Stateful SimplyPrint server session.

    The transport owns link lifecycle and frames. :class:`SimplyPrintConnection` sits on top
    and owns message parsing, protocol events, and version state. ``SimplyPrintConnection`` is
    the public stateful handle that composes the two.
    """

    mode_multi = ConnectionMode.MULTI

    def __init__(
        self,
        websocket_base: URL,
        transport_factory: TransportFactory = default_transport_factory,
        hint: Optional[ConnectionHint] = None,
        logger: logging.Logger = logging.getLogger("ws"),
        *,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
    ) -> None:
        AsyncStoppable.__init__(self)
        EventLoopProvider.__init__(self, provider=provider)

        if hint is not None and hint.websocket_base != websocket_base:
            raise ValueError("connection hint websocket base does not match")
        self.websocket_base = websocket_base
        self.transport_factory = transport_factory
        self.hint = hint or ConnectionHint(websocket_base)
        self.logger = logger
        self.transport: Optional[WsTransport] = None
        self.protocol = SimplyPrintProtocol(self, logger)
        self.event_bus: EventBus[SimplyPrintConnectionEvent] = self.protocol.event_bus

    @property
    def v(self) -> int:
        return self.protocol.v

    @v.setter
    def v(self, value: int) -> None:
        self.protocol.v = value

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
        transport = self.transport
        return transport is not None and transport.connected

    @property
    def running(self):
        task = self.loop_task
        return task is not None and not task.done()

    @property
    def loop_task(self) -> Optional[asyncio.Task]:
        transport = self.transport
        if isinstance(transport, Reconnecting):
            return transport.task
        return None

    def _build_transport(self) -> WsTransport:
        return self.transport_factory(self.url, self, self.logger)

    async def send(
        self, msg: ClientMsg[ClientMsgType], v: Optional[int] = None
    ) -> None:
        await self.protocol.send(msg, v)

    async def connect(self, hint: Optional[ConnectionHint] = None):
        if hint is not None and hint.websocket_base != self.websocket_base:
            raise ValueError("connection hint websocket base does not match")
        self.hint = hint or self.hint

        if self.is_stopped():
            return

        transport = self.transport
        if transport is not None and transport.url != self.url:
            await transport.stop()
            self.protocol.detach()
            self.transport = None

        if self.transport is None:
            self.transport = self._build_transport()
            self.protocol.attach(self.transport)

        self.transport.start()

    async def disconnect(self):
        if self.transport is not None:
            await self.transport.stop()

    def stop(self):
        super().stop()
        self.protocol.detach()
        transport = self.transport
        if isinstance(transport, Reconnecting):
            transport.stopped = True
            task = transport.task
            transport.task = None
            if task is not None and not task.done():
                task.cancel()
