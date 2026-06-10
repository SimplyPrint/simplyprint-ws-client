from __future__ import annotations

__all__ = ["CloudConnection", "ConnectionHint", "ConnectionMode", "TransportFactory"]

import asyncio
import logging
from enum import Enum
from typing import Callable, Hashable, Optional, final

from yarl import URL

from simplyprint_ws_client.core.protocol.events import CloudConnectionEvent
from simplyprint_ws_client.core.protocol.messages import ClientMsg, ClientMsgType
from simplyprint_ws_client.core.protocol.protocol import CloudProtocol
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.transport import WsTransport
from simplyprint_ws_client.wire.websockets import Websockets
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.core.api.url_builder import SimplyPrintURL
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable


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
class CloudConnection(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop], Hashable
):
    """Stateful SimplyPrint server session.

    The transport owns link lifecycle and frames. :class:`CloudProtocol` sits on top
    and owns message parsing, protocol events, and version state. ``CloudConnection`` is
    the public stateful handle that composes the two.
    """

    mode_multi = ConnectionMode.MULTI

    def __init__(
        self,
        transport_factory: TransportFactory = default_transport_factory,
        hint: Optional[ConnectionHint] = None,
        logger: logging.Logger = logging.getLogger("ws"),
        **kwargs,
    ):
        AsyncStoppable.__init__(self, **kwargs)
        EventLoopProvider.__init__(self, **kwargs)

        self.transport_factory = transport_factory
        self.hint = hint or ConnectionHint()
        self.logger = logger
        self.transport: Optional[WsTransport] = None
        self.protocol = CloudProtocol(self, logger)
        self.event_bus: EventBus[CloudConnectionEvent] = self.protocol.event_bus

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
