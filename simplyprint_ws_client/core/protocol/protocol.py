from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionEstablishedEvent,
    SimplyPrintConnectionIncomingEvent,
    SimplyPrintConnectionLostEvent,
    SimplyPrintConnectionOutgoingEvent,
    SimplyPrintConnectionSuspectEvent,
)
from simplyprint_ws_client.core.protocol.messages import (
    ClientMsg,
    ClientMsgType,
    Msg,
    ServerMsg,
)
from simplyprint_ws_client.wire import TransportError
from simplyprint_ws_client.wire.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.messages import WsMessage
from simplyprint_ws_client.wire.transport import WsTransport
from simplyprint_ws_client.common.logging import printer_logger
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.utils.bounded_variable import BoundedInterval

#: Connection-shaped failures a send may swallow. CancelledError is
#: deliberately NOT here: swallowing it would break task cancellation.
WsConnectionErrors = (
    OSError,
    ConnectionError,
    asyncio.TimeoutError,
    TransportError,
)

WsSuspectConnectionBoundedInterval = BoundedInterval[int](7, 1)


class SimplyPrintProtocol:
    """Client-side SimplyPrint WS protocol over a reconnecting transport."""

    def __init__(self, connection, logger: logging.Logger) -> None:
        self.connection = connection
        self.logger = logger
        self.event_bus = EventBus()
        self.event_bus.on(SimplyPrintConnectionOutgoingEvent, self.send)
        self.v = 0
        self._suspect = WsSuspectConnectionBoundedInterval.create_variable(0)
        self.transport: Optional[WsTransport] = None

    def attach(self, transport: WsTransport) -> None:
        self.detach()
        self.transport = transport
        transport.events.on(Connected, self._on_connected)
        transport.events.on(Disconnected, self._on_disconnected)
        transport.events.on(MessageReceived, self._on_message)

    def detach(self) -> None:
        transport = self.transport
        if transport is None:
            return
        transport.events.off(Connected, self._on_connected)
        transport.events.off(Disconnected, self._on_disconnected)
        transport.events.off(MessageReceived, self._on_message)
        self.transport = None

    async def _on_connected(self, _event: Connected) -> None:
        self._suspect.reset()
        await self.event_bus.emit(SimplyPrintConnectionEstablishedEvent(self.v))
        self.logger.info("Connected to %s", self.connection.url)

    async def _on_disconnected(self, event: Disconnected) -> None:
        self.logger.debug("Emitting SimplyPrintConnectionLostEvent.")
        await self.event_bus.emit(SimplyPrintConnectionLostEvent(self.v))
        self.v += 1

        if event.code is not None and self._suspect.guard_until_bound():
            await self.event_bus.emit(SimplyPrintConnectionSuspectEvent, event.code)

    async def _on_message(self, event: MessageReceived) -> None:
        data = self._payload(event.message)
        if data is None:
            return
        try:
            msg = ServerMsg.model_validate_json(data).root
        except ValidationError as e:
            self.logger.error("Invalid message: %s", data, exc_info=e)
            return
        self._message_logger(msg).debug("received %s", msg)
        await self.event_bus.emit(SimplyPrintConnectionIncomingEvent, msg, self.v)

    @staticmethod
    def _payload(message: object) -> Optional[str]:
        if isinstance(message, WsMessage):
            payload = message.payload
        else:
            payload = message
        if isinstance(payload, bytes):
            return payload.decode("utf-8", "replace")
        if isinstance(payload, str):
            return payload
        return None

    def _message_logger(self, msg: Msg) -> logging.Logger:
        if self.connection.hint.mode == self.connection.mode_multi:
            client_id = msg.for_client
        else:
            client_id = self.connection.hint.config.unique_id
        if client_id:
            return printer_logger(str(client_id), "ws")
        return self.logger

    async def send(
        self, msg: ClientMsg[ClientMsgType], v: Optional[int] = None
    ) -> None:
        transport = self.transport
        if transport is None or not transport.connected:
            self.logger.warning("Dropped message %s, not connected.", msg)
            return

        if v is not None and self.v != v:
            self.logger.warning(
                "Dropped message %s, version mismatch. %d != %d", msg, self.v, v
            )
            return

        try:
            data = msg.model_dump_json()
            await transport.send(data)
            self._message_logger(msg).debug(
                "sent %s", data if len(data) < 1024 else msg.msg_type()
            )
        except (PydanticSerializationError, UnicodeError) as e:
            self.logger.error("Serialization error.", exc_info=e)
        except WsConnectionErrors as e:
            # The reconnect loop owns recovery; the dropped send is only logged.
            self.logger.debug("send of %s dropped: %s", msg.msg_type(), e)
