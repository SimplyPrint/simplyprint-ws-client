"""Test doubles for the core WS transport seam."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

import yarl

from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    TransientError,
    WsTransport,
)
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.utils.backoff import ConstantBackoff


class FakeTransport(WsTransport, Reconnecting):
    def __init__(
        self,
        url: yarl.URL = yarl.URL("ws://fake"),
        provider: Optional[EventLoopProvider] = None,
        logger: logging.Logger = logging.getLogger("test.fake_transport"),
        *,
        first_message_timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            url,
            RetryPolicy(backoff=ConstantBackoff(0.01)),
            provider,
            first_message_timeout=first_message_timeout,
            logger=logger,
        )
        self.sent: List[str] = []
        self.connect_calls = 0
        self.closed_with: Optional[Tuple[int, str]] = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._open = False

    async def open(self) -> None:
        self.connect_calls += 1
        self._open = True

    async def write(self, data: object) -> None:
        if not self._open:
            raise TransientError("not connected")
        self.sent.append(str(data))

    async def recv(self) -> Optional[str]:
        item = await self._inbox.get()
        if isinstance(item, BaseException):
            self._open = False
            raise item
        return item

    async def aclose(self) -> None:
        self.closed_with = (1000, "")
        self._open = False

    def open_for_test(self) -> "FakeTransport":
        self._open = True
        self.live = True
        self.state = ConnectionState.CONNECTED
        return self

    def queue_message(self, data: str) -> None:
        self._inbox.put_nowait(data)

    def queue_close(self) -> None:
        self._inbox.put_nowait(TransientError("peer closed"))
