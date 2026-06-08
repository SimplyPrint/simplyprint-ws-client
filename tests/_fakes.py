"""Test doubles for the WebSocket transport seam.

``FakeTransport`` is an in-memory :class:`WebSocket` that lets a test
drive :class:`Connection` deterministically -- no real socket, no aiohttp/
``websockets`` involvement. ``connect`` opens it, ``send`` records into
:attr:`sent`, and ``recv`` waits on an inbox the test feeds with
:meth:`queue_message` / :meth:`queue_close`.
"""

import asyncio
from typing import List, Optional, Tuple

from simplyprint_ws_client.contrib.connection.websocket.base import (
    WebSocket,
    WebSocketClosed,
)


class FakeTransport(WebSocket):
    def __init__(self, logger=None) -> None:
        self.sent: List[str] = []
        self.connect_calls = 0
        self.closed_with: Optional[Tuple[int, str]] = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._open = False

    async def connect(self, url: str, **params) -> None:
        self.connect_calls += 1
        self._open = True

    async def send(self, data: str) -> None:
        if not self._open:
            raise WebSocketClosed("not connected")
        self.sent.append(data)

    async def recv(self) -> Optional[str]:
        item = await self._inbox.get()
        if isinstance(item, BaseException):
            # Observing a close means the socket is now dead (mirrors websockets:
            # the connection is CLOSED once recv() raises ConnectionClosed).
            self._open = False
            raise item
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed_with = (code, reason)
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> "FakeTransport":
        """Mark connected without going through the loop (for unit tests)."""
        self._open = True
        return self

    def queue_message(self, data: str) -> None:
        """Make the next ``recv()`` return ``data``."""
        self._inbox.put_nowait(data)

    def queue_close(self, code: int = 1006) -> None:
        """Make the next ``recv()`` raise ``WebSocketClosed`` (a dropped socket)."""
        self._inbox.put_nowait(WebSocketClosed("peer closed", code=code))
