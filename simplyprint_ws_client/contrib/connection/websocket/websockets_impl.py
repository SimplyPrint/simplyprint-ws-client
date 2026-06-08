"""A :class:`WebSocket` backed by the ``websockets`` library (default)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.protocol import State

from .base import WS_CLOSE_OK, WebSocketClosed, WebSocketError, WebSocket

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

__all__ = ["WebsocketsImpl"]


def _close_code(exc: ConnectionClosed) -> Optional[int]:
    """The close code off a ``ConnectionClosed`` without the deprecated ``.code``."""
    frame = exc.rcvd or exc.sent
    return frame.code if frame is not None else None


class WebsocketsImpl(WebSocket):
    """A :class:`WebSocket` backed by the ``websockets`` library."""

    def __init__(self, logger: logging.Logger = logging.getLogger("ws")) -> None:
        self._logger = logger
        self._conn: Optional[ClientConnection] = None

    async def connect(
        self,
        url: str,
        *,
        ping_interval: Optional[float] = 30,
        ping_timeout: Optional[float] = 30,
        open_timeout: Optional[float] = 60,
        close_timeout: Optional[float] = 10,
        max_size: Optional[int] = None,
        **_: object,
    ) -> None:
        try:
            self._conn = await ws_connect(
                url,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                open_timeout=open_timeout,
                close_timeout=close_timeout,
                max_size=max_size,
            )
        except (WebSocketException, OSError, asyncio.TimeoutError) as e:
            raise WebSocketError(str(e)) from e

    async def send(self, data: str) -> None:
        if self._conn is None:
            raise WebSocketClosed("not connected")
        try:
            await self._conn.send(data)
        except ConnectionClosed as e:
            raise WebSocketClosed(str(e), code=_close_code(e)) from e

    async def recv(self) -> Optional[str]:
        if self._conn is None:
            raise WebSocketClosed("not connected")
        try:
            message = await self._conn.recv()
        except ConnectionClosed as e:
            raise WebSocketClosed(str(e), code=_close_code(e)) from e
        if isinstance(message, bytes):
            return message.decode("utf-8", "replace")
        return message

    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        if self._conn is None:
            return
        try:
            await self._conn.close(code=code, reason=reason)
        except Exception as e:  # close is best-effort; never propagate.
            self._logger.debug("Error closing transport: %s", e)
        finally:
            self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None and self._conn.state is State.OPEN
