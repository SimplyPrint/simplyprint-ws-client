"""A :class:`WebSocketTransport` backed by aiohttp.

This is the library's original WebSocket stack, now expressed behind the
transport seam so it can be selected instead of (or alongside) the default
``websockets`` implementation. Useful when an integration already runs an
aiohttp stack, or to A/B the two against the backend.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    ClientWSTimeout,
    WebSocketError,
    WSMsgType,
)

from ..transport import WS_CLOSE_OK, TransportClosed, TransportError, WebSocketTransport

if TYPE_CHECKING:
    from aiohttp import ClientWebSocketResponse

__all__ = ["AiohttpWebSocketTransport"]


class AiohttpWebSocketTransport(WebSocketTransport):
    """A :class:`WebSocketTransport` backed by aiohttp's ``ws_connect``."""

    def __init__(self, logger: logging.Logger = logging.getLogger("ws")) -> None:
        self._logger = logger
        self._session: Optional[ClientSession] = None
        self._ws: Optional[ClientWebSocketResponse] = None

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
        # aiohttp maps: heartbeat = ping cadence; max_msg_size 0 = unbounded;
        # ws_close = close handshake timeout; session connect timeout = open_timeout.
        try:
            self._session = ClientSession(
                timeout=ClientTimeout(
                    total=None,
                    connect=open_timeout,
                    sock_connect=open_timeout,
                    sock_read=None,
                )
            )
            self._ws = await self._session.ws_connect(
                url,
                autoclose=True,
                autoping=True,
                heartbeat=ping_interval,
                max_msg_size=int(max_size or 0),
                timeout=ClientWSTimeout(
                    ws_receive=None, ws_close=float(close_timeout or 10)
                ),
            )
        except (ClientError, WebSocketError, OSError, asyncio.TimeoutError) as e:
            await self._close_session()
            raise TransportError(str(e)) from e

    async def send(self, data: str) -> None:
        if self._ws is None or self._ws.closed:
            raise TransportClosed("not connected")
        try:
            await self._ws.send_str(data)
        except (ConnectionError, ClientError, WebSocketError) as e:
            raise TransportClosed(str(e)) from e

    async def recv(self) -> Optional[str]:
        if self._ws is None:
            raise TransportClosed("not connected")
        message = await self._ws.receive()
        if message.type in (
            WSMsgType.CLOSE,
            WSMsgType.CLOSING,
            WSMsgType.CLOSED,
            WSMsgType.ERROR,
        ):
            raise TransportClosed(f"closed: {message.type}", code=self._ws.close_code)
        if message.type in (WSMsgType.TEXT, WSMsgType.BINARY):
            data = message.data
            return data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        return None

    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        if self._ws is not None:
            try:
                await self._ws.close(code=code, message=reason.encode("utf-8"))
            except Exception as e:  # close is best-effort; never propagate.
                self._logger.debug("Error closing aiohttp ws: %s", e)
            self._ws = None
        await self._close_session()

    @property
    def is_open(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def _close_session(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                self._logger.debug("Error closing aiohttp session: %s", e)
            self._session = None
