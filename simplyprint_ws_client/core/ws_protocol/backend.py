"""The raw async backend transport the SimplyPrint protocol connection drives.

:class:`~simplyprint_ws_client.core.ws_protocol.connection.Connection` owns the
SimplyPrint protocol -- the version counter, ``ServerMsg`` parsing, version-targeted
send -- and rides the brand-free :mod:`simplyprint_ws_client.contrib.connection` reconnect
loop for the backend lifecycle. The one thing it still keeps in hand is the raw
transport: open / send a text frame / receive a text frame / close, plus a liveness
flag. That is what a :class:`BackendTransport` is.

The backend is deliberately dumb -- no backoff, no version, no event bus -- so the
underlying library stays swappable. Two implementations ship: :class:`WebsocketsBackend`
on the ``websockets`` library (the default) and :class:`AiohttpBackend` on aiohttp.
Each impl imports its wire library lazily, so importing this module pulls in neither.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from aiohttp import ClientSession, ClientWebSocketResponse
    from websockets.asyncio.client import ClientConnection
    from websockets.exceptions import ConnectionClosed

__all__ = [
    "BackendTransport",
    "BackendFactory",
    "BackendError",
    "BackendClosed",
    "WebsocketsBackend",
    "AiohttpBackend",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
]

#: WebSocket close codes the connection cares about.
WS_CLOSE_OK = 1000
WS_CLOSE_PROTOCOL_ERROR = 1002


class BackendError(Exception):
    """A backend operation failed because the WebSocket is unusable."""


class BackendClosed(BackendError):
    """The peer closed, or the backend dropped.

    ``code`` carries the WebSocket close code when one is known.
    """

    def __init__(self, message: str = "", *, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


class BackendTransport(ABC):
    """A raw, protocol-agnostic async backend transport.

    One instance == one physical connection attempt. The protocol connection owns
    the reconnect/backoff loop and builds a fresh backend via a
    :data:`BackendFactory` per attempt.
    """

    @abstractmethod
    async def connect(self, url: str, **params) -> None:
        """Open the backend. Raise :class:`BackendError` on failure."""

    @abstractmethod
    async def send(self, data: str) -> None:
        """Send a text frame. Raise :class:`BackendClosed` if the backend is gone."""

    @abstractmethod
    async def recv(self) -> Optional[str]:
        """Return the next message as text (``None`` to skip an uninteresting frame).

        Raise :class:`BackendClosed` when the peer closed or the backend dropped.
        """

    @abstractmethod
    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        """Close the backend. Idempotent; never raises."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the backend currently holds a live connection."""

    async def shutdown(self) -> None:
        """Release any process-wide resources held by the socket. Default no-op."""


#: Builds a fresh backend for one connection attempt, given a logger.
BackendFactory = Callable[[logging.Logger], BackendTransport]


def websockets_close_code(error: "ConnectionClosed") -> Optional[int]:
    """The close code off a ``websockets`` ``ConnectionClosed`` without the
    deprecated ``.code`` attribute."""
    frame = error.rcvd or error.sent
    return frame.code if frame is not None else None


class WebsocketsBackend(BackendTransport):
    """A :class:`BackendTransport` backed by the ``websockets`` library.

    The wire library is imported lazily on first connect, so referencing this class
    (it is the default :data:`BackendFactory`) costs nothing at import time.
    """

    def __init__(self, logger: logging.Logger = logging.getLogger("ws")) -> None:
        self.logger = logger
        self.conn: Optional["ClientConnection"] = None

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
        import asyncio

        from websockets.asyncio.client import connect as ws_connect
        from websockets.exceptions import WebSocketException

        try:
            self.conn = await ws_connect(
                url,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                open_timeout=open_timeout,
                close_timeout=close_timeout,
                max_size=max_size,
            )
        except (WebSocketException, OSError, asyncio.TimeoutError) as e:
            raise BackendError(str(e)) from e

    async def send(self, data: str) -> None:
        from websockets.exceptions import ConnectionClosed

        if self.conn is None:
            raise BackendClosed("not connected")
        try:
            await self.conn.send(data)
        except ConnectionClosed as e:
            raise BackendClosed(str(e), code=websockets_close_code(e)) from e

    async def recv(self) -> Optional[str]:
        from websockets.exceptions import ConnectionClosed

        if self.conn is None:
            raise BackendClosed("not connected")
        try:
            message = await self.conn.recv()
        except ConnectionClosed as e:
            raise BackendClosed(str(e), code=websockets_close_code(e)) from e
        if isinstance(message, bytes):
            return message.decode("utf-8", "replace")
        return message

    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        if self.conn is None:
            return
        try:
            await self.conn.close(code=code, reason=reason)
        except Exception as e:  # close is best-effort; never propagate.
            self.logger.debug("Error closing transport: %s", e)
        finally:
            self.conn = None

    @property
    def is_open(self) -> bool:
        if self.conn is None:
            return False
        from websockets.protocol import State

        return self.conn.state is State.OPEN


class AiohttpBackend(BackendTransport):
    """A :class:`BackendTransport` backed by aiohttp's ``ws_connect``.

    The alternative to :class:`WebsocketsBackend` for an integration already running an
    aiohttp stack, or to A/B the two against the backend. aiohttp is imported lazily
    on first connect.
    """

    def __init__(self, logger: logging.Logger = logging.getLogger("ws")) -> None:
        self.logger = logger
        self.session: Optional["ClientSession"] = None
        self.ws: Optional["ClientWebSocketResponse"] = None

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
        import asyncio

        from aiohttp import (
            ClientError,
            ClientSession,
            ClientTimeout,
            ClientWSTimeout,
        )
        from aiohttp import WebSocketError as AiohttpWebSocketError

        # aiohttp maps: heartbeat = ping cadence; max_msg_size 0 = unbounded;
        # ws_close = close handshake timeout; session connect timeout = open_timeout.
        try:
            self.session = ClientSession(
                timeout=ClientTimeout(
                    total=None,
                    connect=open_timeout,
                    sock_connect=open_timeout,
                    sock_read=None,
                )
            )
            self.ws = await self.session.ws_connect(
                url,
                autoclose=True,
                autoping=True,
                heartbeat=ping_interval,
                max_msg_size=int(max_size or 0),
                timeout=ClientWSTimeout(
                    ws_receive=None, ws_close=float(close_timeout or 10)
                ),
            )
        except (ClientError, AiohttpWebSocketError, OSError, asyncio.TimeoutError) as e:
            await self.close_session()
            raise BackendError(str(e)) from e

    async def send(self, data: str) -> None:
        from aiohttp import ClientError
        from aiohttp import WebSocketError as AiohttpWebSocketError

        if self.ws is None or self.ws.closed:
            raise BackendClosed("not connected")
        try:
            await self.ws.send_str(data)
        except (ConnectionError, ClientError, AiohttpWebSocketError) as e:
            raise BackendClosed(str(e)) from e

    async def recv(self) -> Optional[str]:
        from aiohttp import WSMsgType

        if self.ws is None:
            raise BackendClosed("not connected")
        message = await self.ws.receive()
        if message.type in (
            WSMsgType.CLOSE,
            WSMsgType.CLOSING,
            WSMsgType.CLOSED,
            WSMsgType.ERROR,
        ):
            raise BackendClosed(f"closed: {message.type}", code=self.ws.close_code)
        if message.type in (WSMsgType.TEXT, WSMsgType.BINARY):
            data = message.data
            return data.decode("utf-8", "replace") if isinstance(data, bytes) else data
        return None

    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        if self.ws is not None:
            try:
                await self.ws.close(code=code, message=reason.encode("utf-8"))
            except Exception as e:  # close is best-effort; never propagate.
                self.logger.debug("Error closing aiohttp ws: %s", e)
            self.ws = None
        await self.close_session()

    @property
    def is_open(self) -> bool:
        return self.ws is not None and not self.ws.closed

    async def close_session(self) -> None:
        if self.session is not None:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.debug("Error closing aiohttp session: %s", e)
            self.session = None
