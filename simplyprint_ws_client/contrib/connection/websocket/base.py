"""The raw async WebSocket socket abstraction.

``Connection`` (``core.ws_protocol.connection``) owns the protocol state machine,
the version counter and the reconnect/backoff loop; it talks to the raw socket
only through a :class:`WebSocket`. That keeps the socket library swappable -- two
implementations ship (:class:`WebsocketsImpl` on the ``websockets`` library, the
default; :class:`AiohttpImpl` on aiohttp) -- and lets an integration reuse either,
or implement the ABC for an exotic socket.

The socket is deliberately dumb: open / send a text frame / receive a text frame /
close, plus a liveness flag. It owns no backoff, no version, no event bus -- those
stay in ``Connection`` where the SimplyPrint semantics live.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

__all__ = [
    "WebSocket",
    "WebSocketFactory",
    "WebSocketError",
    "WebSocketClosed",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
]

#: WebSocket close codes the connection cares about.
WS_CLOSE_OK = 1000
WS_CLOSE_PROTOCOL_ERROR = 1002


class WebSocketError(Exception):
    """A socket operation failed because the WebSocket is unusable."""


class WebSocketClosed(WebSocketError):
    """The peer closed, or the socket dropped.

    ``code`` carries the WebSocket close code when one is known.
    """

    def __init__(self, message: str = "", *, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


class WebSocket(ABC):
    """A raw, protocol-agnostic async WebSocket socket.

    One instance == one physical socket attempt. ``Connection`` owns the
    reconnect/backoff loop and builds a fresh socket (via a
    :data:`WebSocketFactory`) per attempt.
    """

    @abstractmethod
    async def connect(self, url: str, **params) -> None:
        """Open the socket. Raise :class:`WebSocketError` on failure."""

    @abstractmethod
    async def send(self, data: str) -> None:
        """Send a text frame. Raise :class:`WebSocketClosed` if the socket is gone."""

    @abstractmethod
    async def recv(self) -> Optional[str]:
        """Return the next message as text (``None`` to skip an uninteresting frame).

        Raise :class:`WebSocketClosed` when the peer closed or the socket dropped.
        """

    @abstractmethod
    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        """Close the socket. Idempotent; never raises."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the socket currently holds a live connection."""

    async def shutdown(self) -> None:
        """Release any process-wide resources held by the socket. Default no-op."""


#: Builds a fresh socket for one connection attempt, given a logger.
WebSocketFactory = Callable[[logging.Logger], WebSocket]
