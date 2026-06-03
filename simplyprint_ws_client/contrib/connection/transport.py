"""The WebSocket transport abstraction.

``Connection`` (``core.ws_protocol.connection``) owns the protocol state machine,
the version counter and the reconnect/backoff loop; it talks to the raw socket
only through a :class:`WebSocketTransport`. That keeps the socket library
swappable -- two implementations ship (:class:`WebSocketsTransport` on the
``websockets`` library, the default; :class:`AiohttpWebSocketTransport` on
aiohttp) -- and lets an integration reuse either, or implement the ABC for an
exotic socket.

The transport is deliberately dumb: open / send a text frame / receive a text
frame / close, plus a liveness flag. It owns no backoff, no version, no event
bus -- those stay in ``Connection`` where the SimplyPrint semantics live.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

__all__ = [
    "WebSocketTransport",
    "TransportFactory",
    "TransportError",
    "TransportClosed",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
]

#: WebSocket close codes the connection cares about.
WS_CLOSE_OK = 1000
WS_CLOSE_PROTOCOL_ERROR = 1002


class TransportError(Exception):
    """A transport operation failed because the socket is unusable."""


class TransportClosed(TransportError):
    """The peer closed, or the socket dropped.

    ``code`` carries the WebSocket close code when one is known.
    """

    def __init__(self, message: str = "", *, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


class WebSocketTransport(ABC):
    """A raw, protocol-agnostic async WebSocket socket.

    One instance == one physical socket attempt. ``Connection`` owns the
    reconnect/backoff loop and builds a fresh transport (via a
    :data:`TransportFactory`) per attempt.
    """

    @abstractmethod
    async def connect(self, url: str, **params) -> None:
        """Open the socket. Raise :class:`TransportError` on failure."""

    @abstractmethod
    async def send(self, data: str) -> None:
        """Send a text frame. Raise :class:`TransportClosed` if the socket is gone."""

    @abstractmethod
    async def recv(self) -> Optional[str]:
        """Return the next message as text (``None`` to skip an uninteresting frame).

        Raise :class:`TransportClosed` when the peer closed or the socket dropped.
        """

    @abstractmethod
    async def close(self, code: int = WS_CLOSE_OK, reason: str = "") -> None:
        """Close the socket. Idempotent; never raises."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the socket currently holds a live connection."""

    async def shutdown(self) -> None:
        """Release any process-wide resources held by the transport. Default no-op."""


#: Builds a fresh transport for one connection attempt, given a logger.
TransportFactory = Callable[[logging.Logger], WebSocketTransport]
