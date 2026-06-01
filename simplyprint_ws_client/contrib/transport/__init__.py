"""Swappable WebSocket transports.

See :mod:`.base` for the async abstraction the SimplyPrint backend
``Connection`` uses. Two async implementations ship: :class:`WebSocketsTransport`
(the ``websockets`` library; the default) and :class:`AiohttpWebSocketTransport`
(aiohttp).

A third, :class:`ThreadedWebSocketTransport`, is the *threaded*,
websocket-client-based transport printer clients compose (it does not implement
the async ABC -- different execution model, same WebSocket wire).

The concrete implementations are imported lazily (PEP 562 ``__getattr__``) so
that selecting one transport doesn't drag the others' libraries into the import
graph -- e.g. the default ``websockets`` path never imports aiohttp or
websocket-client.
"""

from typing import TYPE_CHECKING

from .base import (
    WS_CLOSE_OK,
    WS_CLOSE_PROTOCOL_ERROR,
    TransportClosed,
    TransportError,
    TransportFactory,
    WebSocketTransport,
)

if TYPE_CHECKING:  # eager names for IDEs / type checkers
    from .aiohttp_transport import AiohttpWebSocketTransport
    from .threaded_transport import ThreadedWebSocketTransport
    from .websockets_transport import WebSocketsTransport

__all__ = [
    "WebSocketTransport",
    "WebSocketsTransport",
    "AiohttpWebSocketTransport",
    "ThreadedWebSocketTransport",
    "TransportFactory",
    "TransportError",
    "TransportClosed",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
]


def __getattr__(name: str):
    if name == "WebSocketsTransport":
        from .websockets_transport import WebSocketsTransport

        return WebSocketsTransport
    if name == "AiohttpWebSocketTransport":
        from .aiohttp_transport import AiohttpWebSocketTransport

        return AiohttpWebSocketTransport
    if name == "ThreadedWebSocketTransport":
        from .threaded_transport import ThreadedWebSocketTransport

        return ThreadedWebSocketTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
