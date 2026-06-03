"""Concrete async :class:`~..transport.WebSocketTransport` wire leaves.

Two ship: :class:`WebSocketsTransport` (the ``websockets`` library; the default)
and :class:`AiohttpWebSocketTransport` (aiohttp). They are imported lazily (PEP
562 ``__getattr__``) so selecting one never drags the other's library into the
import graph.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # eager names for IDEs / type checkers
    from .aiohttp import AiohttpWebSocketTransport
    from .websockets import WebSocketsTransport

__all__ = [
    "WebSocketsTransport",
    "AiohttpWebSocketTransport",
]


def __getattr__(name: str):
    if name == "WebSocketsTransport":
        from .websockets import WebSocketsTransport

        return WebSocketsTransport
    if name == "AiohttpWebSocketTransport":
        from .aiohttp import AiohttpWebSocketTransport

        return AiohttpWebSocketTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
