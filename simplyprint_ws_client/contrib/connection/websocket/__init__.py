"""The WebSocket connection family.

Two contract transports over one shared identity (:class:`WsParams`): :mod:`.sync`
(websocket-client on a daemon thread, wrapped as a :class:`~..transport.Transport`)
and :mod:`.aio` (asyncio ``websockets``/aiohttp, an :class:`~..transport.AsyncTransport`).
Beneath them sits the raw, pluggable socket -- the :class:`WebSocket` ABC in
:mod:`.base` plus the library impls (:class:`WebsocketsImpl`, :class:`AiohttpImpl`,
and the threaded :class:`ThreadedImpl`) -- which the SimplyPrint backend socket
reuses as well.

Importing this package drags none of the wire libraries: the three impl modules
load lazily (PEP 562 ``__getattr__``) and the contract families build their wire
lazily, so a base install imports cleanly.
"""

from typing import TYPE_CHECKING

from simplyprint_ws_client.contrib.connection.websocket.aio import (
    AsyncWebSocketPool,
    AsyncWebSocketTransport,
)
from simplyprint_ws_client.contrib.connection.websocket.base import (
    WS_CLOSE_OK,
    WS_CLOSE_PROTOCOL_ERROR,
    WebSocket,
    WebSocketClosed,
    WebSocketError,
    WebSocketFactory,
)
from simplyprint_ws_client.contrib.connection.websocket.common import WsParams
from simplyprint_ws_client.contrib.connection.websocket.sync import (
    WebSocketConnectionManager,
    WebSocketPool,
    WebSocketTransport,
)

if TYPE_CHECKING:  # eager names for IDEs / type checkers
    from simplyprint_ws_client.contrib.connection.websocket.aiohttp_impl import (
        AiohttpImpl,
    )
    from simplyprint_ws_client.contrib.connection.websocket.threaded_impl import (
        ThreadedImpl,
    )
    from simplyprint_ws_client.contrib.connection.websocket.websockets_impl import (
        WebsocketsImpl,
    )

__all__ = [
    "WsParams",
    "WebSocket",
    "WebSocketError",
    "WebSocketClosed",
    "WebSocketFactory",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
    "WebSocketTransport",
    "WebSocketPool",
    "WebSocketConnectionManager",
    "AsyncWebSocketTransport",
    "AsyncWebSocketPool",
    "WebsocketsImpl",
    "AiohttpImpl",
    "ThreadedImpl",
]


def __getattr__(name: str):
    # The wire libraries (websockets / aiohttp / websocket-client) stay lazy:
    # importing the package must not drag any of them.
    if name == "WebsocketsImpl":
        from simplyprint_ws_client.contrib.connection.websocket.websockets_impl import (
            WebsocketsImpl,
        )

        return WebsocketsImpl
    if name == "AiohttpImpl":
        from simplyprint_ws_client.contrib.connection.websocket.aiohttp_impl import (
            AiohttpImpl,
        )

        return AiohttpImpl
    if name == "ThreadedImpl":
        from simplyprint_ws_client.contrib.connection.websocket.threaded_impl import (
            ThreadedImpl,
        )

        return ThreadedImpl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
