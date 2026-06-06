"""The raw async WebSocket wire, inside the connection subsystem.

This is the low-level, *dumb* socket the SimplyPrint backend ``Connection``
(:mod:`core.ws_protocol.connection`) drives: open / send / recv / close /
liveness, nothing more. The backend still owns its protocol state machine,
version counter and reconnect loop on top (until that loop folds into the
supervised :class:`~simplyprint_ws_client.contrib.connection.transport.AsyncTransport`).

* :class:`WebSocketTransport` -- the async ABC, plus the
  :class:`TransportError` / :class:`TransportClosed` vocabulary, the WS close
  codes, and the :data:`TransportFactory` alias. These load eagerly and carry no
  third-party imports.
* the two shipped wire leaves -- :class:`WebSocketsTransport` (the ``websockets``
  library, the default) and :class:`AiohttpWebSocketTransport` (aiohttp) --
  imported lazily (PEP 562 ``__getattr__``) so selecting one never drags the
  other's library into the import graph.

It lives under ``connection`` because the wire is part of the connection
subsystem, not a sibling of it.
"""

from typing import TYPE_CHECKING

from simplyprint_ws_client.contrib.connection.wire.base import (
    WS_CLOSE_OK,
    WS_CLOSE_PROTOCOL_ERROR,
    TransportClosed,
    TransportError,
    TransportFactory,
    WebSocketTransport,
)

if TYPE_CHECKING:  # eager names for IDEs / type checkers
    from simplyprint_ws_client.contrib.connection.wire.aiohttp import (
        AiohttpWebSocketTransport,
    )
    from simplyprint_ws_client.contrib.connection.wire.websockets import (
        WebSocketsTransport,
    )

__all__ = [
    "AiohttpWebSocketTransport",
    "TransportClosed",
    "TransportError",
    "TransportFactory",
    "WebSocketsTransport",
    "WebSocketTransport",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
]


def __getattr__(name: str):
    if name == "WebSocketsTransport":
        from simplyprint_ws_client.contrib.connection.wire.websockets import (
            WebSocketsTransport,
        )

        return WebSocketsTransport
    if name == "AiohttpWebSocketTransport":
        from simplyprint_ws_client.contrib.connection.wire.aiohttp import (
            AiohttpWebSocketTransport,
        )

        return AiohttpWebSocketTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
