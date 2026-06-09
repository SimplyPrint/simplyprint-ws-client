"""A WebSocket wire backed by the ``websockets`` library.

:class:`Websockets` is a 1:1 WebSocket :class:`~simplyprint_ws_client.contrib.connection.transport.WsTransport`
built on the supervised :class:`~simplyprint_ws_client.contrib.connection.reconnect.Reconnecting`
loop: the base owns the connect/consume/drop/backoff cycle and the lifecycle
events, and this class fills the four wire hooks on itself --
:meth:`~Websockets.open` opens the live connection, :meth:`~Websockets.recv`
yields one inbound frame, :meth:`~Websockets.write` puts one frame on the wire,
and :meth:`~Websockets.aclose` tears it down. There is no separate socket object:
the live ``websockets`` connection is a plain attribute (:attr:`Websockets.socket`)
and the hooks call it directly.

The whole thing runs natively on the provider's asyncio loop, so it never touches
a thread and needs no courier. A dropped connection surfaces as
:class:`~simplyprint_ws_client.contrib.connection.transport.TransientError`, which ends
the attempt and lets the loop reconnect.

The ``websockets`` library is imported lazily, only when a connection is actually
opened, so importing this module costs nothing and pulls in no wire library. Tests
inject a fake ``connect`` factory (one that yields a driveable in-memory socket)
and never need a real server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Union

import yarl

from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.messages import (
    WsMessage,
    ws_message_for_payload,
)
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
from simplyprint_ws_client.contrib.connection.transport import (
    TransientError,
    WsTransport,
)

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

__all__ = ["Websockets", "ConnectFactory", "default_websockets_connect"]

#: Opens a live ``websockets`` connection to a URL, awaiting to the library's
#: ``ClientConnection``. ``websockets.asyncio.client.connect`` is the production
#: factory; a test passes a fake of the same shape (an awaitable yielding an object
#: with ``recv`` / ``send`` / ``close`` coroutines). Extra keyword arguments
#: (``open_timeout``, ``ping_interval`` ...) flow straight through.
ConnectFactory = Callable[..., Awaitable["ClientConnection"]]


def default_websockets_connect() -> ConnectFactory:
    """Lazily resolve ``websockets.asyncio.client.connect``.

    Imported only on first use so this module (and the package ``__init__`` that
    re-exports the contracts) never eager-loads the wire library.
    """
    from websockets.asyncio.client import connect

    return connect


class Websockets(WsTransport, Reconnecting):
    """A self-healing 1:1 WebSocket wire over the ``websockets`` library.

    Construct it with the endpoint URL and, optionally, a
    :class:`~simplyprint_ws_client.contrib.connection.policy.RetryPolicy`, the
    :class:`~simplyprint_ws_client.shared.asyncio.event_loop_provider.EventLoopProvider`
    whose loop the supervision task runs on, a ``connect`` factory (defaulting to
    the library's, resolved lazily), and connect keyword arguments passed to that
    factory on every attempt. :meth:`start` it and drive it by events;
    :meth:`route` stays ``None`` so every frame broadcasts to every lease.
    """

    def __init__(
        self,
        url: yarl.URL,
        policy: Optional[RetryPolicy] = None,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        connect_factory: Optional[ConnectFactory] = None,
        connect_kwargs: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            url, policy, provider, logger=logger or logging.getLogger("conn.websockets")
        )
        self.connect_factory = connect_factory
        self.connect_kwargs = dict(connect_kwargs or {})
        #: The live ``websockets`` connection, or ``None`` while disconnected.
        self.socket: Optional[Any] = None

    async def open(self) -> None:
        """Open the live connection. A failure raises and triggers a retry."""
        factory = self.connect_factory or default_websockets_connect()
        try:
            self.socket = await factory(str(self.url), **self.connect_kwargs)
        except Exception as error:  # noqa: BLE001 -- any wire failure -> retry
            raise TransientError(str(error)) from error

    async def recv(self) -> WsMessage:
        """Return the next inbound frame; a closed socket ends the attempt.

        A ``websockets`` ``ConnectionClosed`` (the wire dropped) is mapped to a
        :class:`~simplyprint_ws_client.contrib.connection.transport.TransientError` so the
        supervision loop tears down and reconnects.
        """
        socket = self.socket
        if socket is None:
            raise TransientError("websocket not connected")
        try:
            return ws_message_for_payload(await socket.recv())
        except Exception as error:  # noqa: BLE001 -- ConnectionClosed et al. -> retry
            raise TransientError(str(error)) from error

    async def write(self, message: object) -> None:
        """Put one frame on the live wire.

        ``str`` is sent as a text frame and ``bytes`` as a binary frame. The
        WebSocket front door is responsible for reducing its ``WsMessage`` family
        to a bare ``str``/``bytes`` before it reaches here, so this transport never
        imports the framing types it does not own.
        """
        socket = self.socket
        if socket is None:
            raise TransientError("websocket not connected")
        await socket.send(as_frame(message))

    async def aclose(self) -> None:
        """Close the live wire. Idempotent and never raises."""
        socket = self.socket
        self.socket = None
        if socket is None:
            return
        try:
            await socket.close()
        except Exception:  # noqa: BLE001 -- aclose must never break supervision
            self.logger.debug("websocket %s close failed", self.url, exc_info=True)


def as_frame(message: object) -> Union[str, bytes]:
    """Reduce an outbound message to the ``str``/``bytes`` the wire accepts.

    A ``str`` is a text frame; ``bytes`` (or ``bytearray``/``memoryview``) is a
    binary frame. Anything else is the WebSocket front door's to unwrap before it
    reaches this transport, so a stray wrapper is a programming error here.
    """
    if isinstance(message, str):
        return message
    if isinstance(message, (bytes, bytearray, memoryview)):
        return bytes(message)
    raise TypeError(f"cannot send {type(message).__name__} over a websocket")
