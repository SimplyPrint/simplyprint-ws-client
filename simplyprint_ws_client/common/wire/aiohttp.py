"""An asyncio WebSocket transport backed by aiohttp.

:class:`Aiohttp` is a concrete :class:`~simplyprint_ws_client.common.wire.transport.WsTransport`
built on the supervised :class:`~simplyprint_ws_client.common.wire.reconnect.Reconnecting`
loop. It fills the four wire hooks the loop drives, holding the raw aiohttp objects
as plain attributes -- the :class:`~aiohttp.ClientSession` (:attr:`Aiohttp.session`)
and the live websocket response (:attr:`Aiohttp.ws`) -- with no wrapper around them:

* :meth:`~Aiohttp.open` -- open a session and a ``ws_connect``;
* :meth:`~Aiohttp.recv` -- await the next frame, surface a closed/errored socket as
  a :class:`~simplyprint_ws_client.common.wire.transport.TransientError`, and hand
  back the payload of a text/binary frame;
* :meth:`~Aiohttp.write` -- put a text or binary frame on the link;
* :meth:`~Aiohttp.aclose` -- tear the link and its session down.

It is a 1:1 wire: the pool has no route function, so every frame broadcasts to
every lease. ``aiohttp`` is imported lazily -- importing this module
never drags the wire library -- and the connect step is injectable, so a test can
drive the transport against a fake socket with no real server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Tuple, Union

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.common.wire.messages import (
    WsMessage,
    ws_message_for_payload,
)
from simplyprint_ws_client.common.wire.policy import RetryPolicy
from simplyprint_ws_client.common.wire.reconnect import Reconnecting
from simplyprint_ws_client.common.wire.transport import (
    TransientError,
    WsTransport,
)

if TYPE_CHECKING:  # eager only for type checkers; the runtime import stays lazy
    from aiohttp import ClientSession, ClientWebSocketResponse

__all__ = ["Aiohttp", "AiohttpConnectFactory", "default_aiohttp_connect"]

#: Opens a connected aiohttp websocket for a URL, awaiting to ``(session, ws)``:
#: the session and the live response, so :meth:`Aiohttp.aclose` can close both.
#: Raise to trigger a retry. Injectable so a test hands in a fake pair.
AiohttpConnectFactory = Callable[
    [yarl.URL, logging.Logger],
    Awaitable[Tuple["ClientSession", "ClientWebSocketResponse"]],
]


async def default_aiohttp_connect(
    url: yarl.URL, logger: logging.Logger, heartbeat: Optional[float] = None
) -> Tuple["ClientSession", "ClientWebSocketResponse"]:
    """Open an aiohttp session + websocket for ``url`` (the default factory).

    aiohttp is imported here, lazily, so importing this module never requires the
    wire library. A failed open raises (an aiohttp ``ClientError`` / ``OSError`` /
    timeout) after closing the half-open session, which the reconnect loop catches
    and retries from.
    """
    from aiohttp import ClientError, ClientSession
    from aiohttp import WebSocketError as AiohttpWebSocketError

    session = ClientSession()
    try:
        ws = await session.ws_connect(
            url, autoclose=True, autoping=True, heartbeat=heartbeat
        )
    except (ClientError, AiohttpWebSocketError, OSError, asyncio.TimeoutError) as error:
        await session.close()
        raise TransientError.wrap(error)
    return session, ws


class Aiohttp(WsTransport, Reconnecting):
    """A supervised aiohttp WebSocket transport.

    Construct it with the endpoint URL and (optionally) a
    :class:`~simplyprint_ws_client.common.wire.policy.RetryPolicy`, an
    :class:`~simplyprint_ws_client.common.asyncio.event_loop_provider.EventLoopProvider`
    for the supervision task, and a :data:`AiohttpConnectFactory` to open the link with
    (the default uses aiohttp; tests pass a fake). Everything else -- the connect /
    consume / reconnect cycle, state, generation, lifecycle events -- comes from
    :class:`Reconnecting`.
    """

    def __init__(
        self,
        url: yarl.URL,
        policy: Optional[RetryPolicy] = None,
        provider: Optional[EventLoopProvider] = None,
        *,
        connect_factory: AiohttpConnectFactory = default_aiohttp_connect,
        first_message_timeout: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            url,
            policy,
            provider,
            first_message_timeout=first_message_timeout,
            logger=logger or logging.getLogger("conn.aiohttp"),
        )
        self.connect_factory = connect_factory
        #: The aiohttp session backing the live socket, or ``None`` when down.
        self.session: Optional["ClientSession"] = None
        #: The live websocket response, or ``None`` when down.
        self.ws: Optional["ClientWebSocketResponse"] = None

    async def open(self) -> None:
        """Open the live session + websocket via the factory."""
        self.session, self.ws = await self.connect_factory(self.url, self.logger)

    async def recv(self) -> Optional[WsMessage]:
        """Await the next frame.

        A text or binary frame yields its payload (``str`` / ``bytes``); a
        close/closing/closed/error frame raises :class:`TransientError` to end the
        attempt (the loop reconnects); a control frame (ping/pong/continuation)
        returns ``None`` to be skipped.
        """
        ws = self.ws
        if ws is None:
            raise TransientError("websocket is not open")

        from aiohttp import WSMsgType

        frame = await ws.receive()
        kind = frame.type

        if kind in (
            WSMsgType.CLOSE,
            WSMsgType.CLOSING,
            WSMsgType.CLOSED,
            WSMsgType.ERROR,
        ):
            close_code = ws.close_code
            code = close_code if close_code is not None else int(kind)
            native_error = ws.exception() if kind == WSMsgType.ERROR else None
            raise TransientError(
                f"websocket closed: {kind}",
                code=code,
                transport_error=native_error,
            )

        if kind in (WSMsgType.TEXT, WSMsgType.BINARY):
            return ws_message_for_payload(frame.data)

        return None

    async def write(self, message: Union[str, bytes, bytearray]) -> None:
        """Put one ``message`` on the link: ``str`` as a text frame, ``bytes`` as a
        binary frame."""
        ws = self.ws
        if ws is None:
            raise TransientError("websocket is not open")
        if isinstance(message, str):
            await ws.send_str(message)
        elif isinstance(message, (bytes, bytearray)):
            await ws.send_bytes(bytes(message))
        else:
            raise TypeError(
                f"aiohttp transport sends str or bytes, not {type(message).__name__}"
            )

    async def aclose(self) -> None:
        """Tear the websocket and its session down. Idempotent and never raises."""
        ws = self.ws
        session = self.session
        self.ws = None
        self.session = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 -- aclose must never break supervision
                self.logger.debug("aiohttp %s ws close failed", self.url, exc_info=True)
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001 -- aclose must never break supervision
                self.logger.debug(
                    "aiohttp %s session close failed", self.url, exc_info=True
                )
