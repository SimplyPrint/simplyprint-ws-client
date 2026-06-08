"""Async-native WebSocket transport + pool.

The printer-side async WebSocket family. The raw socket comes from
``contrib.connection.websocket`` (``base`` + the lib impls); reconnection comes from
the shared :class:`~..reconnect.ReconnectingTransport` wrapper, so this module only adapts a
raw :class:`WebSocket` to the neutral :class:`~..reconnect.Link` (one attempt) and
configures the pool. Per-client leasing / refcount is the pool's job.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Hashable, Optional

from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.utils.backoff import Backoff

from simplyprint_ws_client.contrib.connection.pool import AsyncTransportPool
from simplyprint_ws_client.contrib.connection.reconnect import (
    Link,
    ReconnectingTransport,
)
from simplyprint_ws_client.contrib.connection.websocket.base import WebSocket
from simplyprint_ws_client.contrib.connection.websocket.common import WsParams

__all__ = [
    "AsyncWebSocketPool",
    "AsyncWebSocketTransport",
    "AsyncWebSocketTransportFactory",
    "AsyncWebSocketWireFactory",
    "WsParams",
]


AsyncWebSocketWireFactory = Callable[[WsParams, logging.Logger], WebSocket]


def _default_wire_factory(_params: WsParams, logger: logging.Logger) -> WebSocket:
    from simplyprint_ws_client.contrib.connection.websocket.websockets_impl import (
        WebsocketsImpl,
    )

    return WebsocketsImpl(logger=logger)


class _WebSocketLink(Link):
    """Adapts a raw :class:`WebSocket` socket to the neutral :class:`Link` the
    reconnection engine drives (one connection attempt's worth of wire)."""

    def __init__(
        self, wire: WebSocket, url: str, connect_kwargs: Dict[str, Any]
    ) -> None:
        self._wire = wire
        self._url = url
        self._connect_kwargs = connect_kwargs

    async def open(self) -> None:
        await self._wire.connect(self._url, **self._connect_kwargs)

    async def recv(self) -> Optional[Any]:
        return await self._wire.recv()

    async def send(self, payload: Any) -> None:
        await self._wire.send(payload)

    async def close(self) -> None:
        await self._wire.close()

    @property
    def is_open(self) -> bool:
        return self._wire.is_open


class AsyncWebSocketTransport(ReconnectingTransport[WsParams]):
    """A self-healing asyncio WebSocket transport: a :class:`ReconnectingTransport` driving a
    fresh raw :class:`WebSocket` per attempt (the ``websockets``/aiohttp socket)."""

    def __init__(
        self,
        params: WsParams,
        *,
        logger: Optional[logging.Logger] = None,
        wire_factory: AsyncWebSocketWireFactory = _default_wire_factory,
        backoff: Optional[Backoff] = None,
        connect_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        log = logger or logging.getLogger("websocket.aio")
        kwargs = dict(connect_kwargs or {})

        def link_factory() -> Link:
            return _WebSocketLink(wire_factory(params, log), params.url, kwargs)

        super().__init__(params, link_factory, logger=log, backoff=backoff)


AsyncWebSocketTransportFactory = Callable[[WsParams], AsyncWebSocketTransport]


class AsyncWebSocketPool(AsyncTransportPool[WsParams, AsyncWebSocketTransport]):
    """Pool of async WebSocket transports, one live socket per endpoint."""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[AsyncWebSocketTransportFactory] = None,
        event_loop_provider: Optional[
            EventLoopProvider[asyncio.AbstractEventLoop]
        ] = None,
        connect_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._connect_kwargs = dict(connect_kwargs or {})
        super().__init__(
            logger=logger or logging.getLogger("websocket.aio.pool"),
            transport_factory=transport_factory,
            event_loop_provider=event_loop_provider,
        )

    def _build_transport(self, params: WsParams) -> AsyncWebSocketTransport:
        return AsyncWebSocketTransport(
            params,
            logger=self._logger.getChild(str(params)),
            connect_kwargs=self._connect_kwargs,
        )

    def _lease_route(self, route: Optional[Hashable]) -> Optional[Hashable]:
        return None
