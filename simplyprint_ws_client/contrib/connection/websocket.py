"""The WebSocket front door: a 1:1 self-healing socket, pooled by endpoint.

A printer client reaches a WebSocket endpoint through one call::

    from simplyprint_ws_client.contrib.connection import ws
    conn = ws.connect(yarl.URL("wss://host/path"))
    conn.event_bus.on(MessageReceived, handler)
    await conn.ready()
    await conn.send(WsTextMessage("hello"))     # or just: await conn.send("hello")
    await conn.close()

The wire underneath is one of two async libraries -- ``websockets`` (the default)
or ``aiohttp`` -- chosen by ``impl``. Both live in their own modules and are
imported here, not re-implemented; each imports its library lazily, so importing
this module never requires either to be installed.

This module owns the framing the brand-free wires deliberately do not: the
:class:`WsMessage` family. A wire carries a frame as a bare ``str``/``bytes``; the
front-door :class:`WsConnection` lease wraps an inbound frame into a
:class:`WsMessage` for handlers and reduces an outbound :class:`WsMessage` back to
``str``/``bytes`` for the wire. A per-``impl``
:class:`~simplyprint_ws_client.contrib.connection.pool.Pool` keyed by the full URL shares
one socket across every lease on the same endpoint.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import yarl

from simplyprint_ws_client.contrib.connection.aiohttp import Aiohttp
from simplyprint_ws_client.contrib.connection.connection import WsConnection
from simplyprint_ws_client.contrib.connection.messages import (
    WsBytesMessage,
    WsKind,
    WsMessage,
    WsTextMessage,
    as_ws_message,
    ws_message_for_payload,
)
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.pool import Pool
from simplyprint_ws_client.contrib.connection.transport import WsTransport
from simplyprint_ws_client.contrib.connection.websockets import Websockets

__all__ = [
    "WsKind",
    "WsMessage",
    "WsTextMessage",
    "WsBytesMessage",
    "WsConnection",
    "connect",
    "shutdown",
]

#: The two shipped wire implementations, by name.
SUPPORTED_IMPLS = ("websockets", "aiohttp")


def as_message(message: Union[str, bytes, WsMessage]) -> WsMessage:
    """Coerce a bare ``str`` / ``bytes`` into the matching frame, or pass a
    :class:`WsMessage` through unchanged."""
    return as_ws_message(message)


def frame_of(payload: Union[str, bytes]) -> WsMessage:
    """Wrap a raw inbound frame into the matching :class:`WsMessage`."""
    return ws_message_for_payload(payload)


def build_pool(impl: str, pool: Optional[Pool[WsTransport]]) -> Pool[WsTransport]:
    """The :class:`Pool` to lease from -- the caller's, or a default for ``impl``.

    A default pool is keyed by the full URL (one socket per endpoint) and hands out
    :class:`WsConnection` leases. ``websockets`` builds
    :class:`~simplyprint_ws_client.contrib.connection.websockets.Websockets`; ``aiohttp``
    builds :class:`~simplyprint_ws_client.contrib.connection.aiohttp.Aiohttp`.
    """
    if pool is not None:
        return pool

    existing = DEFAULT_POOLS.get(impl)
    if existing is not None:
        return existing

    def make_transport(url: yarl.URL, params: object) -> WsTransport:
        retry = params if isinstance(params, RetryPolicy) else RetryPolicy()
        if impl == "websockets":
            return Websockets(url, retry)
        if impl == "aiohttp":
            return Aiohttp(url, retry)
        raise ValueError(
            f"ws.connect: unknown impl {impl!r} (use 'websockets'/'aiohttp')"
        )

    built: Pool[WsTransport] = Pool(
        build=make_transport,
        key=lambda url, params: str(url),
        lease_class=WsConnection,
    )
    DEFAULT_POOLS[impl] = built
    return built


#: One default pool per ``impl``, created on first use and torn down by
#: :func:`shutdown`. A caller that passes its own ``pool`` never touches these.
DEFAULT_POOLS: Dict[str, Pool[WsTransport]] = {}


def connect(
    url: Union[str, yarl.URL],
    *,
    impl: str = "websockets",
    retry: Optional[RetryPolicy] = None,
    pool: Optional[Pool[WsTransport]] = None,
) -> WsConnection:
    """Lease a self-healing WebSocket to ``url`` and return the connection handle.

    Synchronous and fire-and-forget: it refcounts a shared socket and returns at
    once; readiness is awaited via :meth:`~simplyprint_ws_client.contrib.connection.connection.Connection.ready`.
    ``impl`` selects the wire library (``"websockets"`` default, or ``"aiohttp"``);
    ``retry`` overrides the backoff/give-up policy for a *new* endpoint socket. Pass
    ``pool`` to lease from your own pool instead of the per-``impl`` default. Raises
    on a non-``ws``/``wss`` URL or an unknown ``impl``.
    """
    url = yarl.URL(url) if isinstance(url, str) else url
    if url.scheme not in ("ws", "wss"):
        raise ValueError(f"expected a ws:// or wss:// URL, got {url.scheme!r}")
    if impl not in SUPPORTED_IMPLS:
        raise ValueError(
            f"unknown websocket impl {impl!r}; choose one of {SUPPORTED_IMPLS}"
        )

    pool = build_pool(impl, pool)
    lease = pool.connect(url, retry or RetryPolicy())
    assert isinstance(lease, WsConnection)
    return lease


def shutdown() -> None:
    """Tear down every default pool. Idempotent.

    Stops each default pool's fan-out and drops its bookkeeping; the live sockets'
    async ``stop`` is the owning loop's to drive (a lease ``close`` already does this
    on the last release). A caller that passed its own ``pool`` manages it itself.
    """
    pools = list(DEFAULT_POOLS.values())
    DEFAULT_POOLS.clear()
    for pool in pools:
        pool.stop()
