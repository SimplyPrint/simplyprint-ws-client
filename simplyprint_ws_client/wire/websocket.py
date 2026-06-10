"""The WebSocket front door: a 1:1 self-healing socket, pooled by endpoint.

A printer client reaches a WebSocket endpoint through one call::

    from simplyprint_ws_client.wire import ws
    conn = ws.connect(yarl.URL("wss://host/path"))
    conn.event_bus.on(MessageReceived, handler)
    await conn.ready()
    await conn.send(WsMessage.text("hello"))    # or just: await conn.send("hello")
    await conn.close()

The wire underneath is one of two async libraries -- ``websockets`` (the default)
or ``aiohttp`` -- chosen by ``impl``. Both live in their own modules and are
imported here, not re-implemented; each imports its library lazily, so importing
this module never requires either to be installed.

This module owns the framing the brand-free wires deliberately do not: the
:class:`WsMessage` family. A wire carries a frame as a bare ``str``/``bytes``; the
front-door :class:`WsLease` wraps an inbound frame into a :class:`WsMessage` for
handlers and reduces an outbound :class:`WsMessage` back to ``str``/``bytes`` for
the wire. A per-``impl``
:class:`~simplyprint_ws_client.wire.pool.Pool` keyed by the full URL
shares one socket across every lease on the same endpoint.

Everything beyond ``url``/``impl``/``pool`` is carried by one
:class:`~simplyprint_ws_client.wire.options.ConnectionOptions`.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire.aiohttp import (
    Aiohttp,
    default_aiohttp_connect,
)
from simplyprint_ws_client.wire.messages import WsKind, WsMessage
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.transport import WsTransport
from simplyprint_ws_client.wire.websockets import Websockets
from simplyprint_ws_client.wire.lease import WsLease
from simplyprint_ws_client.wire.options import (
    ConnectionOptions,
    WireKeepalive,
)
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.pools import DefaultPools

__all__ = [
    "WsKind",
    "WsImpl",
    "WsMessage",
    "WsLease",
    "connect",
    "shutdown",
]

#: The name of a shipped wire implementation. ``websockets`` is the default.
WsImpl = Literal["websockets", "aiohttp"]

#: The two shipped wire implementations, by name.
SUPPORTED_IMPLS: Tuple[WsImpl, ...] = ("websockets", "aiohttp")


def build_pool(
    impl: WsImpl,
    pool: Optional[Pool[WsTransport]],
    provider: Optional[EventLoopProvider] = None,
    wire_keepalive: Optional[WireKeepalive] = None,
) -> Pool[WsTransport]:
    """The :class:`Pool` to lease from -- the caller's, or a default for ``impl``.

    A default pool is keyed by the full URL (one socket per endpoint) and hands out
    :class:`WsLease` leases. ``websockets`` builds
    :class:`~simplyprint_ws_client.wire.websockets.Websockets`; ``aiohttp``
    builds :class:`~simplyprint_ws_client.wire.aiohttp.Aiohttp`.
    """
    if pool is not None:
        return pool

    def make_transport(url: yarl.URL, params: object) -> WsTransport:
        retry = params if isinstance(params, RetryPolicy) else RetryPolicy()
        if impl == "websockets":
            return Websockets(
                url,
                retry,
                provider,
                connect_kwargs=_websockets_keepalive_kwargs(wire_keepalive),
            )
        if impl == "aiohttp":
            return Aiohttp(
                url,
                retry,
                provider,
                connect_factory=lambda u, logger: default_aiohttp_connect(
                    u, logger, heartbeat=_aiohttp_heartbeat(wire_keepalive)
                ),
            )
        raise ValueError(
            f"ws.connect: unknown impl {impl!r} (use 'websockets'/'aiohttp')"
        )

    def make_pool() -> Pool[WsTransport]:
        return Pool(
            build=make_transport,
            key=lambda url, params: str(url),
            lease_class=WsLease,
            provider=provider,
        )

    return DEFAULT_POOLS.get(impl, provider, wire_keepalive, make_pool)


#: One default pool per ``impl``, created on first use and torn down by
#: :func:`shutdown`. A caller that passes its own ``pool`` never touches these.
DEFAULT_POOLS: DefaultPools[WsTransport] = DefaultPools()


def connect(
    url: Union[str, yarl.URL],
    *,
    impl: WsImpl = "websockets",
    pool: Optional[Pool[WsTransport]] = None,
    options: Optional[ConnectionOptions] = None,
) -> WsLease:
    """Lease a self-healing WebSocket to ``url`` and return the lease handle.

    Synchronous and fire-and-forget: it refcounts a shared socket and returns at
    once; readiness is awaited via :meth:`~simplyprint_ws_client.wire.lease.Lease.ready`.
    ``impl`` selects the wire library (``"websockets"`` default, or ``"aiohttp"``);
    everything else (retry policy, loop provider, wire/app keepalive) rides in
    ``options``. Pass ``pool`` to lease from your own pool instead of the
    per-``impl`` default. Raises on a non-``ws``/``wss`` URL or an unknown ``impl``.
    """
    url = yarl.URL(url) if isinstance(url, str) else url
    if url.scheme not in ("ws", "wss"):
        raise ValueError(f"expected a ws:// or wss:// URL, got {url.scheme!r}")
    if impl not in SUPPORTED_IMPLS:
        raise ValueError(
            f"unknown websocket impl {impl!r}; choose one of {SUPPORTED_IMPLS}"
        )

    options = options or ConnectionOptions()
    pool = build_pool(impl, pool, options.provider, options.wire_keepalive)
    lease = pool.connect(url, options.retry or RetryPolicy())
    if not isinstance(lease, WsLease):
        raise TypeError(
            f"ws.connect needs a pool handing out WsLease, got {type(lease).__name__}"
        )
    if options.app_keepalive is not None:
        lease.keepalive(options.app_keepalive)
    return lease


def _websockets_keepalive_kwargs(
    wire_keepalive: Optional[WireKeepalive],
) -> dict:
    if wire_keepalive is None:
        return {}
    kwargs = {}
    if wire_keepalive.interval is not None:
        kwargs["ping_interval"] = wire_keepalive.interval
    if wire_keepalive.timeout is not None:
        kwargs["ping_timeout"] = wire_keepalive.timeout
    return kwargs


def _aiohttp_heartbeat(wire_keepalive: Optional[WireKeepalive]) -> Optional[float]:
    if wire_keepalive is None:
        return None
    return wire_keepalive.interval


def shutdown() -> None:
    """Tear down every default pool of this front door. Idempotent."""
    DEFAULT_POOLS.shutdown()
