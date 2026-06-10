"""The WebSocket front door: a 1:1 self-healing socket, pooled by endpoint.

A printer client reaches a WebSocket endpoint through one call::

    from simplyprint_ws_client.device.connection import ws
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
front-door :class:`WsLease` lease wraps an inbound frame into a
:class:`WsMessage` for handlers and reduces an outbound :class:`WsMessage` back to
``str``/``bytes`` for the wire. A per-``impl``
:class:`~simplyprint_ws_client.device.connection.pool.Pool` keyed by the full URL shares
one socket across every lease on the same endpoint.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Hashable, Optional, Union

import yarl

from simplyprint_ws_client.common.wire.aiohttp import (
    Aiohttp,
    default_aiohttp_connect,
)
from simplyprint_ws_client.device.connection.lease import WsLease
from simplyprint_ws_client.device.connection.keepalive import Keepalive
from simplyprint_ws_client.common.wire.messages import (
    WsKind,
    WsMessage,
    as_ws_message,
    ws_message_for_payload,
)
from simplyprint_ws_client.common.wire.policy import RetryPolicy
from simplyprint_ws_client.device.connection.pool import Pool
from simplyprint_ws_client.device.connection.options import (
    ConnectionOptions,
    WireKeepalive,
)
from simplyprint_ws_client.common.wire.transport import WsTransport
from simplyprint_ws_client.common.wire.websockets import Websockets
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

__all__ = [
    "WsKind",
    "WsMessage",
    "WsLease",
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


def build_pool(
    impl: str,
    pool: Optional[Pool[WsTransport]],
    provider: Optional[EventLoopProvider] = None,
    wire_keepalive: Optional[WireKeepalive] = None,
) -> Pool[WsTransport]:
    """The :class:`Pool` to lease from -- the caller's, or a default for ``impl``.

    A default pool is keyed by the full URL (one socket per endpoint) and hands out
    :class:`WsLease` leases. ``websockets`` builds
    :class:`~simplyprint_ws_client.common.wire.websockets.Websockets`; ``aiohttp``
    builds :class:`~simplyprint_ws_client.common.wire.aiohttp.Aiohttp`.
    """
    if pool is not None:
        return pool

    pool_key: Hashable = _pool_key(impl, provider, wire_keepalive)
    existing = DEFAULT_POOLS.get(pool_key)
    if existing is not None:
        return existing

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

    built: Pool[WsTransport] = Pool(
        build=make_transport,
        key=lambda url, params: str(url),
        lease_class=WsLease,
        provider=provider,
    )
    DEFAULT_POOLS[pool_key] = built
    return built


#: One default pool per ``impl``, created on first use and torn down by
#: :func:`shutdown`. A caller that passes its own ``pool`` never touches these.
DEFAULT_POOLS: Dict[Hashable, Pool[WsTransport]] = {}


def connect(
    url: Union[str, yarl.URL],
    *,
    impl: str = "websockets",
    retry: Optional[RetryPolicy] = None,
    pool: Optional[Pool[WsTransport]] = None,
    provider: Optional[EventLoopProvider] = None,
    keepalive: Optional[Keepalive] = None,
    wire_keepalive: Optional[WireKeepalive] = None,
    options: Optional[ConnectionOptions] = None,
) -> WsLease:
    """Lease a self-healing WebSocket to ``url`` and return the connection handle.

    Synchronous and fire-and-forget: it refcounts a shared socket and returns at
    once; readiness is awaited via :meth:`~simplyprint_ws_client.device.connection.lease.Lease.ready`.
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

    options = _resolve_options(options, retry, provider, keepalive, wire_keepalive)
    pool = build_pool(impl, pool, options.provider, options.wire_keepalive)
    lease = pool.connect(url, options.retry or RetryPolicy())
    assert isinstance(lease, WsLease)
    if options.app_keepalive is not None:
        lease.keepalive(options.app_keepalive)
    return lease


def _resolve_options(
    options: Optional[ConnectionOptions],
    retry: Optional[RetryPolicy],
    provider: Optional[EventLoopProvider],
    keepalive: Optional[Keepalive],
    wire_keepalive: Optional[WireKeepalive],
) -> ConnectionOptions:
    resolved = options or ConnectionOptions()
    if retry is not None:
        resolved = replace(resolved, retry=retry)
    if provider is not None:
        resolved = replace(resolved, provider=provider)
    if keepalive is not None:
        resolved = replace(resolved, app_keepalive=keepalive)
    if wire_keepalive is not None:
        resolved = replace(resolved, wire_keepalive=wire_keepalive)
    return resolved


def _pool_key(
    impl: str,
    provider: Optional[EventLoopProvider],
    wire_keepalive: Optional[WireKeepalive],
) -> Hashable:
    if provider is None and wire_keepalive is None:
        return impl
    provider_key: object = None
    if provider is not None:
        try:
            provider_key = id(provider.event_loop)
        except RuntimeError:
            provider_key = id(provider)
    return impl, provider_key, wire_keepalive


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
    """Tear down every default pool. Idempotent.

    Stops each default pool's fan-out and drops its bookkeeping; the live sockets'
    async ``stop`` is the owning loop's to drive (a lease ``close`` already does this
    on the last release). A caller that passed its own ``pool`` manages it itself.
    """
    pools = list(DEFAULT_POOLS.values())
    DEFAULT_POOLS.clear()
    for pool in pools:
        pool.stop()
