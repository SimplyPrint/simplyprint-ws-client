"""The WebSocket front door: a 1:1 self-healing socket, pooled by endpoint.

A printer client reaches a WebSocket endpoint through one call::

    from simplyprint_ws_client.wire import ws
    conn = ws.connect(yarl.URL("wss://host/path"), pool=pool)
    conn.event_bus.on(MessageReceived, handler)
    await conn.ready()
    await conn.send(WsMessage.text("hello"))    # or just: await conn.send("hello")
    await conn.close()

The wire underneath is the async ``websockets`` library, implemented by
:class:`~simplyprint_ws_client.wire.websockets.Websockets`. It imports the wire
library lazily, so importing this module does not require it to be installed.

This module owns the framing the brand-free wires deliberately do not: the
:class:`WsMessage` family. A wire carries a frame as a bare ``str``/``bytes``; the
front-door :class:`WsLease` wraps an inbound frame into a :class:`WsMessage` for
handlers and reduces an outbound :class:`WsMessage` back to ``str``/``bytes`` for
the wire. A :class:`~simplyprint_ws_client.wire.pool.Pool` keyed by the full URL
shares one socket across every lease on the same endpoint.

Everything beyond ``url``/``pool`` is carried by one
:class:`~simplyprint_ws_client.wire.options.ConnectionOptions`.
"""

from __future__ import annotations

import logging

from typing import NamedTuple, Optional, Union

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
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
from simplyprint_ws_client.wire.pools import PoolRegistry

__all__ = [
    "WsKind",
    "WsMessage",
    "WsLease",
    "connect",
    "pool_for",
]


class WsConnectParams(NamedTuple):
    """What one ``connect`` call carries into the pool.

    The pool shares sockets by URL alone; ``retry`` and ``logger`` configure
    the transport the *first* lease on an endpoint builds (later leases share
    that socket, so per-lease values cannot apply).
    """

    retry: RetryPolicy
    logger: Optional["logging.Logger"] = None
    #: Bound on one connect attempt (``None`` = the transport's own default).
    open_timeout: Optional[float] = None


def pool_for(
    registry: PoolRegistry[WsTransport],
    provider: Optional[EventLoopProvider] = None,
    wire_keepalive: Optional[WireKeepalive] = None,
) -> Pool[WsTransport]:
    """Return this owner's Websockets pool for a loop/keepalive identity.

    The pool is keyed by the full URL (one socket per endpoint) and hands out
    :class:`WsLease` leases backed by
    :class:`~simplyprint_ws_client.wire.websockets.Websockets`.
    """

    def make_transport(url: yarl.URL, params: object) -> WsTransport:
        if isinstance(params, WsConnectParams):
            retry, logger = params.retry, params.logger
            open_timeout = params.open_timeout
        elif isinstance(params, RetryPolicy):
            retry, logger, open_timeout = params, None, None
        else:
            retry, logger, open_timeout = RetryPolicy(), None, None
        # ``None`` means "the transport's own default", not "unbounded".
        timeout_kwargs = {} if open_timeout is None else {"open_timeout": open_timeout}
        return Websockets(
            url,
            retry,
            provider,
            connect_kwargs=_websockets_keepalive_kwargs(wire_keepalive),
            logger=logger,
            **timeout_kwargs,
        )

    def make_pool() -> Pool[WsTransport]:
        return Pool(
            build=make_transport,
            key=lambda url, params: str(url),
            lease_class=WsLease,
            provider=provider,
        )

    return registry.get(provider, wire_keepalive, make_pool)


def connect(
    url: Union[str, yarl.URL],
    *,
    pool: Pool[WsTransport],
    options: Optional[ConnectionOptions] = None,
) -> WsLease:
    """Lease a self-healing WebSocket to ``url`` and return the lease handle.

    Synchronous and fire-and-forget: it refcounts a shared socket and returns at
    once; readiness is awaited via :meth:`~simplyprint_ws_client.wire.lease.Lease.ready`.
    Retry policy, loop provider, and wire/app keepalive ride in ``options``.
    The pool is mandatory and owned by the caller. Raises on a non-``ws``/``wss``
    URL.
    """
    url = yarl.URL(url) if isinstance(url, str) else url
    if url.scheme not in ("ws", "wss"):
        raise ValueError(f"expected a ws:// or wss:// URL, got {url.scheme!r}")
    options = options or ConnectionOptions()
    lease = pool.connect(
        url,
        WsConnectParams(
            options.retry or RetryPolicy(), options.logger, options.open_timeout
        ),
    )
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
