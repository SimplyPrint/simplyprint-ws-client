"""Process-wide default pools, shared by the connect front doors.

Each front door keeps one :class:`DefaultPools` registry: one
:class:`~simplyprint_ws_client.wire.pool.Pool` per
``(impl, loop, wire-keepalive)`` identity, created on first use and torn down by
the front door's ``shutdown()``. A caller that passes its own ``pool`` never
touches these.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, Generic, Hashable, Optional, Set, TypeVar

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire.options import WireKeepalive
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.transport import Transport

T = TypeVar("T", bound=Transport)

#: Keeps scheduled transport-stop tasks alive until they finish (asyncio holds
#: tasks weakly; an unreferenced stop task could be collected mid-teardown).
_STOP_TASKS: Set["asyncio.Task"] = set()


def pool_identity(
    impl: str,
    provider: Optional[EventLoopProvider],
    wire_keepalive: Optional[WireKeepalive],
) -> Hashable:
    """The hashable identity a default pool is shared by."""
    if provider is None and wire_keepalive is None:
        return impl
    provider_key: object = None
    if provider is not None:
        try:
            provider_key = id(provider.event_loop)
        except RuntimeError:
            provider_key = id(provider)
    return impl, provider_key, wire_keepalive


class DefaultPools(Generic[T]):
    """One front door's default-pool cache, keyed by :func:`pool_identity`."""

    def __init__(self) -> None:
        self.pools: Dict[Hashable, Pool[T]] = {}

    def get(
        self,
        impl: str,
        provider: Optional[EventLoopProvider],
        wire_keepalive: Optional[WireKeepalive],
        build: Callable[[], Pool[T]],
    ) -> Pool[T]:
        """The pool for this identity, building (and caching) it on first use."""
        key = pool_identity(impl, provider, wire_keepalive)
        existing = self.pools.get(key)
        if existing is not None:
            return existing
        built = build()
        self.pools[key] = built
        return built

    def shutdown(self) -> None:
        """Tear down every cached pool and stop its live transports. Idempotent.

        A lease released after this finds no endpoint to stop the socket
        through, so each transport's async ``stop`` is scheduled here, on the
        transport's own loop (awaitable from that loop, threadsafe from any
        other).
        """
        pools = list(self.pools.values())
        self.pools.clear()
        for pool in pools:
            for transport in pool.stop():
                _schedule_transport_stop(transport)


def _schedule_transport_stop(transport: Transport) -> None:
    """Run ``transport.stop()`` on the transport's loop from any thread."""
    coro = transport.stop()

    try:
        loop = transport.provider.event_loop
    except (AttributeError, RuntimeError):
        loop = None

    if loop is None or loop.is_closed():
        coro.close()
        return

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is loop:
        task = loop.create_task(coro)
        _STOP_TASKS.add(task)
        task.add_done_callback(_STOP_TASKS.discard)
    else:
        asyncio.run_coroutine_threadsafe(coro, loop)
