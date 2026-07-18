"""An owner-scoped registry of transport pools.

Applications retain one registry per wire protocol and inject those registries
through :class:`~simplyprint_ws_client.core.client_context.ClientContext`.
Nothing in the wire front doors owns process state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Callable, Generic, Hashable, Optional, TypeVar

from simplyprint_ws_client.common.asyncio.concurrent import await_concurrent_future
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire.options import WireKeepalive
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.transport import Transport

T = TypeVar("T", bound=Transport)


def pool_identity(
    provider: Optional[EventLoopProvider],
    wire_keepalive: Optional[WireKeepalive],
) -> Hashable:
    """The hashable identity an owner shares a pool by."""
    provider_key: object = None
    if provider is not None:
        try:
            provider_key = id(provider.event_loop)
        except RuntimeError:
            provider_key = id(provider)
    return provider_key, wire_keepalive


class PoolRegistry(Generic[T]):
    """One owner's pool cache, keyed by :func:`pool_identity`."""

    def __init__(self) -> None:
        self._pools: dict[Hashable, Pool[T]] = {}
        self._lock = threading.Lock()
        # The first close installs this permanent lifecycle barrier; concurrent
        # and later closes join it. ``concurrent.futures.Future`` is deliberately
        # loop-neutral because callers can live on different application threads.
        self._close_future: Optional[concurrent.futures.Future[None]] = None

    def get(
        self,
        provider: Optional[EventLoopProvider],
        wire_keepalive: Optional[WireKeepalive],
        build: Callable[[], Pool[T]],
    ) -> Pool[T]:
        """The pool for this identity, building (and caching) it on first use."""
        key = pool_identity(provider, wire_keepalive)
        with self._lock:
            if self._close_future is not None:
                raise RuntimeError("cannot acquire from a closed pool registry")
            existing = self._pools.get(key)
            if existing is not None:
                return existing
            # Pool construction is synchronous bookkeeping only. Keeping it in
            # the critical section prevents duplicate pools for one identity.
            built = build()
            self._pools[key] = built
            return built

    async def close(self) -> None:
        """Permanently close every cached pool and await its live transports.

        A lease released after this finds no endpoint to stop the socket
        through. Each pool is detached and its transports are awaited on that
        pool's own loop, even when another owner loop initiates shutdown. New
        acquisitions are rejected once close begins; repeated closes join the
        same terminal operation.
        """
        with self._lock:
            close_future = self._close_future
            if close_future is None:
                close_future = concurrent.futures.Future()
                self._close_future = close_future
                pools = list(self._pools.values())
                self._pools.clear()
                leader = True
            else:
                pools = []
                leader = False

        if not leader:
            await await_concurrent_future(close_future)
            return

        close_task = asyncio.create_task(_close_pools(pools))
        cancellation: Optional[asyncio.CancelledError] = None
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError as error:
                # Registry teardown owns live socket/thread handles. Finish it
                # before honoring caller cancellation so clearing the cache can
                # never orphan those handles.
                cancellation = error
            except BaseException:
                # The completed task's result below records the same failure on
                # the loop-neutral close future for every joining caller.
                pass

        try:
            close_task.result()
        except BaseException as error:
            close_future.set_exception(error)
            raise
        close_future.set_result(None)
        if cancellation is not None:
            raise cancellation


async def _close_pool_on_owner_loop(pool: Pool[T]) -> None:
    """Detach ``pool`` and stop its transports on the loop that owns them."""

    owner_loop = pool.provider.event_loop
    running_loop = asyncio.get_running_loop()
    if owner_loop is running_loop or not owner_loop.is_running():
        # A stopped loop has no concurrent event dispatch. This direct path is
        # also what unit-test transports without a detached owner use.
        await _close_pool(pool)
        return

    close_coro = _close_pool(pool)
    try:
        submitted = asyncio.run_coroutine_threadsafe(close_coro, owner_loop)
    except RuntimeError:
        # The owner can finish between ``is_running`` and submission. It can no
        # longer race event dispatch, so close directly on the surviving loop.
        close_coro.close()
        await _close_pool(pool)
        return
    await await_concurrent_future(submitted)


async def _close_pools(pools: list[Pool[T]]) -> None:
    await asyncio.gather(*(_close_pool_on_owner_loop(pool) for pool in pools))


async def _close_pool(pool: Pool[T]) -> None:
    transports = pool.stop()
    await asyncio.gather(*(transport.stop() for transport in transports))
