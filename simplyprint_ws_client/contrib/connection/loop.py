"""The cross-thread coroutine bridge: the one sanctioned hop onto the pool loop.

A sync transport's wire thread (paho, websocket-client) sometimes needs to run a
coroutine on the pool's event loop -- an auth re-handshake, say. :class:`LoopBridge`
is the single, owned way to do that: a pool holds one (bound to its loop via an
:class:`EventLoopProvider`) and hands it to every lease, replacing every ad-hoc
``run_coroutine_threadsafe`` / ``getattr(client, "submit_to_loop")`` reach-in.
``coalesce_key`` collapses a duplicate submission while one is in flight to a no-op.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Callable, Coroutine, Hashable, Optional, Set

from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

__all__ = ["LoopBridge", "CoroFactory"]

#: A thunk returning a fresh coroutine to run on the pool loop (built per call so
#: a coalesced/aborted submission never leaves an un-awaited coroutine behind).
CoroFactory = Callable[[], Coroutine[Any, Any, Any]]


class LoopBridge:
    """The cross-thread coroutine bridge behind ``Lease.submit_to_loop``.

    A pool owns one (bound to its event loop via an :class:`EventLoopProvider`)
    and hands it to every lease. This is the single, formal replacement for every
    ad-hoc ``getattr(client, "submit_to_loop")`` reach-in: the loop is owned here,
    not discovered from a client.
    """

    def __init__(
        self,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._provider = provider or EventLoopProvider.default()
        self._logger = logger or logging.getLogger("connection.loop")
        self._inflight: Set[Hashable] = set()
        self._lock = threading.Lock()

    def submit(
        self, coro_factory: CoroFactory, *, coalesce_key: Optional[Hashable] = None
    ) -> None:
        if coalesce_key is not None:
            with self._lock:
                if coalesce_key in self._inflight:
                    return
                self._inflight.add(coalesce_key)

        try:
            loop = self._provider.event_loop
        except RuntimeError:
            self._logger.warning("submit_to_loop: no loop available")
            self._discard(coalesce_key)
            return

        coro = coro_factory()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            coro.close()  # loop closed; don't leak an un-awaited coroutine
            self._discard(coalesce_key)
            return

        future.add_done_callback(lambda f: self._on_done(f, coalesce_key))

    def call(self, fn: Callable[[], None]) -> None:
        """Run synchronous work on the pool's event loop.

        If the caller is already on the target loop the function runs inline;
        otherwise it is scheduled with ``call_soon_threadsafe``. This is the sync
        companion to :meth:`submit` for event-bus delivery from scheduler/worker
        threads.
        """
        try:
            loop = self._provider.event_loop
        except RuntimeError:
            self._logger.warning("call_on_loop: no loop available")
            return

        try:
            if asyncio.get_running_loop() is loop:
                self._safe_call(fn)
                return
        except RuntimeError:
            pass

        try:
            loop.call_soon_threadsafe(self._safe_call, fn)
        except RuntimeError:
            self._logger.warning("call_on_loop: loop is closed")

    def _safe_call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            self._logger.exception("call_on_loop work failed")

    def _on_done(
        self, future: "concurrent.futures.Future", key: Optional[Hashable]
    ) -> None:
        self._discard(key)
        try:
            exc = future.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            self._logger.warning("submit_to_loop work failed: %r", exc)

    def _discard(self, key: Optional[Hashable]) -> None:
        if key is None:
            return
        with self._lock:
            self._inflight.discard(key)
