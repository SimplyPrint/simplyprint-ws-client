"""A dirty-flag, trailing-edge single-flight runner -- the one primitive that
collapses a storm of triggers into one job.

Many subsystems are poked far more often than the work needs to run: N config
fields change in one tick but the file only needs writing once; a camera URI
flips on/off/on but only the final desired state matters. :class:`CoalescingTask`
is the shared answer: :meth:`trigger` is a cheap, thread-safe "mark dirty"; a
single retained task runs ``fn`` at most once at a time and, if more triggers
land while it runs, runs exactly one more time afterward. There is never a fan of
tasks and never a lost update.

The lock discipline mirrors :class:`~...common.asyncio.courier.Courier`: a
``threading.Lock`` guards the dirty/running flags, the empty->scheduled edge hops
to the loop via ``call_soon_threadsafe``, and a trigger raised when no loop is
running simply leaves the work marked dirty for a later trigger or :meth:`aclose`.

``delay`` adds a debounce: the runner waits ``delay`` before consuming the dirty
flag, so a burst within the window collapses to a single run. Draining
(:meth:`aclose`) skips the debounce and flushes any pending dirt once.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Coroutine, Optional

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

__all__ = ["CoalescingTask"]

#: ``fn()`` -> coroutine; invoked with no arguments (it reads current state).
CoalesceFn = Callable[[], "Coroutine[Any, Any, Any]"]


class CoalescingTask(EventLoopProvider[asyncio.AbstractEventLoop]):
    """Run ``fn`` at most once at a time; collapse concurrent triggers to one
    trailing re-run. ``trigger()`` is sync and thread-safe."""

    def __init__(
        self,
        fn: CoalesceFn,
        *,
        delay: float = 0.0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        provider: Optional[EventLoopProvider] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        EventLoopProvider.__init__(self, loop=loop, provider=provider)
        self._fn = fn
        self._delay = delay
        self._logger = logger or logging.getLogger("coalescing_task")
        self._lock = threading.Lock()
        self._dirty = False
        self._running = False  # a runner task is active (or being scheduled)
        self._closed = False
        self._task: Optional[asyncio.Task] = None

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def trigger(self) -> None:
        """Mark work pending. Thread-safe; safe to call from any thread. A no-op
        after :meth:`aclose`. If no loop is running yet, the work stays dirty for
        a later trigger (it is not lost and never raises)."""
        with self._lock:
            if self._closed:
                return
            self._dirty = True
            if self._running:
                return  # the active runner will see the flag and run again
            self._running = True
        if not self._schedule_runner():
            # No live loop to schedule on; release the running claim so a later
            # trigger (with a running loop) can schedule. The dirty flag stays.
            with self._lock:
                self._running = False

    def _schedule_runner(self) -> bool:
        try:
            self.event_loop.call_soon_threadsafe(self._ensure_runner)
            return True
        except RuntimeError:
            return False

    def _ensure_runner(self) -> None:
        # Runs on the loop thread.
        if self._closed:
            return
        if self._task is not None and not self._task.done():
            return
        self._task = self.event_loop.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    if not self._dirty:
                        self._running = False
                        return
                # Debounce a burst into one run; skip while draining so
                # shutdown does not wait a full window.
                if self._delay and not self._closed:
                    await asyncio.sleep(self._delay)
                with self._lock:
                    self._dirty = False
                await self._invoke()
        finally:
            with self._lock:
                self._running = False

    async def _invoke(self) -> None:
        try:
            await self._fn()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a failed job must not kill the runner
            self._logger.exception("coalesced task failed")

    async def aclose(self) -> None:
        """Stop accepting triggers, drain any pending work once, and leave no
        task behind. Call on teardown while the loop is still alive."""
        with self._lock:
            self._closed = True
            task = self._task

        if task is not None and not task.done():
            # The runner drains remaining dirt (debounce skipped now) and exits.
            try:
                await task
            except asyncio.CancelledError:
                pass
        else:
            # No live runner (e.g. the loop never ran): flush directly if dirty.
            with self._lock:
                pending = self._dirty
                self._dirty = False
            if pending:
                await self._invoke()

        with self._lock:
            self._task = None
            self._running = False
