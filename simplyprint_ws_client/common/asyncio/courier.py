"""The homecoming primitive: deliver items produced in one "world" onto a target
asyncio loop -- safely, in order, and with low latency.

Three worlds exist at runtime: the main thread, the one asyncio loop everything
async runs on (the "SimplyPrint loop"), and assorted producer threads / worker
subprocesses. Work that runs off the loop -- a paho network thread, a camera
reader thread -- must hand its results *back* onto the loop before any async
consumer can touch them. That hop is what this module owns, once, so no transport
or pool writes it again (badly).

Why not ``run_coroutine_threadsafe``? Because it allocates a
``concurrent.futures.Future`` per call and wakes the loop per call. For
fire-and-forget event delivery you need neither. :class:`Courier` instead:

* appends to a :class:`collections.deque` under a short lock, and
* schedules **one** ``loop.call_soon_threadsafe`` only on the empty->non-empty
  edge (wakeup *coalescing*): a burst of N posts collapses to one loop wakeup
  that drains the whole batch. Latency stays ~one loop turn; throughput scales.

The terminal action -- the **sink** -- is a parameter, because consumers differ:
a pure-sync consumer (MQTT message fan-out) takes a sync sink dispatched inline
on the loop; the SimplyPrint backend takes an *async* sink (``EventBus.emit``)
awaited in a single serialized drain task so ordering across events is exact and
async listeners are never silently dropped.

Backpressure is a configurable :class:`OverflowPolicy`. The hot-path default is
latest-wins (drop the oldest queued item). A caller may mark individual items as
lossless; bounded overflow then applies only to ordinary items, so lifecycle events
can share one courier with sheddable telemetry without being evicted by it.
Dropped items are handed to an optional ``on_drop`` callback -- that is how the
camera layer recycles a shared-memory slab whose frame was superseded before the
loop consumed it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Generic,
    Optional,
    TypeVar,
)

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

__all__ = ["OverflowPolicy", "Courier"]

T = TypeVar("T")

#: A sink that runs on the loop and returns nothing (``is_async_sink=False``).
SyncSink = Callable[[T], None]
#: A sink that runs on the loop and is awaited (``is_async_sink=True``).
AsyncSink = Callable[[T], Coroutine[Any, Any, None]]

_NOTHING = object()


class OverflowPolicy(Enum):
    """What a bounded :class:`Courier` does when its buffer is full.

    The same enum is reused by the worker layer for the producer-side slab
    overflow decision, so "latest wins" means the same thing everywhere.
    """

    DROP_OLDEST = "drop_oldest"
    """Default hot-path: evict the oldest queued item to make room for the new
    one (a ring buffer). Telemetry and frames are stale the instant a newer one
    exists, so the newest is the one worth keeping."""

    DROP_NEWEST = "drop_newest"
    """Keep the existing backlog; reject the incoming item. Rare; for when the
    head of the queue matters more than the tail."""

    BLOCK = "block"
    """Bounded: the producer blocks until the loop drains space. Legal only from
    a producer *thread* -- blocking from the target loop thread would deadlock
    and is rejected at runtime."""

    UNBOUNDED = "unbounded"
    """Never drop, never block. ``maxsize`` is ignored. Required for delivery of
    lifecycle events (connect/disconnect) whose loss would desynchronize a state
    machine; memory is the only limit."""


class Courier(Generic[T]):
    """Coalesced, low-latency delivery of items onto a target event loop.

    Construct it on (or bound to) the loop you want items delivered on, then call
    :meth:`post` from any thread. See the module docstring for the design.
    """

    def __init__(
        self,
        *,
        sink: Callable[[T], Any],
        is_async_sink: bool = False,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        maxsize: int = 1024,
        lossless: Optional[Callable[[T], bool]] = None,
        on_drop: Optional[Callable[[T], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if policy is not OverflowPolicy.UNBOUNDED and maxsize <= 0:
            raise ValueError("A bounded OverflowPolicy requires maxsize > 0")

        self._provider = self._resolve_provider(provider, loop)
        self._sink = sink
        self._is_async_sink = is_async_sink
        self._policy = policy
        self._maxsize = 0 if policy is OverflowPolicy.UNBOUNDED else maxsize
        self._lossless = lossless
        self._on_drop = on_drop
        self._logger = logger or logging.getLogger("courier")

        self._queue: Deque[T] = deque()
        self._bounded = 0
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._scheduled = False
        self._closed = False
        self._dropped = 0
        # Touched only on the loop thread (async sink path).
        self._drain_task: Optional[asyncio.Task] = None

    @staticmethod
    def _resolve_provider(
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]],
        loop: Optional[asyncio.AbstractEventLoop],
    ) -> EventLoopProvider[asyncio.AbstractEventLoop]:
        if provider is not None:
            return provider
        if loop is not None:
            return EventLoopProvider(loop=loop)
        # Prefer capturing a concrete loop object now (the common case: the
        # courier is built during async setup). A bare get_running_loop factory
        # would only resolve on the loop thread, breaking cross-thread posts.
        try:
            return EventLoopProvider(loop=asyncio.get_running_loop())
        except RuntimeError:
            return EventLoopProvider.default()

    def post(self, item: T) -> bool:
        """Enqueue ``item`` for delivery on the loop. Non-blocking (unless the
        policy is ``BLOCK``). Returns ``False`` if the item was dropped/rejected
        or the courier is closed."""
        dropped = _NOTHING
        rejected = False
        schedule = False

        with self._lock:
            if self._closed:
                return False

            lossless = self._is_lossless(item)
            if not lossless and self._maxsize and self._bounded >= self._maxsize:
                if self._policy is OverflowPolicy.DROP_NEWEST:
                    rejected = True
                    dropped = item
                elif self._policy is OverflowPolicy.BLOCK:
                    self._raise_if_on_loop()
                    while (
                        self._maxsize
                        and self._bounded >= self._maxsize
                        and not self._closed
                    ):
                        self._not_full.wait()
                    if self._closed:
                        return False
                else:  # DROP_OLDEST
                    dropped = self._drop_oldest_bounded()
                    if dropped is _NOTHING:
                        rejected = True
                        dropped = item

            if rejected:
                self._dropped += 1
            else:
                if dropped is not _NOTHING:
                    self._dropped += 1
                self._queue.append(item)
                if not lossless:
                    self._bounded += 1
                if not self._scheduled:
                    self._scheduled = True
                    schedule = True

        if dropped is not _NOTHING:
            self._safe_on_drop(dropped)  # type: ignore[arg-type]
        if schedule:
            self._schedule_drain()
        return not rejected

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def dropped(self) -> int:
        """Count of items dropped/rejected by the overflow policy."""
        return self._dropped

    def _schedule_drain(self) -> None:
        callback = self._spawn_async_drain if self._is_async_sink else self._drain_sync
        try:
            loop = self._provider.event_loop
            if self._on_target_loop(loop):
                loop.call_soon(callback)
            else:
                loop.call_soon_threadsafe(callback)
        except RuntimeError:
            # No loop yet, or the loop is closing. Release the flag so the next
            # post retries; the queued items stay put until then.
            with self._lock:
                self._scheduled = False

    def _drain_sync(self) -> None:
        """Deliver one batch inline on the loop (sync sink). One batch per
        scheduled callback keeps the loop fair under a flood; a producer that
        appends after the flag is cleared schedules a fresh drain."""
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
            self._bounded = 0
            self._not_full.notify_all()
            # Cleared AFTER taking the batch, under the lock: any item appended
            # from here on sees _scheduled False and reschedules -> no lost wakeup.
            self._scheduled = False
        for item in batch:
            self._safe_invoke_sync(item)

    def _spawn_async_drain(self) -> None:
        """Ensure a single serialized drain task is running (async sink)."""
        if self._drain_task is not None and not self._drain_task.done():
            return
        try:
            self._drain_task = self._provider.event_loop.create_task(
                self._async_drain_loop()
            )
        except RuntimeError:
            with self._lock:
                self._scheduled = False

    async def _async_drain_loop(self) -> None:
        """Await every queued item in FIFO order in ONE task, so ordering across
        events is exact (one-task-per-batch could interleave and reorder)."""
        while True:
            with self._lock:
                batch = list(self._queue)
                self._queue.clear()
                self._bounded = 0
                self._not_full.notify_all()
                if not batch:
                    # Keep _scheduled and _drain_task True/alive until we observe
                    # an empty queue under the lock, then release both together.
                    self._scheduled = False
                    self._drain_task = None
                    return
            for item in batch:
                await self._safe_invoke_async(item)

    def _safe_invoke_sync(self, item: T) -> None:
        try:
            self._sink(item)
        except Exception:  # noqa: BLE001 -- a bad listener must not kill delivery
            self._logger.exception("courier sink failed")

    async def _safe_invoke_async(self, item: T) -> None:
        try:
            await self._sink(item)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self._logger.exception("courier async sink failed")

    def _safe_on_drop(self, item: T) -> None:
        if self._on_drop is None:
            return
        try:
            self._on_drop(item)
        except Exception:  # noqa: BLE001
            self._logger.exception("courier on_drop failed")

    def _raise_if_on_loop(self) -> None:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            target = self._provider.event_loop
        except RuntimeError:
            return
        if running is target:
            raise RuntimeError(
                "OverflowPolicy.BLOCK cannot block on the target loop thread"
            )

    def close(self, *, drain: bool = True) -> None:
        """Stop accepting posts. With ``drain`` (default) deliver whatever is
        queued; otherwise hand the queued items to ``on_drop``.

        Shutdown order matters: stop the *producer* first (so no new post races
        this), then close the courier.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._not_full.notify_all()  # release any BLOCK-ed producer
            pending = list(self._queue)
            self._queue.clear()
            self._bounded = 0
            self._scheduled = False

        if not drain:
            self._cancel_drain_task()

        if not pending:
            return

        if not drain:
            for item in pending:
                self._safe_on_drop(item)
            return

        if not self._provider.event_loop_is_running():
            # Nothing will run our callback; recycle instead of leaking.
            for item in pending:
                self._safe_on_drop(item)
            return

        try:
            loop = self._provider.event_loop
            if self._on_target_loop(loop):
                loop.call_soon(self._final_drain, pending)
            else:
                loop.call_soon_threadsafe(self._final_drain, pending)
        except RuntimeError:
            for item in pending:
                self._safe_on_drop(item)

    def _final_drain(self, pending: "list[T]") -> None:
        if self._is_async_sink:
            self._provider.event_loop.create_task(self._final_drain_async(pending))
            return
        for item in pending:
            self._safe_invoke_sync(item)

    async def _final_drain_async(self, pending: "list[T]") -> None:
        for item in pending:
            await self._safe_invoke_async(item)

    def _is_lossless(self, item: T) -> bool:
        return self._lossless is not None and self._lossless(item)

    @staticmethod
    def _on_target_loop(loop: asyncio.AbstractEventLoop) -> bool:
        try:
            return asyncio.get_running_loop() is loop
        except RuntimeError:
            return False

    def _drop_oldest_bounded(self) -> object:
        for index, item in enumerate(self._queue):
            if not self._is_lossless(item):
                del self._queue[index]
                self._bounded -= 1
                return item
        return _NOTHING

    def _cancel_drain_task(self) -> None:
        task = self._drain_task
        self._drain_task = None
        if task is None or task.done():
            return
        try:
            if asyncio.current_task(loop=task.get_loop()) is task:
                return
        except RuntimeError:
            pass
        try:
            loop = task.get_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
                return
        except RuntimeError:
            pass
        task.cancel()
