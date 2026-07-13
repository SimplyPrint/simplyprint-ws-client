"""Fixed-task dispatch for sheddable background ingress.

This is intentionally narrower than an event bus or task supervisor: callers
submit already-parsed work items, a fixed number of owner tasks invoke one async
handler, and excess queued items are rejected. It is suitable for UDP discovery
traffic where dropping overload is safer than creating an unbounded task per
datagram. Lossless protocol/control paths must choose their own backpressure.
"""

from __future__ import annotations

import asyncio
from typing import (
    Awaitable,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    TypeVar,
    cast,
)

try:
    from asyncio import timeout
except ImportError:  # pragma: no cover -- Python 3.9/3.10 compatibility
    from async_timeout import timeout

T = TypeVar("T")
R = TypeVar("R")


async def map_concurrently(
    handler: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    concurrency: int,
) -> List[R]:
    """Map in input order while creating at most ``concurrency`` tasks.

    A semaphore around ``gather(*(handler(item) ...))`` limits active work but
    still allocates one task per item. This worker-iterator form bounds both.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    indexed = iter(enumerate(items))
    results: List[Optional[R]] = []

    async def worker() -> None:
        while True:
            try:
                index, item = next(indexed)
            except StopIteration:
                return
            if index == len(results):
                results.append(None)
            awaitable = handler(item)
            results[index] = await awaitable

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return cast(List[R], results)


class BoundedAsyncDispatcher(Generic[T]):
    """Dispatch items through at most ``workers`` long-lived tasks."""

    def __init__(
        self,
        handler: Callable[[T], Awaitable[None]],
        *,
        workers: int,
        maxsize: int,
        idle_timeout: Optional[float] = None,
        on_overflow: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        if idle_timeout is not None and idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")

        self._handler = handler
        self._worker_limit = workers
        self._idle_timeout = idle_timeout
        self._on_overflow = on_overflow
        self._on_error = on_error
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._workers: Set[asyncio.Task] = set()
        self._closed = False
        self._overflowed = False

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def active_workers(self) -> int:
        return sum(not task.done() for task in self._workers)

    def open(self) -> None:
        self._closed = False

    def submit(self, item: T) -> bool:
        """Queue an item without blocking; return ``False`` on overload/close."""
        if self._closed:
            return False
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            if not self._overflowed:
                self._overflowed = True
                if self._on_overflow is not None:
                    self._on_overflow()
            return False
        self._ensure_workers()
        return True

    def close(self) -> None:
        """Cancel owners and discard queued, not-yet-started work."""
        if self._closed:
            return
        self._closed = True
        for task in tuple(self._workers):
            task.cancel()
        self._workers.clear()
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()
        self._overflowed = False

    async def join(self) -> None:
        """Wait until every accepted item has completed or been discarded."""
        await self._queue.join()

    def _ensure_workers(self) -> None:
        if self._closed:
            return
        self._workers.difference_update({task for task in self._workers if task.done()})
        while len(self._workers) < self._worker_limit:
            task = asyncio.create_task(self._worker())
            self._workers.add(task)
            task.add_done_callback(self._worker_done)

    def _worker_done(self, task: asyncio.Task) -> None:
        self._workers.discard(task)
        if not self._closed and not self._queue.empty():
            self._ensure_workers()

    async def _worker(self) -> None:
        while True:
            try:
                if self._idle_timeout is None:
                    item = await self._queue.get()
                else:
                    # The timeout context schedules cancellation on this owner
                    # task. ``asyncio.wait_for(queue.get())`` would create a
                    # short-lived helper task for every item.
                    async with timeout(self._idle_timeout):
                        item = await self._queue.get()
            except asyncio.TimeoutError:
                return

            try:
                await self._handler(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one bad item must not kill its owner
                if self._on_error is not None:
                    self._on_error(exc)
            finally:
                self._queue.task_done()
                if self._overflowed and self._queue.qsize() <= (
                    self._queue.maxsize // 2
                ):
                    self._overflowed = False
