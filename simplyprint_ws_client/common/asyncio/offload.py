"""Two bounded executor lanes -- the only sanctioned hop for blocking work off
the event loop.

The app/scheduler loop must never make a blocking call (a synchronous socket
operation, ``fsync``, a compress, a large hash, a CPU-heavy parse). Such work is
handed to one of two named, bounded thread pools and awaited:

* ``io``       -- millisecond-scale syscalls and small CPU (fsync, replace, stat,
                  a small encode). Kept small; meant to drain fast.
* ``transfer`` -- second-to-minute file work (network file transfer, compressing
                  a print file, hashing a large file). Wider, so one slow item
                  does not starve the others.

Why two pools and never the interpreter default executor: the default
``ThreadPoolExecutor`` is also where ``loop.getaddrinfo`` resolves names. A few
multi-minute transfers parked on it would stall every hostname lookup in the
process. Long-haul work gets its own lane; short syscalls get another; neither
can exhaust the other's threads, and a stack sampled mid-stall names the lane
that is busy (``sp-io`` vs ``sp-transfer``).

The ``transfer`` lane's width can change while the app runs
(:meth:`Offload.set_transfer_workers`): an app that admits N concurrent uploads
needs about N threads, and N is a user setting. The ``io`` lane stays fixed.

One :class:`Offload` is owned by the app for its lifetime; :meth:`shutdown` runs
once on teardown.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Deque, Set, TypeVar, Union

from simplyprint_ws_client.common.asyncio.concurrent import await_concurrent_future

__all__ = ["Offload", "ResizableThreadPool", "install_default_executor"]

T = TypeVar("T")

#: Short syscalls drain fast, so a small pool keeps the thread count low.
DEFAULT_IO_WORKERS = 4
#: Long-haul work needs width so one slow item does not block the rest.
DEFAULT_TRANSFER_WORKERS = 8
#: Fallback work on any one owned event loop (``asyncio.to_thread``, DNS, and
#: legacy ``run_in_executor(None, ...)``). Explicit lanes above remain preferred.
DEFAULT_LOOP_WORKERS = 4


def install_default_executor(
    loop: asyncio.AbstractEventLoop,
    *,
    workers: int = DEFAULT_LOOP_WORKERS,
    thread_name_prefix: str = "sp-loop",
) -> ThreadPoolExecutor:
    """Give an owned loop an explicit, bounded fallback executor.

    Python otherwise creates up to ``min(32, cpu_count + 4)`` threads *per
    loop*. SimplyPrint owns several loops, so leaving that implicit multiplies
    the process-wide thread budget. The loop owns and shuts down the returned
    executor as part of its normal close lifecycle.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=thread_name_prefix,
    )
    loop.set_default_executor(executor)
    return executor


def _checked_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return workers


class ResizableThreadPool:
    """A thread pool whose width can change while work runs.

    Why not ``ThreadPoolExecutor``: its width is fixed at construction (nothing
    public changes it), and its threads never exit before shutdown, so a pool
    widened for one busy batch would keep every thread it ever started. A
    semaphore in front of a wide executor has the same problem, and would have
    to be a thread-safe gate because callers come from several event loops.

    * Threads start on demand, up to :attr:`workers`; an idle one waits.
    * Growing starts threads for queued work at once.
    * Shrinking stops nothing: a surplus thread exits as soon as it is idle,
      so running work finishes and only then is the thread given back.
    * Work starts in submission order.

    The ``submit`` / ``shutdown`` subset of ``concurrent.futures.Executor``.
    """

    def __init__(self, workers: int, *, thread_name_prefix: str) -> None:
        self._cond = threading.Condition()
        self._queue: Deque[Any] = deque()
        self._threads: Set[threading.Thread] = set()
        #: Threads running an item now; the rest are idle or just starting,
        #: and each of those will take one queued item.
        self._busy = 0
        self._workers = _checked_workers(workers)
        self._prefix = thread_name_prefix
        self._names = itertools.count()
        self._shutdown = False

    @property
    def workers(self) -> int:
        """The width: at most this many items run at once."""
        return self._workers

    @property
    def threads(self) -> int:
        """Threads alive now. Above :attr:`workers` only briefly after a shrink,
        while the surplus finish their current item."""
        with self._cond:
            return len(self._threads)

    def set_workers(self, workers: int) -> None:
        """Change the width now. Grown: queued work starts at once. Shrunk:
        running work finishes, then the surplus threads exit."""
        workers = _checked_workers(workers)
        with self._cond:
            if self._shutdown:
                return  # nothing left to size
            self._workers = workers
            self._spawn_for_queue()
            # Idle threads re-check: the surplus ones exit now.
            self._cond.notify_all()

    def submit(self, fn: Callable[..., T], *args: Any) -> "Future[T]":
        future: "Future[T]" = Future()
        item = (future, fn, args)
        with self._cond:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.append(item)
            try:
                self._spawn_for_queue()
            except BaseException:
                # No thread could start (the OS limit): don't leave work behind
                # that the caller was told failed.
                self._queue.remove(item)
                raise
            self._cond.notify()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Accept no more work. ``cancel_futures`` drops what has not started;
        otherwise the queue drains first. ``wait`` joins every thread."""
        with self._cond:
            self._shutdown = True
            if cancel_futures:
                while self._queue:
                    self._queue.popleft()[0].cancel()
            threads = list(self._threads)
            self._cond.notify_all()
        if wait:
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join()

    def _spawn_for_queue(self) -> None:
        """Start threads for queued work no free thread will take. Under the lock."""
        while (
            len(self._queue) > len(self._threads) - self._busy
            and len(self._threads) < self._workers
        ):
            # Daemon: an Offload nobody shut down must not hang interpreter
            # exit on its idle threads (ThreadPoolExecutor avoids that with a
            # private atexit hook). The app joins the lane on teardown.
            thread = threading.Thread(
                target=self._run_items,
                name=f"{self._prefix}_{next(self._names)}",
                daemon=True,
            )
            self._threads.add(thread)
            try:
                thread.start()
            except BaseException:
                self._threads.discard(thread)
                raise

    def _surplus(self) -> bool:
        return len(self._threads) > self._workers

    def _run_items(self) -> None:
        me = threading.current_thread()
        while True:
            with self._cond:
                while not (self._queue or self._surplus() or self._shutdown):
                    self._cond.wait()
                if self._surplus() or not self._queue:
                    # Shrunk, or shut down with nothing left to drain.
                    self._threads.discard(me)
                    return
                future, fn, args = self._queue.popleft()
                self._busy += 1
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args)
                    except BaseException as exc:  # noqa: BLE001 -- the caller's to see
                        future.set_exception(exc)
                    else:
                        future.set_result(result)
            finally:
                # Drop references now: an idle thread must not keep the last
                # result (a whole file's bytes, say) alive.
                future = fn = args = result = None  # type: ignore[assignment]
                with self._cond:
                    self._busy -= 1


class Offload:
    """The app's two bounded blocking-work lanes.

    ``run_io`` / ``run_transfer`` submit ``fn(*args)`` to the matching lane and
    await the result on the calling loop. Arguments are positional only (the
    executor takes no keywords); bind keywords with a closure or
    :func:`functools.partial` at the call site.
    """

    def __init__(
        self,
        *,
        io_workers: int = DEFAULT_IO_WORKERS,
        transfer_workers: int = DEFAULT_TRANSFER_WORKERS,
    ) -> None:
        self._io = ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="sp-io"
        )
        # Resizable: its width follows how many uploads the app admits at once
        # (set_transfer_workers). The io lane stays a fixed, small pool.
        self._transfer = ResizableThreadPool(
            transfer_workers, thread_name_prefix="sp-transfer"
        )
        self._closed = False

    @property
    def transfer_workers(self) -> int:
        """The ``transfer`` lane's width now."""
        return self._transfer.workers

    def set_transfer_workers(self, workers: int) -> None:
        """Widen or narrow the ``transfer`` lane, live and thread-safe.

        Widened: queued work starts at once. Narrowed: running work finishes;
        no new item starts until fewer than ``workers`` run, and the surplus
        threads exit as they go idle. A no-op after :meth:`shutdown`."""
        self._transfer.set_workers(workers)

    async def run_io(self, fn: Callable[..., T], *args: Any) -> T:
        """Run a short blocking syscall on the ``io`` lane; await its result."""
        return await self._run(self._io, fn, *args)

    async def run_transfer(self, fn: Callable[..., T], *args: Any) -> T:
        """Run long-haul blocking work on the ``transfer`` lane; await its result."""
        return await self._run(self._transfer, fn, *args)

    @staticmethod
    async def _run(
        executor: Union[ThreadPoolExecutor, ResizableThreadPool],
        fn: Callable[..., T],
        *args: Any,
    ) -> T:
        context = contextvars.copy_context()
        future = executor.submit(context.run, partial(fn, *args))
        try:
            return await await_concurrent_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def shutdown(self, wait: bool = True) -> None:
        """Stop both lanes. Idempotent. Cancels queued (not-yet-started) work and,
        when ``wait``, joins the lane threads -- so teardown leaks no executor."""
        if self._closed:
            return
        self._closed = True
        # cancel_futures drops work that has not started; running items finish
        # (bounded in practice by the socket timeouts the callers set).
        self._io.shutdown(wait=wait, cancel_futures=True)
        self._transfer.shutdown(wait=wait, cancel_futures=True)
