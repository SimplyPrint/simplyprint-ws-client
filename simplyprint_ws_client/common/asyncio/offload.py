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

One :class:`Offload` is owned by the app for its lifetime; :meth:`shutdown` runs
once on teardown.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

from simplyprint_ws_client.common.asyncio.concurrent import await_concurrent_future

__all__ = ["Offload", "install_default_executor"]

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
        self._transfer = ThreadPoolExecutor(
            max_workers=transfer_workers, thread_name_prefix="sp-transfer"
        )
        self._closed = False

    async def run_io(self, fn: Callable[..., T], *args: Any) -> T:
        """Run a short blocking syscall on the ``io`` lane; await its result."""
        return await self._run(self._io, fn, *args)

    async def run_transfer(self, fn: Callable[..., T], *args: Any) -> T:
        """Run long-haul blocking work on the ``transfer`` lane; await its result."""
        return await self._run(self._transfer, fn, *args)

    @staticmethod
    async def _run(executor: ThreadPoolExecutor, fn: Callable[..., T], *args: Any) -> T:
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
