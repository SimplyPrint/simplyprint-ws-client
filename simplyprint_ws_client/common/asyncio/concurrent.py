"""Await executor work without a cross-thread asyncio wake-up."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
from functools import partial
from typing import Any, Callable, TypeVar

__all__ = ["await_concurrent_future", "run_in_thread"]

T = TypeVar("T")


async def await_concurrent_future(
    future: concurrent.futures.Future[T], *, poll_interval: float = 0.01
) -> T:
    """Resolve ``future`` with a timer backstop for a lost selector wake-up.

    Some supported runtimes can enqueue a ``wrap_future`` completion without
    waking a selector running on another thread. The normal bridge remains the
    zero-latency path; a shielded loop timer merely wakes the selector if that
    notification is lost. Cancellation policy belongs to the caller because a
    future may represent shared work.
    """
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    wrapped = asyncio.wrap_future(future)
    while True:
        try:
            return await asyncio.wait_for(
                asyncio.shield(wrapped), timeout=poll_interval
            )
        except asyncio.TimeoutError:
            # ``TimeoutError`` may be the work's actual exception. Distinguish
            # it from the watchdog deadline before retrying.
            if wrapped.done():
                return wrapped.result()


async def run_in_thread(
    fn: Callable[..., T],
    *args: Any,
    thread_name: str = "sp-blocking",
    **kwargs: Any,
) -> T:
    """Run one bounded blocking call without owning the event loop's executor.

    This is the standalone counterpart to an injected :class:`Offload` lane.
    A private one-worker executor avoids both an ownerless global pool and the
    default-executor shutdown wake that short-lived ``asyncio.Runner`` instances
    can lose on supported runtimes.
    """
    context = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=thread_name,
    ) as executor:
        future = executor.submit(context.run, partial(fn, *args, **kwargs))
        try:
            return await await_concurrent_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise
