"""
Boilerplate async utils
"""

__all__ = [
    "cond_notify",
    "cond_notify_all",
    "cond_wait",
    "submit_coro_threadsafe",
]

import asyncio
from typing import Callable, Coroutine, Optional, Tuple


async def cond_notify_all(cond: asyncio.Condition):
    async with cond:
        cond.notify_all()


async def cond_notify(cond: asyncio.Condition):
    async with cond:
        cond.notify()


async def cond_wait(cond: asyncio.Condition):
    async with cond:
        await cond.wait()


def _create_task_on_loop(
    loop: asyncio.AbstractEventLoop, coro: "Coroutine[object, object, object]"
) -> Optional[asyncio.Task]:
    return loop.create_task(coro)


def submit_coro_threadsafe(
    loop: asyncio.AbstractEventLoop,
    coro: "Coroutine[object, object, object]",
    *,
    create_task: Callable[
        [asyncio.AbstractEventLoop, "Coroutine[object, object, object]"],
        Optional[asyncio.Task],
    ] = _create_task_on_loop,
) -> Tuple[bool, Optional[asyncio.Task]]:
    """Submit ``coro`` to ``loop`` from any thread.

    Returns ``(accepted, task)`` -- the task is only available when already
    running on ``loop``; a cross-thread submission creates it on the loop
    later. A rejected coroutine (loop closed or not running) is closed so it
    never leaks a 'never awaited' warning.

    ``create_task`` is the on-loop creation seam (always invoked on ``loop``):
    a caller that retains or instruments its tasks supplies its own callback;
    the default just calls ``loop.create_task``.
    """
    if loop.is_closed() or not loop.is_running():
        coro.close()
        return False, None

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is loop:
        return True, create_task(loop, coro)

    try:
        loop.call_soon_threadsafe(create_task, loop, coro)
    except RuntimeError:
        coro.close()
        return False, None

    return True, None
