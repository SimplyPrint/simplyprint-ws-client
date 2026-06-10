import asyncio
from asyncio import AbstractEventLoop
from typing import Optional, Callable, Coroutine, Any, Generic, TypeVar

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

T = TypeVar("T")


class ContinuousTask(Generic[T], EventLoopProvider[AbstractEventLoop]):
    """
    A continuous task manages a single asyncio task that is created on the first invocation of the task.
    Once it has completed it is created anew. Simply put it is a task container.
    """

    task: Optional[asyncio.Task]
    coro_factory: Callable[..., Coroutine[T, Any, Any]]

    def __init__(self, coro_factory: Callable[..., Coroutine[T, Any, Any]], **kwargs):
        Generic.__init__(self)
        EventLoopProvider.__init__(self, **kwargs)

        self.coro_factory = coro_factory
        self.task = None

    def __await__(self):
        return self.schedule().__await__()

    def __bool__(self):
        return self.task is not None

    def schedule(self, *args, **kwargs) -> asyncio.Task:
        if self.task is None:
            # SAFETY: At most 1 managed instance.
            self.task = self.event_loop.create_task(self.coro_factory(*args, **kwargs))

        return self.task

    def done(self) -> bool:
        return self.task is not None and self.task.done()

    def cancelled(self) -> bool:
        return self.task is not None and self.task.cancelled()

    def result(self) -> T:
        if self.task is None:
            raise RuntimeError("Task not started.")

        return self.task.result()

    def exception(self) -> Optional[BaseException]:
        if self.task is None:
            raise RuntimeError("Task not started.")

        return self.task.exception()

    def cancel(self) -> bool:
        if self.task is None or self.task.cancelled():
            return False

        return self.task.cancel()

    def pop(self) -> Optional[asyncio.Task]:
        if not self.done():
            self.cancel()

        task = self.task
        self.task = None
        return task

    def discard(self) -> None:
        """Special case of pop, consume result if it exists."""
        task = self.pop()

        if task is None or not task.done():
            return

        if not task.exception():
            _ = task.result()
