import asyncio
import contextlib
from typing import Set

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider


class AsyncTaskScope(EventLoopProvider):
    """Automatically cleanup tasks when exiting a context, to prevent leaks."""

    loop: asyncio.AbstractEventLoop
    tasks: Set[asyncio.Task]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = set()

    def create_task(self, *args, **kwargs):
        # SAFETY: By design this is safe as long as the API agreements of this class is upheld.
        task = self.event_loop.create_task(*args, **kwargs)
        self.tasks.add(task)
        return task

    @contextlib.contextmanager
    def scope(self, *args, **kwargs):
        task = self.create_task(*args, **kwargs)

        try:
            yield task
        finally:
            self.tasks.remove(task)
            task.cancel()

    def cancel_all(self):
        for task in self.tasks:
            if task.done():
                continue

            task.cancel()

        self.tasks.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cancel_all()
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cancel_all()
        return False

    def __del__(self):
        self.cancel_all()
