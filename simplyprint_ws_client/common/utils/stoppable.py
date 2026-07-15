import asyncio
import multiprocessing
import threading
from typing import Optional


class SyncStoppable:
    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def clear(self) -> None:
        self._stop_event.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._stop_event.wait(timeout)


class AsyncStoppable:
    def __init__(self) -> None:
        self._stop_event = asyncio.Event()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def clear(self) -> None:
        self._stop_event.clear()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        if self.is_stopped():
            return True

        if timeout is None:
            return await self._stop_event.wait()

        try:
            return await asyncio.wait_for(self._stop_event.wait(), timeout)
        except asyncio.TimeoutError:
            return self.is_stopped()


class ProcessStoppable:
    def __init__(self) -> None:
        self._stop_event = multiprocessing.Event()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def clear(self) -> None:
        self._stop_event.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._stop_event.wait(timeout)
