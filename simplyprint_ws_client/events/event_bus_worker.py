import asyncio
import logging
from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Union, Hashable, NamedTuple, Optional, Dict, Tuple, Any, Coroutine

from .emitter import TEvent, Emitter
from .event_bus import EventBus
from ..shared.utils.stoppable import StoppableThread, AsyncStoppable, StoppableInterface


class _EventQueueItem(NamedTuple):
    is_async: bool
    event: Any
    args: Tuple
    kwargs: Dict


_TEventQueueValue = Optional[_EventQueueItem]

_MAX_QUEUE_SIZE = 10000


class EventBusWorker(Emitter[TEvent], StoppableInterface, ABC):
    event_bus: EventBus[TEvent]
    event_queue: Union[Queue[_TEventQueueValue], asyncio.Queue]
    logger: logging.Logger = logging.getLogger(__name__)
    maxsize: int

    def __init__(
        self,
        event_bus: EventBus[TEvent],
        *args,
        maxsize=_MAX_QUEUE_SIZE,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        self.event_bus = event_bus
        self.logger = logger or self.logger
        self.maxsize = maxsize

    @abstractmethod
    def emit_sync(
        self, event: Union[Hashable, TEvent], *args, **kwargs
    ) -> Union[None, Coroutine[Any, Any, None]]: ...

    @abstractmethod
    def emit(
        self, event: Union[Hashable, TEvent], *args, **kwargs
    ) -> Union[None, Coroutine[Any, Any, None]]: ...

    def _full_warning(self):
        if self.event_queue.full():
            self.logger.warning(
                f"Event queue worker is full, {self.event_queue.qsize()} events are pending!!! Expect degraded "
                f"performance."
            )

    def stop(self):
        super().stop()

        # Clear out queue and put a None to signal the end
        try:
            while True:
                self.event_queue.get_nowait()
        except (Empty, asyncio.QueueEmpty):
            pass

        self.event_queue.put_nowait(None)


class ThreadedEventBusWorker(EventBusWorker[TEvent], StoppableThread):
    def __init__(self, event_bus: EventBus[TEvent], **kwargs):
        EventBusWorker.__init__(self, event_bus, **kwargs)
        StoppableThread.__init__(self, **kwargs)
        self.event_queue = Queue(maxsize=self.maxsize)

    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if self.is_stopped():
            return

        self._full_warning()

        self.event_queue.put_nowait(_EventQueueItem(True, event, args, kwargs))

    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if self.is_stopped():
            return

        self._full_warning()

        self.event_queue.put(_EventQueueItem(False, event, args, kwargs))

    def run(self):
        while not self.is_stopped():
            item = self.event_queue.get()

            if item is None:
                break

            self.event_queue.task_done()

            try:
                if item.is_async:
                    self.event_bus.emit_task(item.event, *item.args, **item.kwargs)
                else:
                    self.event_bus.emit_sync(item.event, *item.args, **item.kwargs)
            except Exception as e:
                self.logger.error(
                    f"Error while processing event {item.event}", exc_info=e
                )


class _SharedQueueItem(NamedTuple):
    event_bus: EventBus
    is_async: bool
    event: Any
    args: Tuple
    kwargs: Dict


class SharedEventBusWorker(StoppableThread):
    """One thread + one queue dispatching events for many EventBuses.

    Replaces N per-client :class:`ThreadedEventBusWorker` threads with a single
    dispatcher: threads are sparse, and one connector may run many printers. Each
    bus that wants its events delivered here takes a lightweight handle from
    :meth:`worker_for` (which satisfies the emit surface the connection pool
    drives); the handle tags every event with its bus, and this worker pops each
    item and dispatches it to the right bus. Per-bus handlers stay isolated --
    only the dispatch thread and queue are shared, and within that thread events
    are processed in arrival order across all buses.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        maxsize: int = _MAX_QUEUE_SIZE,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.maxsize = maxsize
        self.logger = logger or self.logger
        self.event_queue: "Queue[Optional[_SharedQueueItem]]" = Queue(maxsize=maxsize)

    def worker_for(self, event_bus: EventBus) -> "SharedEventBusWorkerHandle":
        """A per-bus handle that routes ``event_bus``'s events through this worker."""
        return SharedEventBusWorkerHandle(self, event_bus)

    def _full_warning(self) -> None:
        if self.event_queue.full():
            self.logger.warning(
                f"Shared event-bus worker queue is full, {self.event_queue.qsize()} "
                f"events are pending!!! Expect degraded performance."
            )

    def _enqueue(self, item: _SharedQueueItem, *, block: bool) -> None:
        if self.is_stopped():
            return
        self._full_warning()
        if block:
            self.event_queue.put(item)
        else:
            self.event_queue.put_nowait(item)

    def run(self):
        while not self.is_stopped():
            item = self.event_queue.get()

            if item is None:
                break

            self.event_queue.task_done()

            try:
                if item.is_async:
                    item.event_bus.emit_task(item.event, *item.args, **item.kwargs)
                else:
                    item.event_bus.emit_sync(item.event, *item.args, **item.kwargs)
            except Exception as e:
                self.logger.error(
                    f"Error while processing event {item.event}", exc_info=e
                )

    def stop(self):
        super().stop()

        # Clear out the queue and put a None to signal the run loop to end.
        try:
            while True:
                self.event_queue.get_nowait()
        except (Empty, asyncio.QueueEmpty):
            pass

        self.event_queue.put_nowait(None)


class SharedEventBusWorkerHandle:
    """One bus's view onto a :class:`SharedEventBusWorker`.

    Satisfies the emit surface the connection pool drives (``emit_sync``) by
    enqueueing the event -- tagged with its bus -- onto the shared worker's
    queue. :meth:`stop` detaches just this bus (its further emits are dropped, as
    a stopped per-client worker dropped them); the shared worker keeps serving
    every other bus.
    """

    def __init__(self, worker: SharedEventBusWorker, event_bus: EventBus) -> None:
        self._worker = worker
        self.event_bus = event_bus
        self._stopped = False

    def is_stopped(self) -> bool:
        return self._stopped

    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if self._stopped:
            return
        self._worker._enqueue(
            _SharedQueueItem(self.event_bus, True, event, args, kwargs), block=False
        )

    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if self._stopped:
            return
        self._worker._enqueue(
            _SharedQueueItem(self.event_bus, False, event, args, kwargs), block=True
        )

    def stop(self) -> None:
        self._stopped = True


class AsyncEventBusWorker(EventBusWorker[TEvent], AsyncStoppable):
    def __init__(self, event_bus: EventBus[TEvent], *args, **kwargs):
        EventBusWorker.__init__(self, event_bus, *args, **kwargs)
        AsyncStoppable.__init__(self, *args, **kwargs)
        self.event_queue = asyncio.Queue(maxsize=self.maxsize)

    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if self.is_stopped():
            return

        self._full_warning()

        await self.event_queue.put(_EventQueueItem(True, event, args, kwargs))

    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if self.is_stopped():
            return

        self._full_warning()

        self.event_queue.put_nowait(_EventQueueItem(False, event, args, kwargs))

    async def run(self):
        while not self.is_stopped():
            item = await self.event_queue.get()

            if item is None:
                break

            self.event_queue.task_done()

            try:
                if item.is_async:
                    await self.event_bus.emit(item.event, *item.args, **item.kwargs)
                else:
                    self.event_bus.emit_sync(item.event, *item.args, **item.kwargs)
            except Exception as e:
                self.logger.error(
                    f"Error while processing event {item.event}", exc_info=e
                )
