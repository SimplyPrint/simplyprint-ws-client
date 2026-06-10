import asyncio
import multiprocessing
import threading
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import Optional, Union, TypeVar, Generic

from simplyprint_ws_client.common.asyncio.async_task_scope import AsyncTaskScope
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

TStopEvent = TypeVar(
    "TStopEvent", bound=Union[threading.Event, asyncio.Event, multiprocessing.Condition]
)
TCondition = TypeVar(
    "TCondition",
    bound=Union[threading.Condition, asyncio.Condition, multiprocessing.Condition],
)
TAnyStoppable = Union["Stoppable", TStopEvent]


def _cleanup_kwargs(kwargs, *allowed_keys):
    for key in list(kwargs.keys()):
        if key not in allowed_keys:
            del kwargs[key]

    return kwargs


class StoppableInterface(ABC):
    @abstractmethod
    def is_stopped(self) -> bool: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


class Stoppable(Generic[TStopEvent, TCondition], StoppableInterface, ABC):
    """A Stoppable class represents an object that has
    some never ending logic that needs to be able to
    terminate for cleanup.

    This paradigm implements "pausing" or "restarting"
    as always stopping entirely first before creating
    a new instance.

    Also has the concept of a parent stop event,
    when the child is stopped, the parent is not stopped.

    But when the parent is stopped, the child is stopped.

    """

    # Implement wait group with a chained condition
    __condition: Optional[TCondition]

    __parent_stop_event: Optional[TStopEvent]
    __stop_event: TStopEvent

    @classmethod
    def _extract_stop_event(
        cls,
        stoppable: Optional[TAnyStoppable] = None,
        parent=False,
        default: Optional[TStopEvent] = None,
    ) -> Optional[TStopEvent]:
        if not stoppable:
            return default

        if isinstance(stoppable, Stoppable):
            if parent and stoppable._parent_stop_event_property:
                return stoppable._parent_stop_event_property

            return stoppable._stop_event_property

        if not isinstance(stoppable, (threading.Event, asyncio.Event, Event)):
            return default

        return stoppable

    @staticmethod
    def _extract_condition(
        nested_stoppable: Optional["SyncStoppable"] = None,
        parent_stoppable: Optional["SyncStoppable"] = None,
        default: Optional[TCondition] = None,
        **kwargs,
    ) -> TCondition:
        stoppable = nested_stoppable or parent_stoppable

        if not stoppable or not isinstance(stoppable, Stoppable):
            return default

        return stoppable._condition_property

    @abstractmethod
    def __init__(
        self,
        nested_stoppable: Optional[TAnyStoppable] = None,
        parent_stoppable: Optional[TAnyStoppable] = None,
        condition: Optional[TCondition] = None,
        default_condition: Optional[TCondition] = None,
        **kwargs,
    ):
        # Only set a condition if it is explicitly passed (and used)
        # for instance, the async stoppable does not use a condition.
        if default_condition:
            self.__condition = self._extract_condition(
                nested_stoppable, parent_stoppable, default=default_condition
            )
        else:
            self.__condition = condition

        self.__parent_stop_event = self._extract_stop_event(
            parent_stoppable, parent=True
        )
        self.__stop_event = self._extract_stop_event(nested_stoppable)

    def is_stopped(self):
        if self._parent_stop_event_property is not None:
            return self.__stop_event.is_set() or self.__parent_stop_event.is_set()

        return self.__stop_event.is_set()

    def stop(self):
        self.__stop_event.set()

        if self.__condition:
            with self.__condition:
                self.__condition.notify_all()

    def clear(self):
        self.__stop_event.clear()

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> bool: ...

    @property
    def _stop_event_property(self):
        return self.__stop_event

    @_stop_event_property.setter
    def _stop_event_property(self, event: TStopEvent):
        self.__stop_event = event

    @property
    def _parent_stop_event_property(self):
        return self.__parent_stop_event

    @property
    def _condition_property(self):
        return self.__condition


class SyncStoppable(Stoppable[threading.Event, threading.Condition]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, default_condition=threading.Condition())
        self._stop_event_property = self._stop_event_property or threading.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        with self._condition_property:
            self._condition_property.wait(timeout)
        return self.is_stopped()


class AsyncStoppable(Stoppable[asyncio.Event, asyncio.Condition]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event_property = self._stop_event_property or asyncio.Event()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """
        FIXME: Currently AsyncStoppable does not require the implementer to also extend
        EventLoopProvider, so we have no explicit loop binding *always* although it was
        previously assumed, for now we always use the `default` loop which currently
        is the running loop.
        """
        with AsyncTaskScope(provider=EventLoopProvider.default()) as task_scope:
            try:
                await asyncio.wait(
                    map(
                        task_scope.create_task,
                        [
                            self._stop_event_property.wait(),
                        ]
                        + (
                            [self._parent_stop_event_property.wait()]
                            if self._parent_stop_event_property
                            else []
                        ),
                    ),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                return self.is_stopped()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return self.is_stopped()


class ProcessStoppable(Stoppable[Event, multiprocessing.Condition]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, default_condition=multiprocessing.Condition())
        self._stop_event_property = self._stop_event_property or multiprocessing.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self.is_stopped():
            return True

        with self._condition_property:
            self._condition_property.wait(timeout)
        return self.is_stopped()


class StoppableThread(SyncStoppable, threading.Thread, ABC):
    def __init__(self, *args, **kwargs):
        SyncStoppable.__init__(self, **kwargs)
        _cleanup_kwargs(kwargs, "group", "target", "name", "args", "kwargs", "daemon")
        threading.Thread.__init__(self, **kwargs)

    @abstractmethod
    def run(self): ...


class StoppableProcess(ProcessStoppable, multiprocessing.Process, ABC):
    def __init__(self, *args, **kwargs):
        ProcessStoppable.__init__(self, **kwargs)
        _cleanup_kwargs(kwargs, "group", "target", "name", "args", "kwargs", "daemon")
        multiprocessing.Process.__init__(self, **kwargs)

    @abstractmethod
    def run(self): ...
