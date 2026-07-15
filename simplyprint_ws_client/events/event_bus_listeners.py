import asyncio
import functools
import heapq
from enum import Enum
from typing import (
    Callable,
    List,
    Union,
    Tuple,
    Optional,
    Iterable,
    Iterator,
)

try:
    from typing import Unpack, NotRequired, TypedDict
except ImportError:
    from typing_extensions import Unpack, NotRequired, TypedDict


class ListenerUniqueness(Enum):
    """The level of uniqueness for an event listener

    Could also be called a replacement strategy.
    """

    NONE = 0
    PRIORITY = 1
    EXCLUSIVE = 2
    EXCLUSIVE_WITH_ERROR = 3


class ListenerLifetime(Enum):
    FOREVER = 0
    ONCE = 1


class EventBusListenerOptions(TypedDict):
    lifetime: NotRequired[ListenerLifetime]
    priority: NotRequired[int]
    unique: NotRequired[ListenerUniqueness]


def _is_async(handler: Callable) -> bool:
    if isinstance(handler, functools.partial):
        return _is_async(handler.func)

    return asyncio.iscoroutinefunction(handler)


class EventBusListener:
    __slots__ = ("lifetime", "priority", "handler", "is_async")

    lifetime: ListenerLifetime
    priority: int
    handler: Callable
    is_async: bool

    async def __call__(self, *args, **kwargs):
        if not self.is_async:
            return self.handler(*args, **kwargs)

        return await self.handler(*args, **kwargs)

    def __init__(
        self, lifetime: ListenerLifetime, priority: int, handler: Callable
    ) -> None:
        self.lifetime = lifetime
        self.priority = priority
        self.handler = handler
        self.is_async = _is_async(handler)

    def __lt__(self, other: "EventBusListener") -> bool:
        return self.priority < other.priority

    def __eq__(self, other: Union["EventBusListener", Callable]) -> bool:
        if isinstance(other, EventBusListener):
            return self.handler == other.handler

        return self.handler == other

    def __hash__(self) -> int:
        return hash(self.handler)

    def __repr__(self):
        return (
            f"EventListener(handler={self.handler!r}, priority={self.priority}, "
            f"is_async={self.is_async})"
        )


class EventBusListeners(Iterable[EventBusListener]):
    __slots__ = ("listeners", "sync_only", "_ordered", "_fast_emit")

    listeners: List[Tuple[int, EventBusListener]]
    sync_only: bool
    _ordered: Optional[List[EventBusListener]]
    _fast_emit: Optional[bool]

    def __init__(self, sync_only=False) -> None:
        self.listeners = []
        self.sync_only = sync_only
        self._ordered = None
        self._fast_emit = None

    def add(
        self, listener: Callable, **kwargs: Unpack[EventBusListenerOptions]
    ) -> None:
        unique = kwargs.get("unique", ListenerUniqueness.NONE)
        priority = kwargs.get("priority", 0)
        lifetime = kwargs.get("lifetime", ListenerLifetime.FOREVER)

        # Handle replacement strategy.
        if (
            unique == ListenerUniqueness.EXCLUSIVE_WITH_ERROR
            and len(self.listeners) > 0
        ):
            raise ValueError("Exclusive listener already registered, raising an error.")

        if unique == ListenerUniqueness.EXCLUSIVE:
            self.listeners = []
        elif unique == ListenerUniqueness.PRIORITY:
            # Remove all listeners with the same priority
            self.listeners = [(p, l) for p, l in self.listeners if p != priority]  # noqa: E741

        if self.contains(listener):
            raise ValueError("Listener already registered")

        listener = EventBusListener(lifetime, priority, listener)

        if self.sync_only and listener.is_async:
            raise ValueError("Listener marked as sync only but is async.")

        heapq.heappush(self.listeners, (priority, listener))
        self._invalidate()

    def remove(self, listener: Callable) -> None:
        for i, (_, reg_listener) in reversed(list(enumerate(self.listeners))):
            if reg_listener == listener:
                self.listeners.pop(i)
                self._invalidate()
                break

    def contains(self, listener: Callable) -> bool:
        for _, reg_listener in self.listeners:
            if reg_listener == listener:
                return True

        return False

    def __iter__(self) -> Iterator[EventBusListener]:
        """Iterate over listeners in priority order."""
        for listener in self.ordered():
            # Only allow once shot listener to be consumed once.
            if listener.lifetime is ListenerLifetime.ONCE:
                self.remove(listener)

            yield listener

    def __len__(self) -> int:
        return len(self.listeners)

    def ordered(self) -> List[EventBusListener]:
        """Listeners in priority order, cached until registration changes."""
        if self._ordered is None:
            self._ordered = [
                listener
                for _, listener in heapq.nlargest(
                    len(self.listeners), list(self.listeners)
                )
            ]
        return self._ordered

    def fast_emit_listeners(self) -> Optional[List[EventBusListener]]:
        """Return the cached listeners if direct dispatch can preserve semantics."""
        if self._fast_emit is None:
            self._fast_emit = all(
                listener.lifetime is not ListenerLifetime.ONCE
                for _, listener in self.listeners
            )
        if not self._fast_emit:
            return None
        return self.ordered()

    def _invalidate(self) -> None:
        self._ordered = None
        self._fast_emit = None
