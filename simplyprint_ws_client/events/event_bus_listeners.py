import asyncio
import functools
import heapq
import inspect
from enum import Enum
from typing import (
    Callable,
    List,
    Union,
    Tuple,
    NamedTuple,
    Optional,
    get_args,
    Iterable,
    Iterator,
)

try:
    from typing import Unpack, NotRequired, TypedDict
except ImportError:
    from typing_extensions import Unpack, NotRequired, TypedDict

from simplyprint_ws_client.events.emitter import Emitter


class ListenerUniqueness(Enum):
    """The level of uniqueness for an event listener

    Could also be called a replacement strategy.
    """

    NONE = 0
    PRIORITY = 1
    EXCLUSIVE = 2
    EXCLUSIVE_WITH_ERROR = 3


class ListenerLifetime(NamedTuple):
    """Implement listener lifetime options as a tagged-union
    to support value based lifetimes such as max-calls in the future.
    """

    ...


class ListenerLifetimeOnce(ListenerLifetime):
    """An event listener that is removed after being called once."""

    ...


class ListenerLifetimeForever(ListenerLifetime):
    """A normal event listener that is never removed."""

    ...


class EventBusListenerOptions(TypedDict):
    lifetime: NotRequired[ListenerLifetime]
    priority: NotRequired[int]
    unique: NotRequired[ListenerUniqueness]


class EventBusListenersOptions(EventBusListenerOptions, TypedDict):
    generic: NotRequired[bool]


def _is_async(handler: Callable) -> bool:
    if isinstance(handler, functools.partial):
        return _is_async(handler.func)

    return asyncio.iscoroutinefunction(handler)


class EventBusListener:
    __slots__ = ("lifetime", "priority", "handler", "is_async", "forward_emitter")

    lifetime: ListenerLifetime
    priority: int
    handler: Callable
    is_async: bool
    forward_emitter: Optional[str]

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
        self.forward_emitter = None

        # If function takes a named argument with the type Emitter, store that kwarg name.
        signature = inspect.signature(handler)

        for parameter in signature.parameters.values():
            annotation = parameter.annotation

            # Check if the annotation is a type or a type hint. And whether it is a subclass of Emitter.
            if not any(
                issubclass(cls, Emitter)
                for cls in get_args(annotation) + (annotation,)
                if isinstance(cls, type)
            ):
                continue

            self.forward_emitter = parameter.name
            break

    def __lt__(self, other: "EventBusListener") -> bool:
        return self.priority < other.priority

    def __eq__(self, other: Union["EventBusListener", Callable]) -> bool:
        if isinstance(other, EventBusListener):
            return self.handler == other.handler

        return self.handler == other

    def __hash__(self) -> int:
        return hash(self.handler)

    def __repr__(self):
        if isinstance(self.handler, functools.partial):
            name = self.handler.func.__name__
        elif hasattr(self.handler, "__name__"):
            name = self.handler.__name__
        else:
            name = "Unknown"

        return f"EventListener(handler={name}, priority={self.priority}, is_async={self.is_async})"


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
        lifetime = kwargs.get("lifetime", ListenerLifetimeForever(**{}))

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
            if isinstance(listener.lifetime, ListenerLifetimeOnce):
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
                not isinstance(listener.lifetime, ListenerLifetimeOnce)
                and listener.forward_emitter is None
                for _, listener in self.listeners
            )
        if not self._fast_emit:
            return None
        return self.ordered()

    def _invalidate(self) -> None:
        self._ordered = None
        self._fast_emit = None
