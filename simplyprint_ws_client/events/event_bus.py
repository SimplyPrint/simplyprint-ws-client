import asyncio
import concurrent.futures
import functools
from asyncio import AbstractEventLoop
from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    TypeVar,
    Union,
    overload,
    Type,
    Any,
    Tuple,
    Iterable,
    Iterator,
    Generic,
    final,
)

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from simplyprint_ws_client.events.emitter import Emitter, TEvent
from simplyprint_ws_client.events.event import Event
from simplyprint_ws_client.events.event_bus_listeners import (
    EventBusListeners,
    EventBusListener,
    EventBusListenerOptions,
)
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

#: The concrete event class a class-keyed ``on(...)`` registration listens for.
#: It cannot be bound to the bus's ``TEvent`` (a TypeVar cannot bound another),
#: so it is inferred per call from the class the caller passes.
E = TypeVar("E")


@final
class _EmitGenerator(Generic[TEvent]):
    """The one per-listener dispatch core every emit path drives.

    Owns the shared loop mechanics -- the stop-event check, the per-listener
    argument snapshot, and the listener return-value
    protocol (:func:`_update_args`) -- so the async, sync and fast emit paths
    cannot drift apart. A path iterates it, invokes each listener however it
    must (awaited or direct), and feeds the return value back via
    :meth:`update`.
    """

    __slots__ = ("listeners", "event", "args", "kwargs", "_stop_check")
    listeners: Iterator[EventBusListener]

    event: Union[Hashable, TEvent]
    args: Tuple[Any, ...]
    kwargs: Dict[Any, Any]

    def __init__(
        self,
        listeners: Iterable[EventBusListener],
        event: Union[Hashable, TEvent],
        args: Tuple[Any, ...],
        kwargs: Dict[Any, Any],
    ):
        self.listeners = iter(listeners)
        self.event = event
        self.args = _initialize_args(self.event, args)
        self.kwargs = kwargs
        self._stop_check = isinstance(event, Event)

    def update(self, returned_args: Union[Tuple[Any, ...], Any, None]):
        if returned_args is None:
            return
        self.args = _update_args(self.event, self.args, returned_args)

    def __next__(self) -> Tuple[EventBusListener, Tuple[Any, ...], Dict[Any, Any]]:
        if self._stop_check and self.event.is_stopped():
            raise StopIteration()

        event_listener = next(self.listeners)

        args = self.args
        kwargs = self.kwargs

        return event_listener, args, kwargs

    def __iter__(self):
        return self


def _initialize_args(
    event: Union[Hashable, TEvent], args: Tuple[Any, ...]
) -> Tuple[Any, ...]:
    """Pass canonical event instances as the first listener argument."""
    if isinstance(event, Event):
        return (event,) + args

    return args


def _update_args(
    event: Union[Hashable, TEvent],
    args: Tuple[Any, ...],
    returned_args: Union[Tuple[Any, ...], Any, None] = None,
) -> Tuple[Any, ...]:
    """
    Transform listener return values into arguments for the next listener.

    - If `returned_args` is None, the original arguments will be used.
    - If `returned_args` is an Event, it will replace the original event.
    - If `returned_args` is something that is not an event, the original arguments will be replaced.
    - If `returned_args` is an empty tuple, the original arguments will be replaced.
    - If `returned_args` is a tuple that starts with an Event, it replaces everything.
    """
    if returned_args is None:
        return args

    # A canonical Event instance is always first in args.
    event_is_first = isinstance(event, Event) and len(args) > 0 and args[0] is event

    if not isinstance(returned_args, tuple):
        if event_is_first:
            return (
                (returned_args, *args[1:])
                if isinstance(returned_args, Event)
                else (event, returned_args)
            )

        return (returned_args,)

    if len(returned_args) == 0:
        return (event,) if event_is_first else ()

    if event_is_first:
        return (
            returned_args
            if isinstance(returned_args[0], Event)
            else (event, *returned_args)
        )

    return returned_args


class EventBus(Emitter[TEvent]):
    __slots__ = ("listeners", "event_loop_provider")

    listeners: Dict[Hashable, EventBusListeners]
    event_loop_provider: EventLoopProvider[AbstractEventLoop]

    def __init__(
        self, event_loop_provider: Optional[EventLoopProvider[AbstractEventLoop]] = None
    ) -> None:
        self.event_loop_provider = event_loop_provider or EventLoopProvider.default()
        self.listeners = {}

    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        generator = self._emit_generator(event, args, kwargs)
        if generator is None:
            return

        for listener, nargs, nkwargs in generator:
            if listener.is_async:
                ret = await listener.handler(*nargs, **nkwargs)
            else:
                ret = listener.handler(*nargs, **nkwargs)
            generator.update(ret)

    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        generator = self._emit_generator(event, args, kwargs)
        if generator is None:
            return

        for listener, nargs, nkwargs in generator:
            # Only invoke non-async listeners.
            if listener.is_async:
                continue
            ret = listener.handler(*nargs, **nkwargs)
            generator.update(ret)

    def _emit_generator(
        self, event: Union[Hashable, TEvent], args: Tuple[Any, ...], kwargs: Dict
    ) -> Optional[_EmitGenerator]:
        """Build the shared dispatch core for one emit, or ``None`` for no-op.

        Picks the cached fast listener list when there are no one-shot
        listeners; otherwise it iterates the live registration collection.
        """
        listeners = self._listeners_for(event)
        if listeners is None:
            return None

        iterable: Iterable[EventBusListener] = (
            listeners.fast_emit_listeners() or listeners
        )

        return _EmitGenerator(iterable, event, args, kwargs)

    def emit_task(
        self, event: Union[Hashable, TEvent], *args, **kwargs
    ) -> "concurrent.futures.Future":
        """Allows for synchronous emitting of events. Useful cross-thread communication.

        Returns the ``concurrent.futures.Future`` from
        :func:`asyncio.run_coroutine_threadsafe` - block on it with
        ``.result()``; it is not awaitable."""
        return asyncio.run_coroutine_threadsafe(
            self.emit(event, *args, **kwargs), self.event_loop_provider.event_loop
        )

    def emit_wrap(
        self, event: Union[Hashable, TEvent], sync_only=False, blocking=False
    ) -> Callable:
        """
        Returns a curried function that emits the given event with any arguments passed to it.

        When sync_only is specified the function will only invoke synchronous listeners.

        If blocking is true we will block synchronously until all listeners have been invoked.

        The fallback is to emit the event asynchronously as a task in the provided event loop.
        """

        if sync_only:
            if not blocking:
                raise NotImplementedError(
                    "Synchronous emitting is not supported without blocking; "
                    "use blocking=True, or deliver onto the loop via a Courier."
                )

            emit_func = self.emit_sync
        else:
            emit_func = self.emit if blocking else self.emit_task

        assert emit_func is not None

        return functools.partial(emit_func, event)

    @overload
    def on(
        self,
        event_type: Type[E],
        listener: Callable[[E], object],
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable[[E], object]: ...

    @overload
    def on(
        self,
        event_type: Type[E],
        listener: None = ...,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable[[Callable[[E], object]], Callable[[E], object]]: ...

    @overload
    def on(
        self,
        event_type: Hashable,
        listener: Callable,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable: ...

    @overload
    def on(
        self,
        event_type: Hashable,
        listener: None = ...,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable[[Callable], Callable]: ...

    def on(
        self,
        event_type: Hashable,
        listener: Optional[Callable] = None,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable:
        if listener is None:
            return lambda registered: self._register_listener(
                event_type, registered, **kwargs
            )

        return self._register_listener(event_type, listener, **kwargs)

    def off(self, event_type: Hashable, listener: Callable) -> None:
        """Remove a listener from the event bus."""
        if event_type not in self.listeners:
            return

        self.listeners[event_type].remove(listener)

        if len(self.listeners[event_type]) == 0:
            self.listeners.pop(event_type)

    def clear(self, *event_types: Hashable) -> None:
        """Clear all listeners for a given event type."""
        for event_type in event_types:
            self.listeners.pop(event_type, None)

    def clear_all(self) -> None:
        """Drop every listener for every event type."""
        self.listeners.clear()

    @staticmethod
    def _event_key(event: Union[Hashable, TEvent]) -> Hashable:
        if isinstance(event, Event):
            return event.__class__
        return event

    def _listeners_for(
        self, event: Union[Hashable, TEvent]
    ) -> Optional[EventBusListeners]:
        return self.listeners.get(self._event_key(event))

    def _register_listener(
        self,
        event_type: Union[Hashable, TEvent],
        listener: Callable,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable:
        """Registers a single listener for a given event type."""
        if event_type not in self.listeners:
            # An event can be marked as "sync_only", meaning non async listeners can be attached.
            self.listeners[event_type] = EventBusListeners(
                event_type.is_sync_only()
                if isinstance(event_type, type) and issubclass(event_type, Event)
                else False
            )

        self.listeners[event_type].add(listener, **kwargs)
        return listener
