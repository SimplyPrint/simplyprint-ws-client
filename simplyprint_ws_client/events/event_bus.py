import asyncio
import concurrent.futures
import functools
from asyncio import AbstractEventLoop
from itertools import chain
from typing import (
    Callable,
    Dict,
    Generator,
    Hashable,
    Optional,
    TypeVar,
    Union,
    get_args,
    overload,
    Type,
    Any,
    Tuple,
    Iterable,
    Iterator,
    Generic,
    TYPE_CHECKING,
    Set,
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

if TYPE_CHECKING:
    from simplyprint_ws_client.events.event_bus_middleware import (
        EventBusMiddleware,
    )

#: The concrete event class a class-keyed ``on(...)`` registration listens for.
#: It cannot be bound to the bus's ``TEvent`` (a TypeVar cannot bound another),
#: so it is inferred per call from the class the caller passes.
E = TypeVar("E")


@final
class _EmitGenerator(Generic[TEvent]):
    """The one per-listener dispatch core every emit path drives.

    Owns the shared loop mechanics -- the stop-event check, the per-listener
    argument snapshot (with emitter forwarding) and the listener return-value
    protocol (:func:`_update_args`) -- so the async, sync and fast emit paths
    cannot drift apart. A path iterates it, invokes each listener however it
    must (awaited or direct), and feeds the return value back via
    :meth:`update`.
    """

    __slots__ = ("event_bus", "listeners", "event", "args", "kwargs", "_stop_check")

    event_bus: "EventBus"
    listeners: Iterator[EventBusListener]

    event: Union[Hashable, TEvent]
    args: Tuple[Any, ...]
    kwargs: Dict[Any, Any]

    def __init__(
        self,
        event_bus: "EventBus",
        listeners: Iterable[EventBusListener],
        event: Union[Hashable, TEvent],
        args: Tuple[Any, ...],
        kwargs: Dict[Any, Any],
    ):
        self.event_bus = event_bus
        self.listeners = iter(listeners)
        self.event = event
        self.args = _initialize_args(self.event_bus, self.event, args)
        self.kwargs = kwargs
        self._stop_check = isinstance(event, Event)

    def update(self, returned_args: Union[Tuple[Any, ...], Any, None]):
        if returned_args is None:
            return
        self.args = _update_args(self.event_bus, self.event, self.args, returned_args)

    def __next__(self) -> Tuple[EventBusListener, Tuple[Any, ...], Dict[Any, Any]]:
        if self._stop_check and self.event.is_stopped():
            raise StopIteration()

        event_listener = next(self.listeners)

        args = self.args
        kwargs = self.kwargs

        # Pass event bus to listener if it has a named argument with the type Emitter.
        if event_listener.forward_emitter:
            kwargs = kwargs.copy()
            kwargs[event_listener.forward_emitter] = self.event_bus

        return event_listener, args, kwargs

    def __iter__(self):
        return self


def _initialize_args(
    event_bus: "EventBus", event: Union[Hashable, TEvent], args: Tuple[Any, ...]
) -> Tuple[Any, ...]:
    """If the event is an instance of the event class, pass it as the first argument."""
    if isinstance(event, event_bus.event_klass):
        return (event,) + args

    return args


def _update_args(
    event_bus: "EventBus",
    event: Union[Hashable, TEvent],
    args: Tuple[Any, ...],
    returned_args: Union[Tuple[Any, ...], Any, None] = None,
) -> Tuple[Any, ...]:
    """
    Transform listener return values into arguments for the next listener.

    - If `returned_args` is None, the original arguments will be used.
    - If `returned_args` is an event of type klass, it will replace the original event.
    - If `returned_args` is something that is not an event, the original arguments will be replaced.
    - If `returned_args` is an empty tuple, the original arguments will be replaced.
    - If `returned_args` is a tuple that includes an event of type klass, it will replace everything.
    """
    if returned_args is None:
        return args

    # If the event is an instance of the event class it is always first in args.
    event_is_first = (
        isinstance(event, event_bus.event_klass) and len(args) > 0 and args[0] == event
    )

    if not isinstance(returned_args, tuple):
        if event_is_first:
            return (
                (returned_args, *args[1:])
                if isinstance(returned_args, event_bus.event_klass)
                else (event, returned_args)
            )

        return (returned_args,)

    if len(returned_args) == 0:
        return (event,) if event_is_first else ()

    if event_is_first:
        return (
            returned_args
            if isinstance(returned_args[0], event_bus.event_klass)
            else (event, *returned_args)
        )

    return returned_args


class EventBus(Emitter[TEvent]):
    __slots__ = ("listeners", "class_listeners", "event_klass", "event_loop_provider")

    # Middlewares are global event listeners.
    middleware: Set["EventBusMiddleware"]

    # Event specific listeners.
    listeners: Dict[Hashable, EventBusListeners]
    class_listeners: Dict[int, EventBusListeners]

    event_klass: Type[TEvent]
    event_loop_provider: EventLoopProvider[AbstractEventLoop]

    def __init__(
        self, event_loop_provider: Optional[EventLoopProvider[AbstractEventLoop]] = None
    ) -> None:
        self.event_loop_provider = event_loop_provider or EventLoopProvider.default()
        self.middleware = set()
        self.listeners = {}
        self.class_listeners = {}

        # Extract the generic type from the class otherwise
        # fallback to the default Event class
        try:
            self.event_klass = get_args(self.__class__.__orig_bases__[0])[0]
        except (AttributeError, KeyError, IndexError):
            self.event_klass = Event

        if not isinstance(self.event_klass, type):
            self.event_klass = Event

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

        Picks the cached fast listener list when direct dispatch preserves
        semantics (no middleware, no one-shot listeners, no emitter
        forwarding); otherwise the full middleware + registration iteration.
        """
        listeners = self.class_listeners.get(id(event.__class__))
        if listeners is None:
            listeners = self._listeners_for(event)
        if listeners is None and len(self.middleware) == 0:
            return None

        iterable: Iterable[EventBusListener]
        if len(self.middleware) == 0 and listeners is not None:
            iterable = listeners.fast_emit_listeners() or listeners
        else:
            iterable = chain(self.middleware, listeners or [])

        return _EmitGenerator(self, iterable, event, args, kwargs)

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
        generic: bool = ...,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable[[E], object]: ...

    @overload
    def on(
        self,
        event_type: Type[E],
        listener: None = ...,
        generic: bool = ...,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable[[Callable[[E], object]], Callable[[E], object]]: ...

    @overload
    def on(
        self,
        event_type: Hashable,
        listener: Callable,
        generic: bool = ...,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable: ...

    @overload
    def on(
        self,
        event_type: Hashable,
        listener: None = ...,
        generic: bool = ...,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable[[Callable], Callable]: ...

    def on(
        self,
        event_type: Hashable,
        listener: Optional[Callable] = None,
        generic: bool = False,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable:
        if listener is None:
            return lambda lst: self._register_listeners(
                event_type, lst, generic=generic, **kwargs
            )

        return self._register_listeners(event_type, listener, generic=generic, **kwargs)

    def off(self, event_type: Hashable, listener: Callable) -> None:
        """Remove a listener from the event bus."""
        if event_type not in self.listeners:
            return

        self.listeners[event_type].remove(listener)

        if len(self.listeners[event_type]) == 0:
            self.listeners.pop(event_type)
            if isinstance(event_type, type) and issubclass(event_type, Event):
                self.class_listeners.pop(id(event_type), None)

    def clear(self, *event_types: Hashable) -> None:
        """Clear all listeners for a given event type."""
        for event_type in event_types:
            self.listeners.pop(event_type, None)
            if isinstance(event_type, type) and issubclass(event_type, Event):
                self.class_listeners.pop(id(event_type), None)

    def clear_all(self) -> None:
        """Drop every listener for every event type."""
        self.listeners.clear()
        self.class_listeners.clear()

    @staticmethod
    def _event_key(event: Union[Hashable, TEvent]) -> Hashable:
        if isinstance(event, Event):
            return event.__class__
        return event

    def _listeners_for(
        self, event: Union[Hashable, TEvent]
    ) -> Optional[EventBusListeners]:
        if isinstance(event, Event):
            listeners = self.class_listeners.get(id(event.__class__))
            if listeners is not None:
                return listeners

        event_key = self._event_key(event)
        listeners = self.listeners.get(event_key)
        if listeners is None and event_key is not event:
            listeners = self.listeners.get(event)
        return listeners

    def _register_listeners(
        self,
        event_type: Union[Hashable, TEvent],
        listener: Callable,
        generic: bool = False,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> Callable:
        """
        Registers all listeners for a generic type given the type is an event type,
        otherwise wraps a single register call.
        """

        if not generic or not issubclass(event_type, self.event_klass):
            self._register_listener(event_type, listener, **kwargs)
            return listener

        for klass in self._iterate_subclasses(event_type):
            self._register_listener(klass, listener, **kwargs)

        return listener

    def _register_listener(
        self,
        event_type: Union[Hashable, TEvent],
        listener: Callable,
        **kwargs: Unpack[EventBusListenerOptions],
    ) -> None:
        """Registers a single listener for a given event type."""
        if event_type not in self.listeners:
            # An event can be marked as "sync_only", meaning non async listeners can be attached.
            self.listeners[event_type] = EventBusListeners(
                event_type.is_sync_only()
                if isinstance(event_type, type) and issubclass(event_type, Event)
                else False
            )
            if isinstance(event_type, type) and issubclass(event_type, Event):
                self.class_listeners[id(event_type)] = self.listeners[event_type]

        self.listeners[event_type].add(listener, **kwargs)

    def _iterate_subclasses(self, klass: type) -> Generator[type, None, None]:
        """Perform class introspection to construct listeners generically"""
        if not issubclass(klass, self.event_klass):
            raise TypeError(f"Expected type of {self.event_klass} but got {klass}")

        for subclass in klass.__subclasses__():
            yield from self._iterate_subclasses(subclass)

        yield klass
