import asyncio
import functools
from asyncio import AbstractEventLoop
from itertools import chain
from typing import (
    Callable,
    Dict,
    Generator,
    Hashable,
    Optional,
    Union,
    get_args,
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

from .emitter import Emitter, TEvent
from .event import Event
from .event_bus_listeners import (
    EventBusListeners,
    EventBusListener,
    EventBusListenerOptions,
)
from ..shared.asyncio.event_loop_provider import EventLoopProvider

if TYPE_CHECKING:
    from .event_bus_middleware import EventBusMiddleware


@final
class _EmitGenerator(Generic[TEvent]):
    """Stateful generator that updates arguments according to input and output."""

    __slots__ = ("event_bus", "listeners", "event", "args", "kwargs")

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
        self.args = self._initialize_args(self.event, args)
        self.kwargs = kwargs

    def update(self, returned_args: Union[Tuple[Any, ...], Any]):
        self.args = self._update_args(returned_args)

    def _initialize_args(
        self, event: Union[Hashable, TEvent], args: Tuple[Any, ...]
    ) -> Tuple[Any, ...]:
        """If the event is an instance of the event class, pass it as the first argument."""
        if isinstance(event, self.event_bus.event_klass):
            return (event,) + args

        return args

    def _update_args(
        self, returned_args: Union[Tuple[Any, ...], Any, None] = None
    ) -> Tuple[Any, ...]:
        """
        This function transforms the arguments returned by an event listener
        into the arguments that will be passed to the next listener.

        - If `returned_args` is None, the original arguments will be used.
        - If `returned_args` is an event of type klass, it will replace the original event.
        - If `returned_args` is something that is not an event, the original arguments will be replaced.
        - If `returned_args` is an empty tuple, the original arguments will be replaced.
        - If `returned_args` is a tuple that includes an event of type klass, it will replace everything.

        Given an event listener function like so:

            def listener(event: Event, *args, **kwargs) -> Event:
                # Do something with event
                return event


        The returned event will now take precedence over the original event.

        Given an event listener function like so:

            def listener(event: Event, *args, **kwargs) -> Tuple[Event, ...]:
                # Do something with event and return more arguments
                return event, ...


        Then returned_args will replace the entire argument list.
        """
        if returned_args is None:
            return self.args

        # If the event is an instance of the event class
        # it is always the first element of args.
        event_is_first = (
            isinstance(self.event, self.event_bus.event_klass)
            and len(self.args) > 0
            and self.args[0] == self.event
        )

        if not isinstance(returned_args, tuple):
            # The listener may return the modified event parameter in
            # the case the emitted event was of an event class type.
            # this will replace the event parameter with the modified one.
            if event_is_first:
                return (
                    (returned_args, *self.args[1:])
                    if isinstance(returned_args, self.event_bus.event_klass)
                    else (self.event, returned_args)
                )

            # Otherwise replace the entire argument list with the returned data.
            return (returned_args,)

        # If we return an empty tuple, we want to replace the entire argument list.
        if len(returned_args) == 0:
            return (self.event,) if event_is_first else ()

        # Now we have to handle the case where the listener returns a tuple.
        if event_is_first:
            return (
                returned_args
                if isinstance(returned_args[0], self.event_bus.event_klass)
                else (self.event, *returned_args)
            )

        # As a fallback we just return the returned arguments.
        return returned_args

    def __next__(self) -> Tuple[EventBusListener, Tuple[Any, ...], Dict[Any, Any]]:
        if isinstance(self.event, Event) and self.event.is_stopped():
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


class EventBus(Emitter[TEvent]):
    __slots__ = ("listeners", "event_klass", "event_loop_provider")

    # Middlewares are global event listeners.
    middleware: Set["EventBusMiddleware"]

    # Event specific listeners.
    listeners: Dict[Hashable, EventBusListeners]

    event_klass: Type[TEvent]
    event_loop_provider: EventLoopProvider[AbstractEventLoop]

    def __init__(
        self, event_loop_provider: Optional[EventLoopProvider[AbstractEventLoop]] = None
    ) -> None:
        self.event_loop_provider = event_loop_provider or EventLoopProvider.default()
        self.middleware = set()
        self.listeners = {}

        # Extract the generic type from the class otherwise
        # fallback to the default Event class
        try:
            self.event_klass = get_args(self.__class__.__orig_bases__[0])[0]
        except (AttributeError, KeyError, IndexError):
            self.event_klass = Event

        if not isinstance(self.event_klass, type):
            self.event_klass = Event

    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if event not in self.listeners and len(self.middleware) == 0:
            return

        generator = _EmitGenerator(
            self,
            chain(self.middleware, self.listeners.get(event, [])),
            event,
            args,
            kwargs,
        )

        for listener, nargs, nkwargs in generator:
            ret = await listener(*nargs, **nkwargs)
            generator.update(ret)

    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        if event not in self.listeners and len(self.middleware) == 0:
            return

        # Only invoke non-async functions.
        generator = _EmitGenerator(
            self,
            chain(
                self.middleware,
                filter(lambda lst: not lst.is_async, self.listeners.get(event, [])),
            ),
            event,
            args,
            kwargs,
        )

        for listener, nargs, nkwargs in generator:
            ret = listener.handler(*nargs, **nkwargs)
            generator.update(ret)

    def emit_task(
        self, event: Union[Hashable, TEvent], *args, **kwargs
    ) -> asyncio.Future:
        """Allows for synchronous emitting of events. Useful cross-thread communication."""
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
                    "Synchronous emitting is not supported without blocking, use EventBusWorker."
                )

            emit_func = self.emit_sync
        else:
            emit_func = self.emit if blocking else self.emit_task

        assert emit_func is not None

        return functools.partial(emit_func, event)

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

    def clear(self, *event_types: Hashable) -> None:
        """Clear all listeners for a given event type."""
        for event_type in event_types:
            self.listeners.pop(event_type, None)

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

        self.listeners[event_type].add(listener, **kwargs)

    def _iterate_subclasses(self, klass: type) -> Generator[type, None, None]:
        """Perform class introspection to construct listeners generically"""
        if not issubclass(klass, self.event_klass):
            raise TypeError(f"Expected type of {self.event_klass} but got {klass}")

        for subclass in klass.__subclasses__():
            yield from self._iterate_subclasses(subclass)

        yield klass
