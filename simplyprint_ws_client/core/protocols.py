"""
Protocol System

Defines inbox/outbox contracts for event-driven components.
Enables automatic event routing based on type introspection.
"""

__all__ = [
    "inbox",
    "outbox",
    "EventEmitter",
    "EventListener",
    "Protocol",
    "EventRouter",
]

from typing import Any, Callable, Dict, List, Optional, Protocol as TypingProtocol, Type, get_type_hints
from abc import ABC, abstractmethod
import asyncio
import logging
import inspect

logger = logging.getLogger(__name__)


def inbox(*event_types: Type):
    """
    Decorator to declare which events a component listens to.

    Automatically discovers handler methods following the pattern:
    - on_{event_name} where event_name is the lowercased event class name without 'Event' suffix
    - on_{event_name}_event for explicit naming

    Args:
        event_types: Event types this component handles

    Usage:
        ```python
        @inbox(TemperatureChangedEvent, StatusChangedEvent)
        class MyPrinter:
            async def on_temperature_changed(self, event: TemperatureChangedEvent):
                # Handler automatically wired
                ...

            async def on_status_changed(self, event: StatusChangedEvent):
                # Handler automatically wired
                ...
        ```
    """

    def decorator(cls):
        if not hasattr(cls, "__inbox__"):
            cls.__inbox__ = {}

        # Auto-discover handler methods
        for event_type in event_types:
            # Try multiple naming patterns
            event_name = event_type.__name__
            handler_candidates = [
                f"on_{_to_snake_case(event_name)}",
                f"on_{_to_snake_case(event_name.replace('Event', ''))}",
                f"handle_{_to_snake_case(event_name)}",
            ]

            for handler_name in handler_candidates:
                if hasattr(cls, handler_name):
                    cls.__inbox__[event_type] = handler_name
                    logger.debug(
                        f"Registered inbox handler: {cls.__name__}.{handler_name} -> {event_name}"
                    )
                    break
            else:
                logger.warning(
                    f"No handler found for {event_name} in {cls.__name__}. "
                    f"Tried: {handler_candidates}"
                )

        return cls

    return decorator


def outbox(*event_types: Type):
    """
    Decorator to declare which events a component emits.

    This is primarily for documentation and type checking.
    The EventRouter can validate that emitted events match the declared outbox.

    Args:
        event_types: Event types this component emits

    Usage:
        ```python
        @outbox(TemperatureChangedEvent, StatusChangedEvent)
        class MyProtocol:
            async def emit_temperature(self, temp: float):
                await self.event_bus.emit(TemperatureChangedEvent(temp=temp))
        ```
    """

    def decorator(cls):
        cls.__outbox__ = tuple(event_types)
        logger.debug(f"Registered outbox for {cls.__name__}: {[e.__name__ for e in event_types]}")
        return cls

    return decorator


class EventEmitter(TypingProtocol):
    """
    Protocol for components that emit events.
    This is a typing protocol - no implementation needed.
    """

    __outbox__: tuple[Type, ...]

    @abstractmethod
    async def emit(self, event: Any) -> None:
        """Emit an event to the bus"""
        ...


class EventListener(TypingProtocol):
    """
    Protocol for components that listen to events.
    This is a typing protocol - no implementation needed.
    """

    __inbox__: Dict[Type, str]


class Protocol(ABC):
    """
    Base class for protocol implementations.

    Protocols are responsible for:
    - Communicating with external systems (MQTT, WebSocket, HTTP, etc.)
    - Transforming external data into typed internal events
    - Emitting events to the application layer

    Protocols should NOT contain business logic.
    """

    @abstractmethod
    async def start(self):
        """Start the protocol (connect, subscribe, etc.)"""
        ...

    @abstractmethod
    async def stop(self):
        """Stop the protocol gracefully"""
        ...


class EventRouter:
    """
    Automatically routes events to registered components based on inbox/outbox declarations.

    Features:
    - Type-safe event routing
    - Async and sync handler support
    - Error handling with isolation
    - Optional validation of outbox declarations
    """

    def __init__(self, *, validate_outbox: bool = False):
        """
        Args:
            validate_outbox: If True, validate that emitted events match declared outbox
        """
        self._components: List[Any] = []
        self._event_handlers: Dict[Type, List[Callable]] = {}
        self._event_types: Dict[Type, str] = {}  # event_type -> component name
        self._validate_outbox = validate_outbox

    def register(self, component: Any, *, name: Optional[str] = None):
        """
        Register a component for event routing.

        Args:
            component: Component instance with __inbox__ and/or __outbox__
            name: Optional name for debugging
        """
        component_name = name or component.__class__.__name__
        self._components.append(component)

        # Register inbox handlers
        if hasattr(component, "__inbox__"):
            inbox_dict: Dict[Type, str] = component.__inbox__

            for event_type, handler_name in inbox_dict.items():
                handler = getattr(component, handler_name, None)

                if handler is None:
                    logger.warning(
                        f"Handler {handler_name} not found on {component_name} for {event_type.__name__}"
                    )
                    continue

                if event_type not in self._event_handlers:
                    self._event_handlers[event_type] = []

                self._event_handlers[event_type].append(handler)
                self._event_types[event_type] = component_name

                logger.info(
                    f"Registered event handler: {component_name}.{handler_name} "
                    f"for {event_type.__name__}"
                )

        # Record outbox (for validation)
        if hasattr(component, "__outbox__"):
            outbox_types = component.__outbox__
            logger.info(
                f"Component {component_name} declares outbox: "
                f"{[e.__name__ for e in outbox_types]}"
            )

    async def emit(self, event: Any, *, source: Optional[str] = None):
        """
        Emit an event to all registered handlers.

        Args:
            event: Event instance
            source: Optional source component name (for validation)

        Raises:
            TypeError: If event type is not declared in source's outbox (when validation enabled)
        """
        event_type = type(event)
        event_name = event_type.__name__

        # Validate outbox if enabled
        if self._validate_outbox and source:
            source_component = next(
                (c for c in self._components if c.__class__.__name__ == source),
                None,
            )

            if source_component and hasattr(source_component, "__outbox__"):
                if event_type not in source_component.__outbox__:
                    raise TypeError(
                        f"Event {event_name} not declared in {source}'s outbox. "
                        f"Declared: {[e.__name__ for e in source_component.__outbox__]}"
                    )

        # Find handlers
        handlers = self._event_handlers.get(event_type, [])

        if not handlers:
            logger.debug(f"No handlers registered for {event_name}")
            return

        logger.debug(f"Routing {event_name} to {len(handlers)} handler(s)")

        # Execute all handlers (in parallel for async, sequentially for sync)
        tasks = []

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Async handler - schedule as task
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                else:
                    # Sync handler - call directly
                    handler(event)
            except Exception as e:
                logger.error(
                    f"Error invoking handler {handler.__name__} for {event_name}: {e}",
                    exc_info=True,
                )

        # Wait for all async handlers
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Handler raised exception: {result}", exc_info=result)

    def unregister(self, component: Any):
        """
        Unregister a component from event routing.

        Args:
            component: Component to remove
        """
        self._components = [c for c in self._components if c is not component]

        # Remove handlers
        if hasattr(component, "__inbox__"):
            for event_type, handler_name in component.__inbox__.items():
                handler = getattr(component, handler_name, None)

                if handler and event_type in self._event_handlers:
                    self._event_handlers[event_type] = [
                        h for h in self._event_handlers[event_type] if h is not handler
                    ]

        logger.info(f"Unregistered component {component.__class__.__name__}")

    def clear(self):
        """Clear all registered components and handlers"""
        self._components.clear()
        self._event_handlers.clear()
        self._event_types.clear()


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case"""
    import re

    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters that follow lowercase or digits
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()
