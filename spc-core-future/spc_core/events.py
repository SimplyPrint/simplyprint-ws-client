"""
Event System - Simple Message Bus

Clean, type-safe event routing without the complexity.
"""

__all__ = ["MessageBus", "handles", "emits", "Event"]

from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass
import asyncio
import logging
import inspect

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Base event class.

    Usage:
        @dataclass
        class TemperatureChanged(Event):
            temp: float
            tool: int = 0
    """
    pass


def handles(*event_types: Type[Event]):
    """
    Decorator: Declare which events this component handles.

    Auto-discovers handler methods using naming convention:
    - handle_{event_name}
    - on_{event_name}

    Usage:
        @handles(TemperatureChanged, StatusChanged)
        class Printer:
            def handle_temperature_changed(self, event: TemperatureChanged):
                print(f"Temp: {event.temp}")

            def handle_status_changed(self, event: StatusChanged):
                print(f"Status: {event.status}")
    """
    def decorator(cls):
        if not hasattr(cls, "__handles__"):
            cls.__handles__ = {}

        for event_type in event_types:
            # Try naming patterns
            event_name = _event_name(event_type)
            patterns = [
                f"handle_{event_name}",
                f"on_{event_name}",
            ]

            for pattern in patterns:
                if hasattr(cls, pattern):
                    cls.__handles__[event_type] = pattern
                    break
            else:
                logger.warning(
                    f"{cls.__name__}: No handler found for {event_type.__name__}. "
                    f"Expected: {patterns}"
                )

        return cls
    return decorator


def emits(*event_types: Type[Event]):
    """
    Decorator: Declare which events this component emits.

    Mainly for documentation and validation.

    Usage:
        @emits(TemperatureChanged, StatusChanged)
        class TemperatureSensor:
            async def read(self):
                # ... emit events
                pass
    """
    def decorator(cls):
        cls.__emits__ = tuple(event_types)
        return cls
    return decorator


class MessageBus:
    """
    Simple message bus for event-driven communication.

    Features:
    - Type-safe event routing
    - Async and sync handlers
    - Error isolation
    - Easy testing

    Usage:
        bus = MessageBus()

        @handles(TemperatureChanged)
        class Printer:
            def handle_temperature_changed(self, event):
                print(f"Temp: {event.temp}")

        printer = Printer()
        bus.subscribe(printer)

        await bus.publish(TemperatureChanged(temp=200))
    """

    def __init__(self):
        self._handlers: Dict[Type[Event], List[Callable]] = {}
        self._components: List[Any] = []

    def subscribe(self, component: Any):
        """
        Subscribe a component to events.

        Component must use @handles decorator.

        Args:
            component: Component with @handles decorator
        """
        if not hasattr(component, "__handles__"):
            logger.warning(
                f"{component.__class__.__name__} has no @handles decorator. "
                f"Use @handles(EventType) on the class."
            )
            return

        self._components.append(component)

        for event_type, handler_name in component.__handles__.items():
            handler = getattr(component, handler_name)

            if event_type not in self._handlers:
                self._handlers[event_type] = []

            self._handlers[event_type].append(handler)

            logger.debug(
                f"Subscribed {component.__class__.__name__}.{handler_name} "
                f"to {event_type.__name__}"
            )

    def unsubscribe(self, component: Any):
        """Unsubscribe a component from all events"""
        if component not in self._components:
            return

        self._components.remove(component)

        if not hasattr(component, "__handles__"):
            return

        for event_type, handler_name in component.__handles__.items():
            handler = getattr(component, handler_name, None)
            if handler and event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h is not handler
                ]

        logger.debug(f"Unsubscribed {component.__class__.__name__}")

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.

        Args:
            event: Event instance to publish
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug(f"No handlers for {event_type.__name__}")
            return

        logger.debug(f"Publishing {event_type.__name__} to {len(handlers)} handler(s)")

        # Execute handlers
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                else:
                    handler(event)
            except Exception as e:
                logger.error(
                    f"Error in handler {handler.__name__}: {e}",
                    exc_info=True
                )

        # Wait for async handlers
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Handler raised: {result}", exc_info=result)

    def publish_sync(self, event: Event):
        """
        Publish event synchronously (only calls sync handlers).

        Useful for non-async contexts.
        """
        event_type = type(event)
        handlers = [
            h for h in self._handlers.get(event_type, [])
            if not asyncio.iscoroutinefunction(h)
        ]

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in handler {handler.__name__}: {e}",
                    exc_info=True
                )

    def clear(self):
        """Clear all subscriptions"""
        self._handlers.clear()
        self._components.clear()


def _event_name(event_type: Type[Event]) -> str:
    """Convert EventType to snake_case"""
    import re
    name = event_type.__name__
    # Remove 'Event' suffix if present
    if name.endswith("Event"):
        name = name[:-5]
    # Convert to snake_case
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()
