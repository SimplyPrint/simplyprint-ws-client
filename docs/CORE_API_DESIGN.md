# SimplyPrint Client Core API Design

**Version:** 1.0
**Status:** Design Proposal
**Date:** 2025-11-05

## Executive Summary

This document proposes a comprehensive redesign of the SimplyPrint client core API to achieve:
- **Async-by-default** architecture with no blocking calls
- **Protocol composition** through event-driven interfaces
- **Type-safe dependency injection** using `Depends()` pattern
- **Declarative event routing** with inbox/outbox contracts
- **Flexible scheduling** with decorators like `@tick(n)`
- **Transaction semantics** for complex state operations

The design builds on the existing architecture (EventBus, autowiring, state models) while introducing new patterns for composability.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current State Analysis](#current-state-analysis)
3. [Core Abstractions](#core-abstractions)
4. [Dependency Injection System](#dependency-injection-system)
5. [Event System Enhancements](#event-system-enhancements)
6. [Protocol Composition](#protocol-composition)
7. [Scheduling System](#scheduling-system)
8. [Transaction Management](#transaction-management)
9. [Implementation Examples](#implementation-examples)
10. [Migration Strategy](#migration-strategy)
11. [Design Decisions](#design-decisions)

---

## Architecture Overview

### High-Level Principles

1. **Separation of Concerns**: Protocols handle I/O, clients handle business logic
2. **Composability**: Mix multiple protocols (MQTT, WebSocket, HTTP) via event buses
3. **Type Safety**: Leverage Python's type system for event routing and DI
4. **No Shared State**: Communication only through typed events
5. **Declarative Configuration**: Use decorators and type hints over imperative code

### System Layers

```
┌─────────────────────────────────────────────────────┐
│              Application Layer                       │
│  (Printer Implementations: BambuPrinter, etc.)      │
└───────────────┬─────────────────────────────────────┘
                │ Events (typed)
┌───────────────▼─────────────────────────────────────┐
│           Protocol Layer                             │
│  (MQTT, WebSocket, HTTP Polling, Direct IPC)        │
└───────────────┬─────────────────────────────────────┘
                │ Events (typed)
┌───────────────▼─────────────────────────────────────┐
│         Infrastructure Layer                         │
│  (EventBus, Scheduler, DI Container, State)         │
└─────────────────────────────────────────────────────┘
```

---

## Current State Analysis

### What Exists (✓)

- **Event Bus** (`events/event_bus.py`): Full-featured with middleware, predicates
- **Autowiring** (`core/autowire.py`): Function-based event mapping via decorators
- **Client Base** (`core/client.py`): Lifecycle methods, state machine
- **Scheduler** (`core/scheduler.py`): Multi-client management with tick-based scheduling
- **State Models** (`core/state/`): Pydantic-based with change detection
- **Async Primitives** (`shared/asyncio/`): ContinuousTask, event loop providers
- **Config System** (`core/config/`): Persistence abstraction

### What's Needed (✗)

- **Depends() Pattern**: FastAPI-style dependency injection
- **EventEmitter/Listener Interfaces**: Clear inbox/outbox contracts
- **@tick(n) Decorator**: Scheduled method execution
- **Protocol Composition**: Mix protocols declaratively
- **Transaction Contexts**: `with printer.job:` style operations
- **Automatic Event Wiring**: Based on type introspection
- **Better Type Safety**: Typed event unions for better IDE support

---

## Core Abstractions

### 1. Config (Already Exists)

```python
from simplyprint_ws_client.core.config import PrinterConfig

class OctoPrintConfig(PrinterConfig):
    """Configuration for OctoPrint integration"""
    host: str
    port: int
    api_key: str
    use_websocket: bool = True
```

### 2. EventEmitter & EventListener (New)

```python
from typing import Protocol, TypeVar, Union, Literal
from abc import abstractmethod

TEvent = TypeVar("TEvent")

class EventEmitter(Protocol[TEvent]):
    """
    Declares what events this component emits.
    Events flow OUT of this component.
    """
    __outbox__: tuple[type[TEvent], ...]

    @abstractmethod
    async def emit(self, event: TEvent) -> None:
        """Emit an event to the bus"""
        ...

class EventListener(Protocol[TEvent]):
    """
    Declares what events this component listens to.
    Events flow IN to this component.
    """
    __inbox__: dict[type[TEvent], str]  # Event type -> handler method name
```

### 3. Printer Interface (Enhanced)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TConfig = TypeVar("TConfig", bound=PrinterConfig)

class Printer(ABC, Generic[TConfig]):
    """Base printer abstraction - implements business logic"""

    config: TConfig
    state: PrinterState

    # Lifecycle hooks
    async def init(self) -> None: ...
    async def tick(self, delta: timedelta) -> None: ...
    async def halt(self) -> None: ...
    async def teardown(self) -> None: ...
```

### 4. Camera Interface (Already Exists)

```python
# Already in shared/camera/base.py
class BaseCameraProtocol(ABC):
    polling_mode: CameraProtocolPollingMode
    is_async: bool

    @abstractmethod
    def read(self) -> Iterator[bytes]: ...
```

---

## Dependency Injection System

### The `Depends()` Pattern

Inspired by FastAPI, we introduce a dependency injection system that:
1. Analyzes type annotations
2. Resolves dependencies recursively
3. Manages lifecycle (singleton, per-request, transient)
4. Supports both sync and async initialization

### Implementation

```python
# File: simplyprint_ws_client/core/di.py

from typing import TypeVar, Generic, Callable, Any, Optional
from dataclasses import dataclass, field
import inspect
import asyncio

T = TypeVar("T")

@dataclass
class Dependency(Generic[T]):
    """Marker for dependency injection"""
    factory: Optional[Callable[..., T]] = None
    scope: Literal["singleton", "transient"] = "singleton"

    def __call__(self) -> T:
        # This is never actually called - it's for type hints only
        raise NotImplementedError("Depends should be resolved by DIContainer")

def Depends(
    dependency: Optional[Callable[..., T]] = None,
    *,
    scope: Literal["singleton", "transient"] = "singleton"
) -> T:
    """
    Declare a dependency to be injected.

    Usage:
        class MyProtocol:
            http: HTTPClient = Depends(HTTPClient)
            config: MyConfig = Depends()  # Infer from type
    """
    return Dependency(factory=dependency, scope=scope)


class DIContainer:
    """Dependency injection container"""

    def __init__(self):
        self._singletons: dict[type, Any] = {}
        self._factories: dict[type, Callable] = {}
        self._resolving: set[type] = set()  # Circular dependency detection

    def register(self, type_: type, factory: Optional[Callable] = None):
        """Register a type with its factory"""
        self._factories[type_] = factory or type_

    async def resolve(self, type_: type, **overrides) -> Any:
        """Resolve a dependency (async)"""
        # Check for override
        if type_ in overrides:
            return overrides[type_]

        # Check singleton cache
        if type_ in self._singletons:
            return self._singletons[type_]

        # Circular dependency check
        if type_ in self._resolving:
            raise RuntimeError(f"Circular dependency detected: {type_}")

        self._resolving.add(type_)

        try:
            # Get factory
            factory = self._factories.get(type_, type_)

            # Introspect factory signature
            sig = inspect.signature(factory)
            kwargs = {}

            for param_name, param in sig.parameters.items():
                if param.annotation == inspect.Parameter.empty:
                    continue

                # Check if default is a Dependency
                if isinstance(param.default, Dependency):
                    dep_type = param.annotation
                    dep_factory = param.default.factory or dep_type

                    # Resolve recursively
                    if param.default.scope == "singleton":
                        kwargs[param_name] = await self.resolve(dep_type, **overrides)
                    else:
                        kwargs[param_name] = await self._create_instance(dep_factory, **overrides)

            # Create instance
            instance = await self._create_instance(factory, **kwargs)

            # Cache if singleton
            self._singletons[type_] = instance

            return instance
        finally:
            self._resolving.discard(type_)

    async def _create_instance(self, factory: Callable, **kwargs) -> Any:
        """Create instance handling both sync and async factories"""
        if inspect.iscoroutinefunction(factory):
            return await factory(**kwargs)
        else:
            result = factory(**kwargs)
            if inspect.iscoroutine(result):
                return await result
            return result


# Global container
_container = DIContainer()

def get_container() -> DIContainer:
    return _container
```

### Usage Example

```python
class HTTPClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._session = httpx.AsyncClient(timeout=timeout)

    async def get(self, url: str) -> httpx.Response:
        return await self._session.get(url)

class MQTTClient:
    def __init__(self, host: str = "localhost", port: int = 1883):
        self.client = paho.mqtt.client.Client()

    async def connect(self, host: str, port: int):
        ...

class BambuMQTTProtocol:
    """Protocol layer - handles MQTT communication"""
    mqtt: MQTTClient = Depends()
    config: BambuConfig = Depends()

    def __init__(self):
        # Dependencies are injected before __init__ or via __post_init__
        pass

    async def start(self):
        await self.mqtt.connect(self.config.host, self.config.port)

class BambuPrinter(Printer):
    """Application layer - handles business logic"""
    protocol: BambuMQTTProtocol = Depends()

    async def init(self):
        await self.protocol.start()
```

---

## Event System Enhancements

### 1. Typed Event Unions

```python
# File: simplyprint_ws_client/events/typed_events.py

from typing import Union, Literal
from pydantic import BaseModel

# Define events as Pydantic models for validation
class TemperatureChangedEvent(BaseModel):
    tool: int
    actual: float
    target: float

class StatusChangedEvent(BaseModel):
    old_status: PrinterStatus
    new_status: PrinterStatus

class JobProgressEvent(BaseModel):
    progress: float  # 0.0 to 100.0
    time_remaining: Optional[int]

# Union type for all printer events
PrinterEvent = Union[
    TemperatureChangedEvent,
    StatusChangedEvent,
    JobProgressEvent,
]
```

### 2. Inbox/Outbox Declaration

```python
# File: simplyprint_ws_client/core/protocols.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class HasEventBus(Protocol):
    """Marker for components that have an event bus"""
    __inbox__: dict[type, str]  # Event type -> handler method
    __outbox__: tuple[type, ...]  # Events this component emits


def inbox(*event_types: type):
    """Decorator to declare inbox events"""
    def decorator(cls):
        if not hasattr(cls, "__inbox__"):
            cls.__inbox__ = {}

        # Auto-discover handlers
        for event_type in event_types:
            handler_name = f"on_{event_type.__name__.lower().replace('event', '')}"
            if hasattr(cls, handler_name):
                cls.__inbox__[event_type] = handler_name

        return cls
    return decorator


def outbox(*event_types: type):
    """Decorator to declare outbox events"""
    def decorator(cls):
        cls.__outbox__ = event_types
        return cls
    return decorator


# Usage
@inbox(TemperatureChangedEvent, StatusChangedEvent)
@outbox(GcodeCommandEvent, PrinterErrorEvent)
class OctoPrintProtocol:
    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        # Handle temperature update
        ...
```

### 3. Automatic Event Bus Wiring

```python
# File: simplyprint_ws_client/core/event_router.py

class EventRouter:
    """Automatically wire event buses based on inbox/outbox"""

    def __init__(self):
        self._components: list[Any] = []
        self._event_map: dict[type, list[Callable]] = {}

    def register(self, component: Any):
        """Register a component for event routing"""
        self._components.append(component)

        # Wire inbox
        if hasattr(component, "__inbox__"):
            for event_type, handler_name in component.__inbox__.items():
                handler = getattr(component, handler_name)
                if event_type not in self._event_map:
                    self._event_map[event_type] = []
                self._event_map[event_type].append(handler)

    async def emit(self, event: Any):
        """Route event to all registered handlers"""
        event_type = type(event)
        handlers = self._event_map.get(event_type, [])

        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
```

---

## Protocol Composition

### Design Pattern

Protocols are composable units that:
1. Consume events from external sources (MQTT, WebSocket, HTTP)
2. Transform them into typed internal events
3. Emit events to the application layer

```python
# File: examples/protocols/mqtt_protocol.py

from typing import Protocol as TypingProtocol

class MQTTMessageProtocol(TypingProtocol):
    """Protocol for handling MQTT messages"""

    @abstractmethod
    async def on_mqtt_message(self, message: paho.mqtt.MQTTMessage):
        """Handle incoming MQTT message"""
        ...


@inbox()  # This protocol doesn't listen to internal events
@outbox(TemperatureChangedEvent, StatusChangedEvent)
class BambuMQTTProtocol(EventEmitter, MQTTMessageProtocol):
    """Bambu Lab MQTT protocol implementation"""

    mqtt: MQTTClient = Depends()
    event_bus: EventBus = Depends()

    async def start(self):
        """Start MQTT client and subscribe"""
        await self.mqtt.connect()
        await self.mqtt.subscribe("device/#", self.on_mqtt_message)

    async def on_mqtt_message(self, message: paho.mqtt.MQTTMessage):
        """Parse MQTT message and emit typed events"""
        data = json.loads(message.payload)

        # Parse Bambu-specific protocol
        if "temp" in data:
            await self.event_bus.emit(TemperatureChangedEvent(
                tool=data.get("nozzle_id", 0),
                actual=data["temp"]["current"],
                target=data["temp"]["target"],
            ))
```

### Composing Multiple Protocols

```python
# File: examples/printers/octoprint_printer.py

@inbox(TemperatureChangedEvent, StatusChangedEvent, JobProgressEvent)
@outbox(GcodeCommandEvent, FileUploadEvent)
class OctoPrintPrinter(Printer):
    """OctoPrint printer implementation"""

    # Compose multiple protocols
    ws_protocol: OctoPrintWebSocketProtocol = Depends()
    http_protocol: OctoPrintHTTPPollProtocol = Depends()
    simplyprint_ws: SimplyPrintWebSocketProtocol = Depends()

    config: OctoPrintConfig = Depends()

    async def init(self):
        # All protocols run in parallel
        await asyncio.gather(
            self.ws_protocol.start(),
            self.http_protocol.start(),
            self.simplyprint_ws.start(),
        )

    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        """Handle temperature from OctoPrint, forward to SimplyPrint"""
        self.state.tool(event.tool).temperature.actual = event.actual
        self.state.tool(event.tool).temperature.target = event.target


# Protocol implementations
@outbox(TemperatureChangedEvent)
class OctoPrintWebSocketProtocol:
    ws: WebSocketClient = Depends()
    config: OctoPrintConfig = Depends()
    event_bus: EventBus = Depends()

    async def start(self):
        await self.ws.connect(f"ws://{self.config.host}/sockjs/websocket")
        async for message in self.ws:
            await self.on_ws_message(message)

    async def on_ws_message(self, message: str):
        data = json.loads(message)
        if "temps" in data:
            # Emit typed event
            ...


@outbox(StatusChangedEvent)
class OctoPrintHTTPPollProtocol:
    http: HTTPClient = Depends()
    config: OctoPrintConfig = Depends()
    event_bus: EventBus = Depends()

    @tick(5)  # Poll every 5 seconds
    async def poll_status(self):
        response = await self.http.get(
            f"http://{self.config.host}/api/printer",
            headers={"X-Api-Key": self.config.api_key}
        )
        data = response.json()
        # Emit status event
        ...
```

---

## Scheduling System

### The `@tick` Decorator

```python
# File: simplyprint_ws_client/core/decorators.py

from typing import Callable, Union
from functools import wraps
import asyncio

def tick(
    interval: Union[int, float],
    *,
    immediate: bool = False,
    name: Optional[str] = None
):
    """
    Schedule a method to run every `interval` seconds.

    Args:
        interval: Seconds between executions
        immediate: Run immediately on start (default: False)
        name: Optional name for the scheduled task

    Usage:
        @tick(5)
        async def poll_status(self):
            # Runs every 5 seconds
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Mark function for scheduler
        wrapper._is_scheduled = True
        wrapper._schedule_interval = interval
        wrapper._schedule_immediate = immediate
        wrapper._schedule_name = name or func.__name__

        return wrapper
    return decorator


class SchedulerMixin:
    """Mixin to add scheduling capabilities"""

    def __init__(self):
        self._scheduled_tasks: dict[str, asyncio.Task] = {}

    async def _start_scheduled_tasks(self):
        """Start all @tick decorated methods"""
        for attr_name in dir(self):
            attr = getattr(self.__class__, attr_name, None)
            if not hasattr(attr, "_is_scheduled"):
                continue

            method = getattr(self, attr_name)
            task = asyncio.create_task(
                self._run_scheduled(method, attr._schedule_interval, attr._schedule_immediate)
            )
            self._scheduled_tasks[attr._schedule_name] = task

    async def _run_scheduled(self, method: Callable, interval: float, immediate: bool):
        """Run a scheduled method"""
        if immediate:
            await method()

        while True:
            await asyncio.sleep(interval)
            try:
                await method()
            except Exception as e:
                # Log error but keep running
                logging.error(f"Error in scheduled task {method.__name__}: {e}")

    async def _stop_scheduled_tasks(self):
        """Stop all scheduled tasks"""
        for task in self._scheduled_tasks.values():
            task.cancel()
        await asyncio.gather(*self._scheduled_tasks.values(), return_exceptions=True)
        self._scheduled_tasks.clear()
```

### Usage

```python
class PrusaLinkPrinter(Printer, SchedulerMixin):
    config: PrusaLinkConfig = Depends()
    http: HTTPClient = Depends()

    async def init(self):
        await self._start_scheduled_tasks()

    @tick(1)
    async def poll_temperature(self):
        """Poll temperature every second"""
        response = await self.http.get(f"{self.config.host}/api/v1/status")
        data = response.json()
        # Update state
        ...

    @tick(5)
    async def poll_job_status(self):
        """Poll job status every 5 seconds"""
        ...

    @tick(60, immediate=True)
    async def send_heartbeat(self):
        """Send heartbeat every minute (immediately on start)"""
        ...

    async def halt(self):
        await self._stop_scheduled_tasks()
```

---

## Transaction Management

### Context Manager for State Changes

```python
# File: simplyprint_ws_client/core/state/transactions.py

from contextlib import asynccontextmanager
from typing import AsyncIterator

class Transaction:
    """Transaction context for state changes"""

    def __init__(self, state: PrinterState):
        self._state = state
        self._snapshot: dict[str, Any] = {}
        self._committed = False

    def __enter__(self):
        # Take snapshot
        self._snapshot = self._state.model_dump()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on exception
            self._state.model_update(self._snapshot)
            return False

        # Commit changes
        self._committed = True
        return True

    def rollback(self):
        """Manually rollback transaction"""
        if not self._committed:
            self._state.model_update(self._snapshot)


class PrinterState(StateModel):
    """Enhanced printer state with transaction support"""

    def transaction(self) -> Transaction:
        """Start a transaction"""
        return Transaction(self)

    @asynccontextmanager
    async def job_context(self) -> AsyncIterator["JobInfo"]:
        """Context manager for job operations"""
        with self.transaction():
            yield self.job_info
            # Validate job state
            if not self.job_info.is_valid():
                raise ValueError("Invalid job state")


# Usage
async def start_print(printer: Printer, file_data: FileDemandData):
    """Start a print with transactional semantics"""

    try:
        async with printer.state.job_context() as job:
            job.started = True
            job.progress = 0.0
            job.file_id = file_data.file_id
            # If any error occurs, changes are rolled back
    except ValueError as e:
        printer.logger.error(f"Failed to start print: {e}")
        return

# Alternative style
def update_temperatures(printer: Printer, temps: dict):
    with printer.state.transaction():
        for tool_id, temp in temps.items():
            printer.state.tool(tool_id).temperature.actual = temp
        # Automatically committed on success
```

---

## Implementation Examples

### Complete Example: Bambu Lab Printer

```python
# File: examples/printers/bambu_printer.py

from simplyprint_ws_client.core import Printer, Depends
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.protocols import inbox, outbox
from simplyprint_ws_client.events.typed_events import *

# 1. Define Config
class BambuConfig(PrinterConfig):
    mqtt_host: str
    mqtt_port: int = 1883
    access_code: str
    serial_number: str


# 2. Define Events
class BambuPushStatusEvent(BaseModel):
    """Event from Bambu MQTT"""
    temps: dict
    status: str
    progress: float


# 3. Define Protocol
@outbox(
    TemperatureChangedEvent,
    StatusChangedEvent,
    JobProgressEvent
)
class BambuMQTTProtocol:
    mqtt: MQTTClient = Depends()
    config: BambuConfig = Depends()
    event_bus: EventBus = Depends()

    async def start(self):
        await self.mqtt.connect(self.config.mqtt_host, self.config.mqtt_port)
        await self.mqtt.subscribe(
            f"device/{self.config.serial_number}/report",
            self.on_mqtt_message
        )

    async def on_mqtt_message(self, message: paho.mqtt.MQTTMessage):
        """Parse Bambu protocol and emit typed events"""
        data = json.loads(message.payload)

        # Emit temperature event
        if "nozzle_temper" in data:
            await self.event_bus.emit(TemperatureChangedEvent(
                tool=0,
                actual=data["nozzle_temper"],
                target=data.get("nozzle_target_temper", 0),
            ))

        # Emit status event
        if "gcode_state" in data:
            await self.event_bus.emit(StatusChangedEvent(
                old_status=self._last_status,
                new_status=self._parse_status(data["gcode_state"]),
            ))

        # Emit progress event
        if "mc_percent" in data:
            await self.event_bus.emit(JobProgressEvent(
                progress=data["mc_percent"],
                time_remaining=data.get("mc_remaining_time"),
            ))


# 4. Define Printer
@inbox(
    TemperatureChangedEvent,
    StatusChangedEvent,
    JobProgressEvent,
    GcodeCommandEvent  # From SimplyPrint
)
class BambuPrinter(Printer[BambuConfig]):
    """Bambu Lab printer implementation"""

    protocol: BambuMQTTProtocol = Depends()
    simplyprint: SimplyPrintWebSocketProtocol = Depends()

    async def init(self):
        """Initialize printer"""
        await asyncio.gather(
            self.protocol.start(),
            self.simplyprint.start(),
        )

    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        """Update state from protocol"""
        self.state.tool(event.tool).temperature.actual = event.actual
        self.state.tool(event.tool).temperature.target = event.target

    async def on_status_changed(self, event: StatusChangedEvent):
        """Update printer status"""
        self.state.status = event.new_status

    async def on_job_progress(self, event: JobProgressEvent):
        """Update job progress"""
        async with self.state.job_context() as job:
            job.progress = event.progress
            job.time_remaining = event.time_remaining

    async def on_gcode_command(self, event: GcodeCommandEvent):
        """Execute gcode from SimplyPrint"""
        await self.protocol.mqtt.publish(
            f"device/{self.config.serial_number}/request",
            json.dumps({"gcode": event.commands})
        )
```

### Example: Multi-Protocol Printer (OctoPrint)

```python
# File: examples/printers/octoprint_printer.py

class OctoPrintConfig(PrinterConfig):
    host: str
    port: int = 80
    api_key: str
    use_websocket: bool = True


@outbox(TemperatureChangedEvent)
class OctoPrintWebSocketProtocol:
    """Real-time updates via WebSocket"""
    ws: WebSocketClient = Depends()
    event_bus: EventBus = Depends()

    async def start(self):
        await self.ws.connect(f"ws://{self.config.host}/sockjs/websocket")
        async for message in self.ws:
            await self._handle_message(message)


@outbox(StatusChangedEvent)
class OctoPrintHTTPProtocol:
    """Fallback polling via HTTP"""
    http: HTTPClient = Depends()
    event_bus: EventBus = Depends()

    @tick(5)
    async def poll_status(self):
        response = await self.http.get(f"http://{self.config.host}/api/printer")
        # Emit events
        ...


@inbox(TemperatureChangedEvent, StatusChangedEvent, GcodeCommandEvent)
class OctoPrintPrinter(Printer[OctoPrintConfig]):
    """OctoPrint printer with multiple protocols"""

    # Both protocols can emit events
    ws_protocol: OctoPrintWebSocketProtocol = Depends()
    http_protocol: OctoPrintHTTPProtocol = Depends()

    # Connect to SimplyPrint
    simplyprint: SimplyPrintWebSocketProtocol = Depends()

    async def init(self):
        tasks = [self.simplyprint.start()]

        if self.config.use_websocket:
            tasks.append(self.ws_protocol.start())
        else:
            tasks.append(self.http_protocol.start())

        await asyncio.gather(*tasks)
```

### Example: Prusa Direct Protocol (Custom IPC)

```python
# File: examples/printers/prusa_direct_printer.py

class PrusaDirectConfig(PrinterConfig):
    ipc_socket: str = "/var/run/prusa-link.sock"


@outbox(TemperatureChangedEvent, StatusChangedEvent, FilamentChangeEvent)
class PrusaDirectProtocol:
    """Direct IPC with Prusa firmware"""
    redis: RedisClient = Depends()
    event_bus: EventBus = Depends()

    async def start(self):
        # Subscribe to Redis pubsub
        async for message in self.redis.subscribe("prusa:events"):
            await self._handle_event(message)

    async def _handle_event(self, message: dict):
        event_type = message["type"]

        if event_type == "TEMP_UPDATE":
            await self.event_bus.emit(TemperatureChangedEvent(**message["data"]))


@inbox(TemperatureChangedEvent, StatusChangedEvent, FilamentChangeEvent)
class PrusaDirectPrinter(Printer[PrusaDirectConfig]):
    protocol: PrusaDirectProtocol = Depends()

    async def on_filament_change(self, event: FilamentChangeEvent):
        """Handle Prusa-specific event"""
        if event.action == "inserted":
            # Detect filament type
            ...
```

---

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)

1. **Implement DI system** (`core/di.py`)
   - `Depends()` marker
   - `DIContainer` with recursive resolution
   - Lifecycle management

2. **Add inbox/outbox decorators** (`core/protocols.py`)
   - `@inbox` and `@outbox` decorators
   - Protocol interfaces
   - Type checking utilities

3. **Implement EventRouter** (`core/event_router.py`)
   - Automatic wiring based on inbox/outbox
   - Event type validation

### Phase 2: Scheduling (Week 3)

4. **Implement @tick decorator** (`core/decorators.py`)
   - `SchedulerMixin` class
   - Integration with existing scheduler
   - Task lifecycle management

### Phase 3: Transactions (Week 4)

5. **Add transaction support** (`core/state/transactions.py`)
   - `Transaction` context manager
   - State snapshots and rollback
   - Validation hooks

### Phase 4: Examples & Migration (Weeks 5-6)

6. **Create example implementations**
   - Virtual printer (already exists, enhance it)
   - Bambu Lab printer (new)
   - OctoPrint printer (new)

7. **Migrate existing clients**
   - Update tutorial client
   - Update virtual client
   - Documentation

### Phase 5: Testing & Polish (Week 7)

8. **Comprehensive testing**
   - Unit tests for DI system
   - Integration tests for protocols
   - Performance testing

9. **Documentation**
   - API reference
   - Migration guide
   - Best practices

---

## Design Decisions

### 1. Why Depends() over manual DI?

**Pros:**
- Familiar pattern (FastAPI users)
- Type-safe with IDE support
- Reduces boilerplate
- Testable (easy to override dependencies)

**Cons:**
- Magic behavior (implicit resolution)
- Requires introspection
- Learning curve

**Decision:** Use `Depends()` but keep it explicit. No auto-discovery of dependencies without type hints.

### 2. Why Pydantic for events?

**Pros:**
- Validation built-in
- JSON serialization
- Type safety
- Already used in the codebase

**Cons:**
- Performance overhead
- Complexity for simple events

**Decision:** Use Pydantic for complex events, allow plain dataclasses for simple ones.

### 3. Why inbox/outbox decorators?

**Pros:**
- Clear contract of what events flow through a component
- Enables automatic wiring
- Self-documenting code

**Cons:**
- Decorator complexity
- Runtime introspection overhead

**Decision:** Make decorators optional. Manually wired event handlers still work.

### 4. Event ordering guarantees?

**Options:**
- **Fully ordered:** Single queue, sequential processing (slow)
- **Unordered:** Parallel processing (fast, complex)
- **Partial ordering:** Per-source ordering (balanced)

**Decision:** Default to unordered with parallel processing. Add `@configure(priority=N)` for explicit ordering when needed.

### 5. Backpressure handling?

**Options:**
- **Unbounded queues:** Simple but memory risk
- **Bounded queues with dropping:** Can lose events
- **Bounded queues with backpressure:** Complex but reliable

**Decision:** Use bounded queues (size=1000) with backpressure. Add metrics to monitor queue depth.

### 6. Transaction semantics?

**Options:**
- **Optimistic:** Record changes, rollback on conflict
- **Pessimistic:** Lock during transaction
- **Snapshot:** Copy-on-write

**Decision:** Use snapshot-based transactions. Simple, safe, but higher memory usage for large states.

### 7. Pydantic usage?

**Where to use:**
- ✓ Config objects
- ✓ Complex events with validation
- ✓ API message models (already used)

**Where NOT to use:**
- ✗ Internal state (already has StateModel)
- ✗ Simple events (use dataclasses)
- ✗ High-frequency updates (too slow)

**Decision:** Keep Pydantic for config and messages. StateModel for runtime state.

### 8. Shared state vs. event-only?

**Guideline:**
- **Events:** Cross-component communication
- **Shared state:** Within a component (e.g., `PrinterState` within `Printer`)

**Decision:** No shared state between protocols and printers. Only events.

---

## Open Questions

1. **File storage (local & remote)?**
   - Create a `FileStorageProtocol` abstraction?
   - S3-compatible interface?

2. **Tunneling support?**
   - How to expose printer API over tunnel?
   - WebRTC? WireGuard?

3. **Self-version awareness?**
   - How to handle client upgrades?
   - Rolling deployments?

4. **Testing strategy?**
   - Mock protocols for unit tests?
   - Simulation framework?

5. **Performance targets?**
   - How many printers per instance?
   - Event throughput requirements?

---

## Conclusion

This design provides:
- **Composability** via protocol layering
- **Type safety** via typed events and DI
- **Flexibility** via mix-and-match protocols
- **Maintainability** via clear separation of concerns

The incremental migration strategy allows us to:
- Keep existing code working
- Introduce new patterns gradually
- Validate design decisions with real usage

Next steps:
1. Review and discuss this design
2. Prototype DI system
3. Implement one example printer (Bambu)
4. Iterate based on feedback
