# Core API Usage Guide

Quick start guide for using the new Core API features.

## Overview

The Core API introduces several new patterns to make building printer integrations easier:

1. **Dependency Injection** - `Depends()` for automatic wiring
2. **Event-Driven Architecture** - `@inbox` and `@outbox` for type-safe events
3. **Scheduling** - `@tick` for periodic tasks
4. **Transactions** - Context managers for atomic state changes

## Quick Start

### 1. Basic Dependency Injection

```python
from simplyprint_ws_client.core.di import Depends, DIContainer

class HTTPClient:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

class MyService:
    # Dependency automatically injected
    http: HTTPClient = Depends()

# Setup container
container = DIContainer()
container.register(HTTPClient)
container.register(MyService)

# Resolve (builds entire dependency graph)
service = await container.resolve(MyService)
```

### 2. Event-Driven Protocols

```python
from simplyprint_ws_client.core.protocols import inbox, outbox, EventRouter

# Define events
@dataclass
class TemperatureEvent:
    temp: float

# Define protocol that EMITS events
@outbox(TemperatureEvent)
class SensorProtocol:
    event_router: EventRouter = Depends()

    async def read_sensor(self):
        temp = await self.sensor.read()
        await self.event_router.emit(TemperatureEvent(temp=temp))

# Define component that RECEIVES events
@inbox(TemperatureEvent)
class Printer:
    async def on_temperature(self, event: TemperatureEvent):
        print(f"Temperature: {event.temp}")

# Wire everything together
router = EventRouter()
sensor = SensorProtocol()
printer = Printer()

router.register(sensor)
router.register(printer)
```

### 3. Scheduled Tasks

```python
from simplyprint_ws_client.core.decorators import tick, SchedulerMixin

class MyPrinter(SchedulerMixin):
    async def init(self):
        await self.start_scheduled_tasks()

    @tick(5)  # Every 5 seconds
    async def poll_temperature(self):
        print("Polling...")

    @tick(60, immediate=True)  # Every minute, run immediately
    async def send_heartbeat(self):
        print("Heartbeat")

    async def halt(self):
        await self.stop_scheduled_tasks()
```

### 4. Transactions

```python
from simplyprint_ws_client.core.transactions import Transaction

# Automatic rollback on error
try:
    with Transaction(printer.state):
        printer.state.temperature = 500  # Invalid
        raise ValueError("Oops")
except ValueError:
    pass

# State automatically rolled back
print(printer.state.temperature)  # Still the old value
```

## Complete Example

See `examples/core_api_demo.py` for a complete working example that demonstrates:

- Multiple protocols (HTTP polling + MQTT)
- Event routing between components
- Dependency injection
- Scheduled tasks
- Lifecycle management

Run it with:

```bash
python -m examples.core_api_demo
```

## Building a Printer Integration

### Step 1: Define Your Config

```python
from simplyprint_ws_client.core.config import PrinterConfig

class MyPrinterConfig(PrinterConfig):
    host: str
    port: int = 80
    api_key: str
```

### Step 2: Define Your Events

```python
from dataclasses import dataclass

@dataclass
class TemperatureChangedEvent:
    tool: int
    actual: float
    target: float

@dataclass
class StatusChangedEvent:
    status: str
```

### Step 3: Define Your Protocol(s)

```python
from simplyprint_ws_client.core.protocols import outbox, Protocol
from simplyprint_ws_client.core.di import Depends

@outbox(TemperatureChangedEvent, StatusChangedEvent)
class MyProtocol(Protocol):
    http: HTTPClient = Depends()
    config: MyPrinterConfig = Depends()
    event_router: EventRouter = Depends()

    async def start(self):
        # Connect to printer
        await self.http.connect(self.config.host)

        # Start polling or listening
        while True:
            data = await self.http.get("/status")
            await self.event_router.emit(
                TemperatureChangedEvent(
                    tool=0,
                    actual=data["temp"],
                    target=data["target"]
                )
            )
            await asyncio.sleep(1)

    async def stop(self):
        # Cleanup
        pass
```

### Step 4: Define Your Printer

```python
from simplyprint_ws_client.core.protocols import inbox

@inbox(TemperatureChangedEvent, StatusChangedEvent)
class MyPrinter:
    protocol: MyProtocol = Depends()
    config: MyPrinterConfig = Depends()

    async def start(self):
        await self.protocol.start()

    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        # Update state
        print(f"Temperature: {event.actual} / {event.target}")

    async def on_status_changed(self, event: StatusChangedEvent):
        # Update state
        print(f"Status: {event.status}")
```

### Step 5: Wire Everything Together

```python
from simplyprint_ws_client.core.di import DIContainer
from simplyprint_ws_client.core.protocols import EventRouter

async def main():
    # Create container
    container = DIContainer()

    # Register dependencies
    config = MyPrinterConfig(host="192.168.1.100", api_key="secret")
    container.register_instance(MyPrinterConfig, config)
    container.register(HTTPClient)
    container.register(EventRouter)
    container.register(MyProtocol)
    container.register(MyPrinter)

    # Resolve printer
    printer = await container.resolve(MyPrinter)
    event_router = await container.resolve(EventRouter)

    # Register with event router
    event_router.register(printer.protocol)
    event_router.register(printer)

    # Start
    await printer.start()

asyncio.run(main())
```

## Advanced Patterns

### Composing Multiple Protocols

A printer can use multiple protocols simultaneously:

```python
@inbox(TemperatureChangedEvent, StatusChangedEvent)
class MultiProtocolPrinter:
    # Real-time updates via WebSocket
    ws_protocol: WebSocketProtocol = Depends()

    # Fallback polling via HTTP
    http_protocol: HTTPPollingProtocol = Depends()

    async def start(self):
        # Start both protocols
        await asyncio.gather(
            self.ws_protocol.start(),
            self.http_protocol.start(),
        )

    # Same handler receives events from BOTH protocols
    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        # Don't care which protocol it came from
        self.state.temperature = event.actual
```

### Conditional Dependencies

```python
class ConditionalPrinter:
    def __init__(self, config: MyConfig = Depends()):
        # Choose protocol based on config
        if config.use_websocket:
            self.protocol: Protocol = Depends(WebSocketProtocol)
        else:
            self.protocol: Protocol = Depends(HTTPPollingProtocol)
```

### Transient Dependencies

By default, `Depends()` creates singletons. Use `scope="transient"` for new instances:

```python
class MyService:
    # New logger instance every time
    logger: Logger = Depends(lambda: logging.getLogger(__name__), scope="transient")
```

### Custom Factories

```python
def create_http_client():
    return HTTPClient(timeout=60, retries=3)

class MyService:
    http: HTTPClient = Depends(create_http_client)
```

## Testing

### Mocking Dependencies

```python
class MockHTTPClient:
    async def get(self, path):
        return {"temp": 25.0}

# Override in tests
container = DIContainer()
service = await container.resolve(
    MyService,
    HTTPClient=MockHTTPClient()  # Override
)
```

### Testing Event Handlers

```python
# Create event router
router = EventRouter()

# Register component
printer = MyPrinter()
router.register(printer)

# Emit test event
await router.emit(TemperatureChangedEvent(tool=0, actual=200, target=220))

# Assert state changed
assert printer.state.temperature == 200
```

## Best Practices

### 1. Keep Protocols Simple

Protocols should only:
- Connect to external systems
- Parse/transform data
- Emit typed events

Don't put business logic in protocols.

### 2. Use Type Hints

Always use type hints for better IDE support and type checking:

```python
async def on_temperature(self, event: TemperatureChangedEvent) -> None:
    ...
```

### 3. Validate Events

Use Pydantic for complex events with validation:

```python
from pydantic import BaseModel, Field

class TemperatureEvent(BaseModel):
    temp: float = Field(ge=-273.15, le=500.0)  # Validate range
```

### 4. Handle Errors Gracefully

Event handlers should not crash:

```python
async def on_temperature(self, event: TemperatureChangedEvent):
    try:
        await self.update_display(event.temp)
    except Exception as e:
        logger.error(f"Failed to update display: {e}")
        # Continue processing other events
```

### 5. Use Transactions for Complex Updates

```python
async def start_print(self, file_data):
    async with AsyncTransaction(self.state):
        self.state.status = "printing"
        self.state.job_id = file_data.id
        self.state.progress = 0.0
        # If any error, all changes rolled back
```

## Migration Guide

### From Old Client to New API

**Before:**
```python
class OldClient(DefaultClient):
    @configure(ServerMsgType.TEMPERATURE)
    async def on_temp(self, msg):
        self.printer.temperature = msg.data.temp
```

**After:**
```python
@inbox(TemperatureChangedEvent)
class NewClient:
    protocol: MyProtocol = Depends()

    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        self.state.temperature = event.actual
```

### Key Differences

| Old API | New API |
|---------|---------|
| `@configure` | `@inbox` / `@outbox` |
| Manual event bus access | Automatic routing |
| Inheritance-based | Composition-based |
| String-based events | Type-safe events |
| No DI | `Depends()` |

## Troubleshooting

### "Circular dependency detected"

Check your dependency graph. Example issue:

```python
class A:
    b: B = Depends()

class B:
    a: A = Depends()  # Circular!
```

**Solution:** Use a shared event bus or introduce an interface.

### "No handler found for Event"

The `@inbox` decorator couldn't find a handler method. Check naming:

```python
@inbox(TemperatureChangedEvent)
class MyPrinter:
    # Must be named: on_temperature_changed
    async def on_temperature_changed(self, event): ...
```

### "Dependency not registered"

Register all dependencies with the container:

```python
container.register(HTTPClient)
container.register(MQTTClient)
container.register(MyProtocol)
```

### Events not received

Make sure to register components with the event router:

```python
router = EventRouter()
router.register(my_protocol)
router.register(my_printer)
```

## API Reference

See `docs/CORE_API_DESIGN.md` for complete architecture documentation.

### Core Modules

- `simplyprint_ws_client.core.di` - Dependency injection
- `simplyprint_ws_client.core.protocols` - Event routing
- `simplyprint_ws_client.core.decorators` - Scheduling
- `simplyprint_ws_client.core.transactions` - State transactions

### Key Classes

- `DIContainer` - Dependency injection container
- `EventRouter` - Event routing engine
- `SchedulerMixin` - Scheduled task support
- `Transaction` - Transactional state changes
- `Protocol` - Base protocol class

### Decorators

- `@inbox(*events)` - Declare incoming events
- `@outbox(*events)` - Declare outgoing events
- `@tick(interval)` - Schedule periodic execution
- `@on_init` - Run on initialization
- `@on_destroy` - Run on destruction

## Examples

See `examples/` directory for complete examples:

- `core_api_demo.py` - Complete working demo
- (More to come...)

## Support

For questions or issues:
1. Check `docs/CORE_API_DESIGN.md`
2. Run the demo: `python -m examples.core_api_demo`
3. Open an issue on GitHub
