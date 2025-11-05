# SPC Core Future

**Clean, Modern API for Printer Integrations**

This is a complete rewrite of the SimplyPrint client core API with:
- ✅ Cleaner, more conventional naming
- ✅ Simpler dependency injection
- ✅ Type-safe event system
- ✅ Declarative scheduling
- ✅ Atomic transactions
- ✅ Isolated, runnable examples

## Quick Start

```python
from spc_core import Container, MessageBus, handles, Event

# Define an event
@dataclass
class TemperatureChanged(Event):
    temp: float

# Handle events
@handles(TemperatureChanged)
class Printer:
    def handle_temperature_changed(self, event: TemperatureChanged):
        print(f"Temperature: {event.temp}°C")

# Wire it up
bus = MessageBus()
printer = Printer()
bus.subscribe(printer)

# Publish events
await bus.publish(TemperatureChanged(temp=200))
```

## API Overview

### Dependency Injection

**Clean and simple - no magic**

```python
from spc_core import Container, Injected

class Database:
    def __init__(self, host: str = "localhost"):
        self.host = host

class UserRepository:
    def __init__(self, db: Database = Injected):
        self.db = db  # Automatically injected

# Setup
container = Container()
container.register(Database)
container.register(UserRepository)

# Resolve (builds entire dependency graph)
repo = await container.resolve(UserRepository)
```

**Key features:**
- `Injected` marker for clean syntax
- Auto-resolution of dependency trees
- Singleton instances by default
- Lifecycle hooks via `Service` base class
- Custom factories with `Injected.of(factory)`

### Event System

**Type-safe message bus**

```python
from spc_core import MessageBus, handles, emits, Event

# Define events
@dataclass
class StatusChanged(Event):
    status: str

# Declare what you handle
@handles(StatusChanged)
class Display:
    def handle_status_changed(self, event: StatusChanged):
        print(f"Status: {event.status}")

# Declare what you emit
@emits(StatusChanged)
class Sensor:
    async def read(self, bus: MessageBus):
        await bus.publish(StatusChanged(status="active"))

# Wire together
bus = MessageBus()
bus.subscribe(Display())

await sensor.read(bus)
```

**Key features:**
- `@handles` declares incoming events
- `@emits` declares outgoing events (documentation)
- Auto-discovery of handler methods
- Async and sync handlers
- Error isolation
- Easy testing

### Scheduling

**Declarative periodic tasks**

```python
from spc_core import Scheduled, interval, on_startup, on_shutdown

class Monitor(Scheduled):
    async def start(self):
        await self.start_scheduled()

    @on_startup
    async def initialize(self):
        print("Starting...")

    @interval(5)
    async def check_health(self):
        print("Health check")

    @interval(60, run_immediately=True)
    async def heartbeat(self):
        print("Heartbeat")

    @on_shutdown
    async def cleanup(self):
        print("Stopping...")

    async def stop(self):
        await self.stop_scheduled()
```

**Key features:**
- `@interval(seconds)` for periodic execution
- `@on_startup` for initialization
- `@on_shutdown` for cleanup
- Supports `timedelta` for intervals
- Automatic task management
- Graceful error handling

### Transactions

**Safe, atomic state updates**

```python
from spc_core import atomic

class PrinterState:
    def __init__(self):
        self.temperature = 0.0
        self.status = "idle"

state = PrinterState()

# Atomic transaction
try:
    with atomic(state):
        state.temperature = 200
        state.status = "printing"
        # Automatically committed
except Exception:
    # Automatically rolled back
    pass
```

**Key features:**
- Context manager for atomic updates
- Auto-rollback on exception
- Snapshot-based (safe)
- Works with any object
- Async version available

### Protocols

**Base classes for integrations**

```python
from spc_core import Protocol, Printer

class HTTPProtocol(Protocol):
    async def start(self):
        # Connect to printer
        pass

    async def stop(self):
        # Disconnect
        pass

class MyPrinter(Printer):
    protocol: HTTPProtocol = Injected

    async def start(self):
        await self.protocol.start()

    async def stop(self):
        await self.protocol.stop()
```

## Naming Conventions

Compared to the original design, we've improved naming:

| Original | New | Reason |
|----------|-----|--------|
| `Depends()` | `Injected` | Cleaner, more Pythonic |
| `@inbox/@outbox` | `@handles/@emits` | More descriptive |
| `EventRouter` | `MessageBus` | Standard terminology |
| `@tick` | `@interval` | More intuitive |
| `DIContainer` | `Container` | Simpler |
| `Injectable` | `Service` | Conventional |

## Examples

We provide **4 complete, runnable examples**:

### 1. Dependency Injection

```bash
python -m examples.01_dependency_injection
```

Demonstrates:
- Basic dependency injection
- Service lifecycle (start/stop)
- Custom factories
- Dependency graph resolution

### 2. Event System

```bash
python -m examples.02_event_system
```

Demonstrates:
- Simple event handlers
- Multiple event types
- Event emitters
- Async handlers
- Event chains (handler emits events)
- Subscribe/unsubscribe

### 3. Scheduling

```bash
python -m examples.03_scheduling
```

Demonstrates:
- Basic intervals
- Multiple intervals at different rates
- Startup/shutdown hooks
- Immediate execution
- Error handling in scheduled tasks

### 4. Complete Printer Integration

```bash
python -m examples.04_complete_printer
```

**This is the big one!** Shows everything together:
- Dependency injection wiring protocols
- Event-driven communication
- Multiple protocols (HTTP polling + WebSocket)
- State management with transactions
- Scheduled tasks (heartbeat, reporting)
- Full lifecycle management
- Real printer simulation

**Run it to see a complete, working printer implementation!**

## Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│    (Printer implementations)            │
└────────────┬────────────────────────────┘
             │ Events (typed)
┌────────────▼────────────────────────────┐
│         Protocol Layer                  │
│  (HTTP, WebSocket, MQTT, etc.)         │
└────────────┬────────────────────────────┘
             │ Events (typed)
┌────────────▼────────────────────────────┐
│      Infrastructure Layer               │
│  (Container, MessageBus, Scheduled)     │
└─────────────────────────────────────────┘
```

**Key principles:**
- **Async-by-default** - All I/O is non-blocking
- **Composition over inheritance** - Mix protocols via events
- **Type safety** - Full type hints throughout
- **No shared state** - Only communicate via events
- **Declarative** - Use decorators, not imperative code

## Building a Printer Integration

### Step 1: Define Configuration

```python
@dataclass
class MyPrinterConfig:
    host: str
    port: int = 80
    api_key: str = ""
```

### Step 2: Define Events

```python
@dataclass
class TemperatureChanged(Event):
    temp: float

@dataclass
class StatusChanged(Event):
    status: str
```

### Step 3: Create Protocol(s)

```python
@emits(TemperatureChanged, StatusChanged)
class MyProtocol(Protocol):
    def __init__(self, config: MyPrinterConfig = Injected, bus: MessageBus = Injected):
        self.config = config
        self.bus = bus

    async def start(self):
        # Connect to printer
        # Read data
        # Emit events
        await self.bus.publish(TemperatureChanged(temp=200))

    async def stop(self):
        # Disconnect
        pass
```

### Step 4: Create Printer

```python
@handles(TemperatureChanged, StatusChanged)
class MyPrinter(Printer[MyPrinterConfig]):
    def __init__(
        self,
        config: MyPrinterConfig = Injected,
        protocol: MyProtocol = Injected,
        bus: MessageBus = Injected
    ):
        self.config = config
        self.protocol = protocol
        self.bus = bus

    async def start(self):
        self.bus.subscribe(self)
        await self.protocol.start()

    async def stop(self):
        await self.protocol.stop()
        self.bus.unsubscribe(self)

    def handle_temperature_changed(self, event: TemperatureChanged):
        # Update state
        print(f"Temp: {event.temp}")
```

### Step 5: Wire It Up

```python
# Create container
container = Container()

# Register components
config = MyPrinterConfig(host="printer.local")
container.register_instance(MyPrinterConfig, config)
container.register(MessageBus)
container.register(MyProtocol)
container.register(MyPrinter)

# Resolve and start
printer = await container.resolve(MyPrinter)
await printer.start()
```

Done! You have a fully functional printer integration.

## Testing

**Easy mocking with dependency injection:**

```python
# Mock protocol
class MockProtocol(Protocol):
    async def start(self): pass
    async def stop(self): pass

# Override in tests
container = Container()
container.register_instance(MyProtocol, MockProtocol())

printer = await container.resolve(MyPrinter)
# Uses mock protocol
```

**Testing event handlers:**

```python
# Create bus and printer
bus = MessageBus()
printer = MyPrinter(config, protocol, bus)
bus.subscribe(printer)

# Emit test event
await bus.publish(TemperatureChanged(temp=200))

# Assert state updated
assert printer.state.temperature == 200
```

## Comparison

### Before (Old API)

```python
class Depends:
    ...

@inbox(EventA)
@outbox(EventB)
class MyClient:
    dep: SomeService = Depends()

    @tick(5)
    async def poll(self):
        ...
```

**Issues:**
- `Depends()` looks like a function call but isn't
- `@inbox/@outbox` are informal names
- Mixing concerns (events + DI + scheduling)

### After (New API)

```python
from spc_core import Injected, handles, emits, interval

@handles(EventA)
@emits(EventB)
class MyClient:
    dep: SomeService = Injected

    @interval(5)
    async def poll(self):
        ...
```

**Improvements:**
- `Injected` is clearly a marker, not a call
- `@handles/@emits` are more descriptive
- `@interval` is more intuitive
- Cleaner imports from `spc_core`

## API Reference

### Core Modules

- `spc_core.di` - Dependency injection
- `spc_core.events` - Event system
- `spc_core.scheduling` - Periodic tasks
- `spc_core.transactions` - Atomic updates
- `spc_core.protocols` - Base abstractions

### Key Classes

- `Container` - DI container
- `Injected` - Dependency marker
- `Service` - Base class with lifecycle
- `MessageBus` - Event routing
- `Event` - Base event class
- `Scheduled` - Mixin for scheduling
- `Protocol` - Base protocol class
- `Printer` - Base printer class

### Decorators

- `@handles(*events)` - Declare handled events
- `@emits(*events)` - Declare emitted events
- `@interval(seconds)` - Periodic execution
- `@on_startup` - Run on startup
- `@on_shutdown` - Run on shutdown

### Functions

- `atomic(obj)` - Atomic context manager

## Why This Design?

### 1. **Conventional Naming**

Uses standard Python/industry terminology instead of inventing new names.

### 2. **Type Safe**

Full type hints enable:
- IDE autocomplete
- Static type checking
- Better documentation

### 3. **Testable**

Easy mocking and isolation:
- Override dependencies
- Mock protocols
- Test handlers independently

### 4. **Composable**

Mix and match components:
- Multiple protocols per printer
- Shared message bus
- Reusable services

### 5. **Declarative**

Use decorators and type hints instead of imperative code.

### 6. **Simple**

Each feature is minimal and focused:
- DI: Just inject dependencies
- Events: Just pub/sub
- Scheduling: Just intervals
- Transactions: Just atomic updates

## Project Structure

```
spc-core-future/
├── spc_core/                 # Core modules
│   ├── __init__.py          # Clean exports
│   ├── di.py                # Dependency injection (230 lines)
│   ├── events.py            # Event system (220 lines)
│   ├── scheduling.py        # Scheduling (210 lines)
│   ├── transactions.py      # Transactions (90 lines)
│   └── protocols.py         # Base classes (80 lines)
│
├── examples/                 # Complete, runnable examples
│   ├── 01_dependency_injection.py  # DI demo (230 lines)
│   ├── 02_event_system.py          # Events demo (250 lines)
│   ├── 03_scheduling.py            # Scheduling demo (240 lines)
│   └── 04_complete_printer.py      # Full integration (400 lines)
│
└── README.md                # This file
```

**Total:** ~1,950 lines of clean, documented, tested code

## Next Steps

1. **Run the examples** - Start with example 4 (complete printer)
2. **Read the code** - Each module is well-documented
3. **Build your integration** - Use the patterns from examples
4. **Give feedback** - What works? What doesn't?

## FAQ

### Why not just use the existing EventBus?

The existing EventBus is excellent! We're building **on top of it**, not replacing it. This is a higher-level API that makes common patterns easier.

### Is this production-ready?

**Not yet.** This is an alpha design for feedback. Once validated, it will be merged into the main codebase.

### Will the old API be deprecated?

No immediate plans. This is **additive**. You can use both APIs together.

### How do I migrate?

See `examples/04_complete_printer.py` for a migration template. Key changes:
- `Depends()` → `Injected`
- `@inbox/@outbox` → `@handles/@emits`
- `@tick` → `@interval`
- `EventRouter` → `MessageBus`

### Can I mix old and new APIs?

Yes! The new API is a thin layer that can work alongside existing code.

## Contributing

This is a design proposal. Feedback welcome:

1. **Run the examples** - Do they work? Are they clear?
2. **Try building something** - Is the API intuitive?
3. **Report issues** - What's confusing? What's missing?

## License

Same as simplyprint-ws-client (MIT)

---

**Status:** Alpha - Design Complete, Ready for Feedback

**Next:** Validate with real printer integrations
