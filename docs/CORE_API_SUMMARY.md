# Core API Design Summary

## What Was Built

This design proposal introduces a modern, composable architecture for the SimplyPrint WebSocket Client with:

### 1. Dependency Injection System (`core/di.py`)

- **FastAPI-style `Depends()` pattern** for type-safe dependency injection
- **Automatic resolution** of dependency graphs
- **Singleton and transient scopes**
- **Async support** for factory functions
- **Lifecycle hooks** via `Injectable` base class

**Benefits:**
- Cleaner code with less boilerplate
- Easy testing via dependency overrides
- Type-safe with full IDE support
- Prevents tight coupling

### 2. Event System Enhancements (`core/protocols.py`)

- **`@inbox` and `@outbox` decorators** for declaring event contracts
- **Type-safe event routing** via `EventRouter`
- **Automatic handler discovery** based on naming conventions
- **Validation** of event flow (optional)
- **Isolation** - errors in one handler don't affect others

**Benefits:**
- Self-documenting code (clear event contracts)
- Type safety catches errors at development time
- Composable protocols via event intersection
- Easy to test individual components

### 3. Scheduling System (`core/decorators.py`)

- **`@tick(interval)` decorator** for periodic task execution
- **`@on_init` and `@on_destroy` hooks** for lifecycle management
- **Automatic task management** via `SchedulerMixin`
- **Error handling** with configurable retry behavior
- **No concurrent execution** of the same task

**Benefits:**
- Declarative task scheduling
- No manual task tracking
- Clean lifecycle management
- Prevents resource leaks

### 4. Transaction Support (`core/transactions.py`)

- **Context managers** for atomic state changes
- **Automatic rollback** on exceptions
- **Snapshot-based** implementation (simple and safe)
- **Async support** via `AsyncTransaction`
- **Validation hooks** via `Transactional` base class

**Benefits:**
- Prevents partial state updates
- Easy error recovery
- Clear transaction boundaries
- Testable state changes

### 5. Complete Working Example (`examples/core_api_demo.py`)

A fully functional demonstration showing:
- Multiple protocols (HTTP polling + MQTT)
- Event routing between components
- Dependency injection in action
- Scheduled tasks
- Complete lifecycle management

**Run it:** `python -m examples.core_api_demo`

## Architecture Principles

1. **Async-by-default** - All I/O is non-blocking
2. **Separation of concerns** - Protocols handle I/O, printers handle business logic
3. **Composition over inheritance** - Mix protocols via events, not subclassing
4. **Type safety** - Leverage Python's type system throughout
5. **No shared state** - Communication only through typed events
6. **Declarative** - Use decorators and type hints over imperative code

## Key Patterns

### Protocol Composition

```python
@inbox(TempEvent, StatusEvent)
class MyPrinter:
    # Compose multiple protocols
    ws: WebSocketProtocol = Depends()
    http: HTTPPollingProtocol = Depends()
    mqtt: MQTTProtocol = Depends()

    # Single handler receives events from ALL protocols
    async def on_temp(self, event: TempEvent):
        self.state.temp = event.value
```

### Dependency Injection

```python
class MyProtocol:
    # Dependencies automatically injected
    http: HTTPClient = Depends()
    config: MyConfig = Depends()

    async def start(self):
        await self.http.connect(self.config.host)
```

### Event-Driven

```python
@outbox(TempEvent, StatusEvent)  # What I emit
@inbox(GcodeEvent)                # What I receive
class MyProtocol:
    async def on_gcode(self, event: GcodeEvent):
        # Handle incoming event
        ...
```

### Scheduled Tasks

```python
class MyPrinter(SchedulerMixin):
    @tick(5)  # Every 5 seconds
    async def poll_temperature(self):
        ...
```

## Files Created

### Core Implementation
- `simplyprint_ws_client/core/di.py` - Dependency injection (328 lines)
- `simplyprint_ws_client/core/protocols.py` - Event routing (342 lines)
- `simplyprint_ws_client/core/decorators.py` - Scheduling (323 lines)
- `simplyprint_ws_client/core/transactions.py` - Transactions (347 lines)

### Documentation
- `docs/CORE_API_DESIGN.md` - Complete architecture design (1000+ lines)
- `docs/CORE_API_USAGE.md` - Usage guide with examples (600+ lines)
- `docs/CORE_API_SUMMARY.md` - This summary

### Examples
- `examples/core_api_demo.py` - Working demo (400+ lines)

**Total:** ~3,500 lines of code, documentation, and examples

## Migration Strategy

The design is **incremental** and **backward-compatible**:

### Phase 1: Foundation (Weeks 1-2)
- ✅ DI system implementation
- ✅ Event router implementation
- ✅ Documentation

### Phase 2: Integration (Weeks 3-4)
- Integrate with existing `Client` class
- Add `@tick` to existing scheduler
- Transaction support in `StateModel`

### Phase 3: Examples (Weeks 5-6)
- Bambu Lab printer example
- OctoPrint printer example
- Prusa Link printer example

### Phase 4: Migration (Weeks 7-8)
- Migrate existing clients
- Update tutorials
- Performance testing

## Benefits

### For Integration Developers

- **Less boilerplate** - DI handles wiring
- **Type safety** - Catch errors early
- **Easier testing** - Mock dependencies
- **Clear contracts** - inbox/outbox self-document
- **Flexible composition** - Mix protocols easily

### For Core Maintainers

- **Better separation** - Protocols vs business logic
- **Easier to extend** - Add new protocols without touching printers
- **Testable** - Components isolated
- **Maintainable** - Clear structure

### For SimplyPrint Platform

- **More integrations** - Easier to add new printers
- **More reliable** - Type safety, transactions, error isolation
- **Better performance** - Async-by-default
- **Future-proof** - Modern patterns

## Next Steps

1. **Review** - Get feedback on design
2. **Prototype** - Build one real printer integration (e.g., Bambu)
3. **Iterate** - Refine based on real usage
4. **Integrate** - Merge with existing codebase
5. **Migrate** - Update existing clients
6. **Document** - Tutorial and API reference

## Design Questions for Discussion

1. **DI Scope** - Should we support more scopes (request, session)?
2. **Event Validation** - Enable by default or opt-in?
3. **Transaction Strategy** - Snapshot vs. change tracking?
4. **Pydantic Usage** - Where to use it vs. plain dataclasses?
5. **Performance** - What are the requirements? (printers per instance, event throughput)
6. **File Storage** - Need abstraction for local/remote files?
7. **Tunneling** - How to handle remote access?
8. **Testing** - Mock framework or manual mocks?

## Conclusion

This design provides a **modern, composable, type-safe architecture** for building printer integrations while maintaining **backward compatibility** with the existing codebase.

The **incremental migration strategy** allows for:
- Gradual adoption
- Validation of design decisions
- Minimal disruption to existing code

The **working demo** proves the concepts and provides a template for new integrations.

---

**Status:** ✅ Design complete, ready for review and discussion
**Next:** Get feedback, prototype real integration, iterate
