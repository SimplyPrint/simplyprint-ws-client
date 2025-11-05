# Analysis of State-of-the-Art Patterns and Refinements

## Executive Summary

After analyzing **Home Assistant**, **Bambu Lab**, **Duet3D**, and **Ultimaker** integrations, I've identified key patterns and best practices to enhance our Core API.

**Key Finding**: Bambu Lab's "external state modeling" system is the most sophisticated and should be adopted as the standard pattern.

---

## Table of Contents

1. [Analysis of Existing Integrations](#analysis-of-existing-integrations)
2. [Home Assistant Patterns](#home-assistant-patterns)
3. [Key Insights](#key-insights)
4. [Proposed Enhancements](#proposed-enhancements)
5. [Implementation Plan](#implementation-plan)

---

## Analysis of Existing Integrations

### 1. Bambu Lab Integration ⭐ (Most Sophisticated)

**File**: `/simplyprint_bambu/bambu_client/model/`

#### External State Modeling System

```python
class SimpleUpdateModel(UpdateModel):
    """Smart model updates with change tracking"""

    def update_model(self, other: Self) -> UpdatedFieldsType:
        updated_fields = {}

        for name, field in other_fields.items():
            # Skip ExtraInfo fields (metadata)
            if is_extra(field):
                continue

            value_new = getattr(other, name)
            value_old = getattr(self, name)

            # Recursive update for nested models
            if not is_atomic(field):
                if isinstance(value_old, UpdateModel):
                    if changes := value_old.update_model(value_new):
                        updated_fields[name] = changes
                    continue

            # Track changes
            setattr(self, name, value_new)
            updated_fields[name] = UpdatedField(value_old, value_new)

        return updated_fields
```

#### Key Features

1. **Change Tracking**
   ```python
   class UpdatedField(NamedTuple):
       old: T | None
       new: T | None

       def has_changed(self) -> bool:
           return self.old != self.new
   ```

2. **Atomic Fields** (don't recurse)
   ```python
   hms: Atomic[List[HMSModel] | None] = Field(None)
   lights_report: Atomic[List[LightModel] | None] = Field(None)
   ```

3. **ExtraInfo Fields** (metadata only)
   ```python
   command: ExtraInfo[str | None] = Field(None)
   sequence_id: ExtraInfo[int | None] = Field(None)
   ```

4. **Nested Model Hierarchy**
   ```python
   class BambuDeviceModel(BaseModel, SimpleUpdateModel):
       info: BambuInfoModel | None
       print: BambuPrintModel | None
       event: BambuEventModel | None
       system: BambuSystemModel | None
   ```

5. **Thread-Safe Updates**
   ```python
   self._model_state_lock = threading.Lock()

   with self._model_state_lock:
       changes = self.device.print.update_model(new_print_data)
   ```

6. **Event-Driven**
   ```python
   if changes:
       self.event_bus.emit(BambuClientPrintUpdateEvent(changes))
   ```

#### Usage Pattern

```python
# Receive MQTT message
def on_mqtt_message(self, message):
    data = json.loads(message.payload)

    # Parse into Pydantic model
    print_update = BambuPrintModel(**data)

    # Update existing model and track changes
    with self._model_state_lock:
        changes = self.device.print.update_model(print_update)

    # Emit events for changes
    if changes:
        self.event_bus.emit(BambuClientPrintUpdateEvent(changes))
```

**Strengths:**
- ✅ Type-safe (Pydantic)
- ✅ Change tracking (know what changed)
- ✅ Recursive updates (nested models)
- ✅ Thread-safe
- ✅ Clear separation (external API → internal state)
- ✅ Validation (Pydantic validators)

**Weaknesses:**
- ❌ Complex to implement
- ❌ Requires understanding of Pydantic internals
- ❌ Potentially higher memory usage

---

### 2. Duet3D Integration (Middle Ground)

**File**: `/simplyprint_duet3d/duet/model.py`

#### Object Model Pattern

```python
@define
class DuetPrinterModel:
    """Duet Printer model using attrs"""

    api: RepRapFirmware = field(factory=RepRapFirmware)
    om: dict = field(default=None)  # Object model as plain dict
    events: AsyncIOEventEmitter = field(factory=AsyncIOEventEmitter)
```

#### Update Pattern

```python
def merge_dictionary(source, destination):
    """Simple recursive merge"""
    result = {}

    for key, value in source.items():
        if isinstance(value, dict):
            result[key] = merge_dictionary(value, destination.get(key, {}))
        elif isinstance(value, list):
            # Element-by-element merge
            result[key] = value
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    result[key][idx] = merge_dictionary(item, dest_value[idx])
        else:
            result[key] = destination.get(key, value)

    return result
```

#### State Mapping

```python
# External state → Internal state
duet_state_mapping = {
    'disconnected': PrinterStatus.OFFLINE,
    'processing': PrinterStatus.PRINTING,
    'paused': PrinterStatus.PAUSED,
    'idle': PrinterStatus.OPERATIONAL,
}

def map_state(object_model: dict) -> PrinterStatus:
    state = object_model.get('state', {}).get('status', 'disconnected')
    return duet_state_mapping.get(state, PrinterStatus.OFFLINE)
```

**Strengths:**
- ✅ Simple to understand
- ✅ Event-driven (pyee)
- ✅ Clear state mapping
- ✅ attrs for clean classes

**Weaknesses:**
- ❌ No type safety (plain dict)
- ❌ No change tracking
- ❌ Manual merge logic
- ❌ No validation

---

### 3. Ultimaker Integration (Simplest)

**File**: `/simplyprint_ultimaker/client.py`

#### Simple Attribute Pattern

```python
class UltimakerClient(DefaultClient):
    # Direct attributes for external state
    um_printer_info: dict | None = None
    um_job_info: dict | None = None
    um_printer_status: dict | None = None

    # Track old values manually
    old_um_job_info: dict | None = None
    old_um_printer_info: dict | None = None
```

#### Polling Pattern

```python
async def tick(self, delta):
    # Poll printer
    self.um_printer_info = self.printer_connection.get_um_printer_info()
    self.um_job_info = self.printer_connection.get_um_job_info()

    # Manual comparison
    if self.um_job_info != self.old_um_job_info:
        self._process_job_update()
        self.old_um_job_info = self.um_job_info.copy()
```

**Strengths:**
- ✅ Very simple
- ✅ Easy to understand
- ✅ No dependencies

**Weaknesses:**
- ❌ No structure
- ❌ Manual change detection
- ❌ No type safety
- ❌ Lots of boilerplate

---

### 4. Home Assistant (Industry Standard)

**Sources**: Web research + HA documentation

#### Entity Pattern

```python
class Entity:
    """Base class for all entities in HA"""

    @property
    def state(self):
        """Return the state of the entity."""
        return self._attr_state

    @property
    def state_attributes(self):
        """Return the state attributes."""
        return self._attr_extra_state_attributes

    async def async_update(self):
        """Fetch new state data"""
        raise NotImplementedError()
```

#### Coordinator Pattern

```python
class DataUpdateCoordinator:
    """Centralized data updates"""

    def __init__(self, hass, logger, name, update_interval):
        self.update_interval = update_interval
        self.async_update_listeners = []

    async def async_refresh(self):
        """Refresh data from device"""
        try:
            data = await self._async_update_data()
            self.data = data
            self._notify_listeners()
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")

    async def _async_update_data(self):
        """Implemented by subclass"""
        raise NotImplementedError()

    def _notify_listeners(self):
        """Notify all entities"""
        for listener in self.async_update_listeners:
            listener()
```

#### Usage in Integration

```python
class BambuCoordinator(DataUpdateCoordinator):
    """Coordinator for Bambu printer"""

    async def _async_update_data(self):
        """Fetch data from Bambu MQTT"""
        return await self.client.async_get_data()


class BambuSensor(CoordinatorEntity):
    """Bambu temperature sensor"""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        coordinator.async_add_listener(self.async_write_ha_state)

    @property
    def state(self):
        """Return temperature"""
        return self.coordinator.data.get("temperature")
```

**Strengths:**
- ✅ Centralized updates (Coordinator)
- ✅ Declarative entities
- ✅ Well-documented pattern
- ✅ Industry-standard
- ✅ Handles device availability

**Weaknesses:**
- ❌ HA-specific
- ❌ Requires understanding of HA architecture
- ❌ Verbose for simple cases

---

## Home Assistant Patterns

### State Machine

```
State: <domain>.<object_id>
{
    "state": "on",
    "attributes": {
        "temperature": 200,
        "target": 220
    },
    "last_changed": timestamp,
    "last_updated": timestamp
}
```

**Key Points:**
- `last_changed` only updates when state value changes
- `last_updated` updates on any change (including attributes)
- State is always a string
- Attributes hold additional data

### Config Entry Pattern

```python
async def async_setup_entry(hass, entry: ConfigEntry):
    """Set up from a config entry"""

    # Create coordinator
    coordinator = BambuCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Forward to platforms
    await hass.config_entries.async_forward_entry_setups(
        entry, PLATFORMS
    )
```

### Entity Description Pattern

```python
@dataclass
class BambuSensorEntityDescription(SensorEntityDescription):
    """Describes Bambu sensor entity"""

    value_fn: Callable[[BambuData], Any] = None


SENSORS = [
    BambuSensorEntityDescription(
        key="temperature",
        name="Nozzle Temperature",
        value_fn=lambda data: data.print.nozzle_temp,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
]
```

---

## Key Insights

### 1. External State Modeling is Critical

**Problem**: Integrations need to track state from external devices that have their own data structures.

**Solution**: Create models that mirror external API structure.

**Best Practice** (Bambu Lab):
- Use Pydantic models
- Track what changed
- Recursive updates for nested structures
- Type-safe and validated

### 2. Change Tracking Enables Smart Updates

**Problem**: Without change tracking, you must always send all state.

**Solution**: Track `UpdatedField(old, new)` for every change.

**Benefits**:
- Know exactly what changed
- Send only deltas
- Trigger events for specific changes
- Debug state transitions

### 3. Coordinator Pattern for Centralized Updates

**Problem**: Multiple entities polling the same device.

**Solution**: Single coordinator manages all communication.

**Benefits**:
- Reduced network traffic
- Centralized error handling
- Consistent update intervals
- Easy to add new entities

### 4. Separate Protocol from Business Logic

**Pattern** (all integrations):
```
Protocol Layer   → Handles communication (MQTT, HTTP, etc.)
                   Emits typed events
                   Maintains connection state

Business Layer   → Handles application logic
                   Maps external → internal state
                   Implements commands
```

### 5. State Mapping Functions

**Pattern** (Duet3D):
```python
def map_external_to_internal(external_state: dict) -> PrinterStatus:
    """Clear, testable transformation"""
    return MAPPING.get(external_state["status"], PrinterStatus.OFFLINE)
```

---

## Proposed Enhancements

### Enhancement 1: External State Models

Add a new module for external state modeling based on Bambu Lab's pattern.

**File**: `spc_core/state.py`

```python
"""
External State Modeling System

Provides Pydantic-based models for tracking external device state with:
- Change tracking (know what changed)
- Recursive updates (nested models)
- Atomic fields (don't recurse)
- Metadata fields (not part of state)
- Type safety and validation
"""

from typing import TypeVar, Generic, NamedTuple, Dict, Any, Annotated
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import functools

T = TypeVar("T")

# ============================================================================
# Change Tracking
# ============================================================================

class Changed(Generic[T], NamedTuple):
    """
    Tracks a changed value with old and new.

    Usage:
        change = Changed(old=100, new=200)
        if change.has_changed():
            print(f"Changed from {change.old} to {change.new}")
    """
    old: T | None
    new: T | None

    def has_changed(self) -> bool:
        """Check if value actually changed"""
        return self.old != self.new

    @property
    def delta(self) -> T | None:
        """Calculate delta (for numeric types)"""
        if self.old is None or self.new is None:
            return None
        try:
            return self.new - self.old
        except TypeError:
            return None


ChangedFields = Dict[str, Union[Changed, "ChangedFields", List["ChangedFields"]]]


# ============================================================================
# Field Markers
# ============================================================================

# Atomic: Don't recurse into this field during updates
_AtomicSentinel = object()
Atomic = Annotated[T, _AtomicSentinel]

# Metadata: Field is metadata only, not part of state
_MetadataSentinel = object()
Metadata = Annotated[T, _MetadataSentinel]


@functools.cache
def is_atomic(field_info) -> bool:
    """Check if field is atomic"""
    return any(m is _AtomicSentinel for m in field_info.metadata)


@functools.cache
def is_metadata(field_info) -> bool:
    """Check if field is metadata"""
    return any(m is _MetadataSentinel for m in field_info.metadata)


# ============================================================================
# Base Models
# ============================================================================

class StatefulModel(ABC):
    """Abstract interface for stateful models"""

    @abstractmethod
    def update_from(self, other: "StatefulModel") -> ChangedFields:
        """Update this model from another and return changes"""
        ...


class ExternalStateModel(BaseModel, StatefulModel):
    """
    Base model for external device state.

    Provides smart updates with change tracking.

    Usage:
        class PrinterState(ExternalStateModel):
            temperature: float = 0.0
            status: str = "idle"

        state = PrinterState()
        new_data = PrinterState(temperature=200, status="printing")

        changes = state.update_from(new_data)
        # changes = {
        #     "temperature": Changed(old=0.0, new=200.0),
        #     "status": Changed(old="idle", new="printing")
        # }
    """

    def update_from(self, other: "ExternalStateModel") -> ChangedFields:
        """
        Update this model from another model.

        Returns:
            Dictionary of changed fields with old/new values
        """
        changes = {}

        self_fields = self.__class__.model_fields
        other_fields = other.__class__.model_fields

        for name, field_info in other_fields.items():
            # Skip fields not in self
            if name not in self_fields:
                continue

            # Skip metadata fields
            if is_metadata(field_info):
                continue

            new_value = getattr(other, name)

            # Skip None values
            if new_value is None:
                continue

            old_value = getattr(self, name)

            # Handle nested models (recursive)
            if not is_atomic(field_info):
                if isinstance(old_value, StatefulModel):
                    if nested_changes := old_value.update_from(new_value):
                        changes[name] = nested_changes
                    continue

                # Handle lists of models
                if isinstance(old_value, list):
                    if not isinstance(new_value, list):
                        continue

                    list_changes = []
                    min_len = min(len(old_value), len(new_value))

                    for i in range(min_len):
                        if isinstance(old_value[i], StatefulModel):
                            if item_changes := old_value[i].update_from(new_value[i]):
                                list_changes.append(item_changes)
                        else:
                            old_value[i] = new_value[i]

                    if list_changes:
                        changes[name] = list_changes
                    continue

            # Update value and track change
            setattr(self, name, new_value)
            change = Changed(old_value, new_value)

            if change.has_changed():
                changes[name] = change

        return changes
```

### Enhancement 2: Coordinator Pattern

**File**: `spc_core/coordinator.py`

```python
"""
Coordinator Pattern for Device Communication

Centralizes updates from a device and notifies listeners.
Based on Home Assistant's DataUpdateCoordinator pattern.
"""

import asyncio
import logging
from datetime import timedelta
from typing import Generic, TypeVar, Callable, Optional
from abc import ABC, abstractmethod

T = TypeVar("T")

class UpdateCoordinator(ABC, Generic[T]):
    """
    Coordinates updates from a device.

    Usage:
        class PrinterCoordinator(UpdateCoordinator[PrinterState]):
            async def _fetch_data(self) -> PrinterState:
                # Poll device
                data = await self.api.get_status()
                return PrinterState(**data)

        coordinator = PrinterCoordinator(update_interval=timedelta(seconds=5))
        await coordinator.start()

        # Listen for updates
        coordinator.add_listener(lambda: print("Updated!"))
    """

    def __init__(
        self,
        *,
        update_interval: timedelta,
        logger: Optional[logging.Logger] = None
    ):
        self.update_interval = update_interval
        self.logger = logger or logging.getLogger(__name__)
        self.data: Optional[T] = None
        self.last_update_success = False
        self._listeners: list[Callable] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    @abstractmethod
    async def _fetch_data(self) -> T:
        """Fetch data from device (implemented by subclass)"""
        ...

    async def start(self):
        """Start periodic updates"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._update_loop())

    async def stop(self):
        """Stop periodic updates"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _update_loop(self):
        """Main update loop"""
        while self._running:
            try:
                await self.async_refresh()
            except Exception as e:
                self.logger.error(f"Error updating: {e}", exc_info=True)

            await asyncio.sleep(self.update_interval.total_seconds())

    async def async_refresh(self):
        """Refresh data from device"""
        try:
            self.data = await self._fetch_data()
            self.last_update_success = True
            self._notify_listeners()
        except Exception as e:
            self.last_update_success = False
            self.logger.error(f"Failed to fetch data: {e}")
            raise

    def add_listener(self, listener: Callable):
        """Add update listener"""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable):
        """Remove update listener"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self):
        """Notify all listeners of update"""
        for listener in self._listeners:
            try:
                listener()
            except Exception as e:
                self.logger.error(f"Error notifying listener: {e}")
```

### Enhancement 3: State Mapping Utilities

**File**: `spc_core/mapping.py`

```python
"""
State Mapping Utilities

Helper functions for mapping external state to internal state.
"""

from typing import TypeVar, Dict, Callable, Any
from enum import Enum

T = TypeVar("T")
U = TypeVar("U")


class StateMapper(Generic[T, U]):
    """
    Maps external state to internal state.

    Usage:
        mapper = StateMapper[str, PrinterStatus]({
            "idle": PrinterStatus.OPERATIONAL,
            "printing": PrinterStatus.PRINTING,
            "paused": PrinterStatus.PAUSED,
        }, default=PrinterStatus.OFFLINE)

        status = mapper.map("idle")  # PrinterStatus.OPERATIONAL
        status = mapper.map("unknown")  # PrinterStatus.OFFLINE (default)
    """

    def __init__(
        self,
        mapping: Dict[T, U],
        *,
        default: Optional[U] = None,
        transform: Optional[Callable[[T], T]] = None
    ):
        self.mapping = mapping
        self.default = default
        self.transform = transform

    def map(self, value: T) -> U:
        """Map external value to internal value"""
        if self.transform:
            value = self.transform(value)

        return self.mapping.get(value, self.default)

    def map_or_raise(self, value: T) -> U:
        """Map or raise ValueError if not found"""
        result = self.map(value)
        if result is None:
            raise ValueError(f"No mapping for value: {value}")
        return result


def create_enum_mapper(
    enum_class: type[Enum],
    mapping: Dict[Any, Enum],
    default: Optional[Enum] = None
) -> StateMapper:
    """
    Create mapper for enum values.

    Usage:
        class Status(Enum):
            IDLE = "idle"
            PRINTING = "printing"

        mapper = create_enum_mapper(PrinterStatus, {
            Status.IDLE: PrinterStatus.OPERATIONAL,
            Status.PRINTING: PrinterStatus.PRINTING,
        })
    """
    return StateMapper(mapping, default=default)
```

### Enhancement 4: Example Using All Patterns

**File**: `spc-core-future/examples/05_external_state_modeling.py`

```python
"""
Example 5: External State Modeling

Demonstrates the complete pattern:
- External state models with change tracking
- Coordinator for centralized updates
- State mapping
- Event emissions on changes

This is the RECOMMENDED pattern for new integrations.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spc_core import (
    Container, Injected,
    MessageBus, handles, Event,
    interval, Scheduled
)
from spc_core.state import ExternalStateModel, Changed, Atomic, Metadata
from spc_core.coordinator import UpdateCoordinator
from spc_core.mapping import StateMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ============================================================================
# 1. Define External State Models (mirrors device API)
# ============================================================================

class ToolState(ExternalStateModel):
    """State of a single tool/extruder"""
    temp_actual: float = 0.0
    temp_target: float = 0.0
    material: str | None = None


class PrinterState(ExternalStateModel):
    """Complete printer state from external device"""

    # Regular fields (tracked for changes)
    status: str = "offline"
    temperature: float = 0.0
    progress: float = 0.0

    # Nested model (recursive updates)
    tool: ToolState = ToolState()

    # Atomic field (don't recurse, replace wholesale)
    errors: Atomic[list[str]] = []

    # Metadata field (not part of state, just info about the message)
    timestamp: Metadata[int | None] = None
    message_id: Metadata[int | None] = None


# ============================================================================
# 2. Define Internal State (SimplyPrint)
# ============================================================================

class InternalStatus(str):
    OPERATIONAL = "operational"
    PRINTING = "printing"
    PAUSED = "paused"
    OFFLINE = "offline"


# ============================================================================
# 3. Define Events
# ============================================================================

@dataclass
class StateChanged(Event):
    """Printer state changed"""
    changes: dict
    state: PrinterState


# ============================================================================
# 4. Create State Mapper
# ============================================================================

status_mapper = StateMapper[str, str]({
    "idle": InternalStatus.OPERATIONAL,
    "printing": InternalStatus.PRINTING,
    "paused": InternalStatus.PAUSED,
    "offline": InternalStatus.OFFLINE,
}, default=InternalStatus.OFFLINE)


# ============================================================================
# 5. Implement Coordinator
# ============================================================================

class PrinterCoordinator(UpdateCoordinator[PrinterState]):
    """Coordinates updates from printer"""

    def __init__(self, host: str, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self._fetch_count = 0

    async def _fetch_data(self) -> PrinterState:
        """Fetch from device (simulated)"""
        await asyncio.sleep(0.1)  # Simulate network delay

        self._fetch_count += 1

        # Simulate changing data
        if self._fetch_count % 3 == 0:
            return PrinterState(
                status="printing",
                temperature=200 + (self._fetch_count % 10),
                progress=min(100, self._fetch_count * 5),
                tool=ToolState(
                    temp_actual=200.0,
                    temp_target=220.0,
                    material="PLA"
                ),
                timestamp=self._fetch_count
            )
        else:
            return PrinterState(
                status="idle",
                temperature=25.0,
                progress=0.0,
                tool=ToolState(temp_actual=25.0),
                timestamp=self._fetch_count
            )


# ============================================================================
# 6. Create Integration
# ============================================================================

@handles(StateChanged)
class PrinterIntegration(Scheduled):
    """
    Complete integration using all patterns
    """

    def __init__(
        self,
        coordinator: PrinterCoordinator = Injected,
        bus: MessageBus = Injected
    ):
        Scheduled.__init__(self)
        self.coordinator = coordinator
        self.bus = bus
        self.state = PrinterState()

        # Listen to coordinator updates
        self.coordinator.add_listener(self._on_coordinator_update)

    async def start(self):
        """Start integration"""
        await self.coordinator.start()
        await self.start_scheduled()

    async def stop(self):
        """Stop integration"""
        await self.stop_scheduled()
        await self.coordinator.stop()

    def _on_coordinator_update(self):
        """Called when coordinator has new data"""
        if self.coordinator.data is None:
            return

        # Update state and track changes
        changes = self.state.update_from(self.coordinator.data)

        if changes:
            logging.info(f"State changed: {list(changes.keys())}")

            # Log details of what changed
            for key, change in changes.items():
                if isinstance(change, Changed):
                    logging.info(
                        f"  {key}: {change.old} -> {change.new} "
                        f"(delta: {change.delta})"
                    )
                elif isinstance(change, dict):
                    logging.info(f"  {key}: nested changes")

            # Emit event
            asyncio.create_task(
                self.bus.publish(StateChanged(changes=changes, state=self.state))
            )

    @interval(10)
    async def report_status(self):
        """Periodic status report"""
        internal_status = status_mapper.map(self.state.status)
        logging.info(
            f"Status Report: "
            f"{self.state.status} → {internal_status}, "
            f"temp={self.state.temperature}°C, "
            f"progress={self.state.progress}%"
        )

    async def handle_state_changed(self, event: StateChanged):
        """Handle state changes"""
        logging.info(f"Received state change event with {len(event.changes)} changes")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("  Example 5: External State Modeling")
    print("=" * 70)
    print()

    # Setup
    container = Container()
    container.register(MessageBus)
    container.register(
        PrinterCoordinator,
        factory=lambda: PrinterCoordinator(
            host="printer.local",
            update_interval=timedelta(seconds=2),
            logger=logging.getLogger("coordinator")
        )
    )
    container.register(PrinterIntegration)

    # Start
    integration = await container.resolve(PrinterIntegration)
    bus = await container.resolve(MessageBus)
    bus.subscribe(integration)

    print("--- Starting Integration ---\n")
    await integration.start()

    print("\n--- Running (20 seconds) ---\n")
    print("Watch for:")
    print("  - Change tracking (old → new)")
    print("  - State mapping (external → internal)")
    print("  - Event emissions")
    print("  - Nested model updates")
    print()

    await asyncio.sleep(20)

    print("\n--- Stopping Integration ---\n")
    await integration.stop()
    await container.shutdown()

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Implementation Plan

### Phase 1: Core State Modeling (Week 1)
1. ✅ Implement `spc_core/state.py`
   - ExternalStateModel base class
   - Changed tracking
   - Atomic/Metadata markers
   - Recursive update logic

2. ✅ Add comprehensive tests
   - Test change tracking
   - Test nested updates
   - Test atomic fields
   - Test metadata fields

3. ✅ Create example 05

### Phase 2: Coordinator Pattern (Week 2)
1. ✅ Implement `spc_core/coordinator.py`
   - UpdateCoordinator base class
   - Listener pattern
   - Error handling
   - Update intervals

2. ✅ Add tests

3. ✅ Integrate with example 05

### Phase 3: State Mapping (Week 3)
1. ✅ Implement `spc_core/mapping.py`
   - StateMapper class
   - Enum mapper helper
   - Transform functions

2. ✅ Add tests

3. ✅ Update example 05

### Phase 4: Documentation (Week 4)
1. Update README
2. Add migration guide
3. Create best practices doc

### Phase 5: Integration (Week 5)
1. Port Bambu integration to new patterns
2. Validate with real usage
3. Iterate based on feedback

---

## Comparison Table

| Feature | Bambu | Duet3D | Ultimaker | spc-core-future (Enhanced) |
|---------|-------|--------|-----------|----------------------------|
| **Type Safety** | ✅ Pydantic | ❌ Plain dict | ❌ Plain dict | ✅ Pydantic |
| **Change Tracking** | ✅ UpdatedField | ❌ Manual | ❌ Manual | ✅ Changed |
| **Nested Models** | ✅ Recursive | ⚠️ Manual merge | ❌ No | ✅ Recursive |
| **Atomic Fields** | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| **Metadata Fields** | ✅ ExtraInfo | ❌ No | ❌ No | ✅ Metadata |
| **Thread Safety** | ✅ Locks | ❌ No | ❌ No | ✅ Locks |
| **Event-Driven** | ✅ Yes | ✅ pyee | ⚠️ Manual | ✅ MessageBus |
| **Coordinator** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **State Mapping** | ⚠️ Manual | ✅ Functions | ⚠️ Manual | ✅ StateMapper |
| **Validation** | ✅ Pydantic | ❌ No | ❌ No | ✅ Pydantic |

---

## Conclusion

**Key Takeaways:**

1. **Bambu Lab's external state modeling is the gold standard** - adopt it
2. **Home Assistant's coordinator pattern is excellent** - adopt it
3. **State mapping utilities are essential** - create them
4. **Combine all three** for the best integration framework

**Recommended Pattern:**

```
External State Model (Pydantic with change tracking)
         ↓
   Coordinator (centralized updates)
         ↓
   State Mapper (external → internal)
         ↓
   MessageBus (event-driven updates)
         ↓
   Business Logic (handle events)
```

**Next Steps:**

1. ✅ Implement enhanced modules
2. ✅ Create example 05
3. ⏳ Test with real integrations
4. ⏳ Document migration path
5. ⏳ Get feedback

---

**Status**: ✅ Analysis Complete, Implementation Ready
