"""
Example 2: Event System

Demonstrates the clean MessageBus with @handles and @emits decorators.

Run: python -m examples.02_event_system
"""

import asyncio
import logging
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spc_core import MessageBus, handles, emits, Event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# Define Events
# ============================================================================

@dataclass
class TemperatureChanged(Event):
    """Event: Temperature has changed"""
    tool: int
    actual: float
    target: float


@dataclass
class StatusChanged(Event):
    """Event: Printer status has changed"""
    status: str


@dataclass
class JobProgress(Event):
    """Event: Print job progress"""
    percent: float
    time_remaining: int


@dataclass
class GcodeCommand(Event):
    """Event: Execute gcode"""
    commands: list[str]


# ============================================================================
# Example 1: Simple Event Handler
# ============================================================================

@handles(TemperatureChanged)
class TemperatureLogger:
    """Logs temperature changes"""

    def handle_temperature_changed(self, event: TemperatureChanged):
        logger.info(
            f"[LOGGER] Temperature: tool={event.tool}, "
            f"actual={event.actual}°C, target={event.target}°C"
        )


# ============================================================================
# Example 2: Multiple Event Handlers
# ============================================================================

@handles(TemperatureChanged, StatusChanged, JobProgress)
class Display:
    """Updates display with printer status"""

    def __init__(self):
        self.temperature = 0.0
        self.status = "idle"
        self.progress = 0.0

    def handle_temperature_changed(self, event: TemperatureChanged):
        self.temperature = event.actual
        logger.info(f"[DISPLAY] Updated temperature: {self.temperature}°C")

    def handle_status_changed(self, event: StatusChanged):
        self.status = event.status
        logger.info(f"[DISPLAY] Updated status: {self.status}")

    def handle_job_progress(self, event: JobProgress):
        self.progress = event.percent
        logger.info(f"[DISPLAY] Updated progress: {self.progress}%")

    def show(self):
        logger.info(
            f"[DISPLAY] Status: {self.status}, "
            f"Temp: {self.temperature}°C, "
            f"Progress: {self.progress}%"
        )


# ============================================================================
# Example 3: Event Emitter
# ============================================================================

@emits(TemperatureChanged, StatusChanged)
class TemperatureSensor:
    """Sensor that emits temperature events"""

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.temperature = 25.0

    async def read(self):
        """Simulate reading temperature"""
        await asyncio.sleep(0.1)

        # Simulate temperature change
        import random
        self.temperature += random.uniform(-2, 2)

        # Emit event
        await self.bus.publish(TemperatureChanged(
            tool=0,
            actual=self.temperature,
            target=200.0
        ))


# ============================================================================
# Example 4: Async Event Handler
# ============================================================================

@handles(GcodeCommand)
class PrinterController:
    """Handles gcode commands"""

    async def handle_gcode_command(self, event: GcodeCommand):
        logger.info(f"[CONTROLLER] Executing gcode: {event.commands}")

        for cmd in event.commands:
            await asyncio.sleep(0.1)  # Simulate execution
            logger.info(f"[CONTROLLER]   Executed: {cmd}")


# ============================================================================
# Example 5: Event Chain
# ============================================================================

@handles(JobProgress)
@emits(StatusChanged)
class JobMonitor:
    """Monitors job progress and emits status changes"""

    def __init__(self, bus: MessageBus):
        self.bus = bus

    async def handle_job_progress(self, event: JobProgress):
        logger.info(f"[MONITOR] Job at {event.percent}%")

        # When job completes, emit status change
        if event.percent >= 100:
            logger.info("[MONITOR] Job complete! Changing status...")
            await self.bus.publish(StatusChanged(status="completed"))


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("  Example 2: Event System")
    print("=" * 70)
    print()

    # Create message bus
    bus = MessageBus()

    print("--- Simple Event Handler ---\n")

    # Subscribe temperature logger
    temp_logger = TemperatureLogger()
    bus.subscribe(temp_logger)

    # Publish event
    await bus.publish(TemperatureChanged(tool=0, actual=195.5, target=200.0))

    print("\n--- Multiple Event Handlers ---\n")

    # Subscribe display
    display = Display()
    bus.subscribe(display)

    # Publish multiple events
    await bus.publish(TemperatureChanged(tool=0, actual=200.0, target=200.0))
    await bus.publish(StatusChanged(status="printing"))
    await bus.publish(JobProgress(percent=45.2, time_remaining=600))

    # Show display
    display.show()

    print("\n--- Event Emitter ---\n")

    # Create sensor that emits events
    sensor = TemperatureSensor(bus)

    # Read multiple times
    for i in range(3):
        await sensor.read()
        await asyncio.sleep(0.2)

    print("\n--- Async Event Handler ---\n")

    # Subscribe controller
    controller = PrinterController()
    bus.subscribe(controller)

    # Send gcode commands
    await bus.publish(GcodeCommand(commands=["G28", "G0 Z10", "M104 S200"]))

    print("\n--- Event Chain ---\n")

    # Subscribe job monitor (emits events in response to events)
    monitor = JobMonitor(bus)
    bus.subscribe(monitor)

    # Simulate job progress
    for percent in [10, 50, 90, 100]:
        await bus.publish(JobProgress(percent=percent, time_remaining=int((100-percent)*10)))
        await asyncio.sleep(0.3)

    print("\n--- Unsubscribe ---\n")

    # Unsubscribe components
    bus.unsubscribe(temp_logger)
    bus.unsubscribe(display)

    # These events won't be handled by logger or display
    await bus.publish(TemperatureChanged(tool=0, actual=210.0, target=200.0))
    logger.info("[DEMO] Temperature event published (only monitor/controller see it)")

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
