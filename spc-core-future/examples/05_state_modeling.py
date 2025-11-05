"""
Example 5: Advanced State Modeling

Demonstrates the complete state modeling system:
- ExternalStateModel with change tracking (Bambu Lab pattern)
- UpdateCoordinator for centralized updates (Home Assistant pattern)
- StateMapper for external-to-internal transformations
- Integration with event system

Run: python -m examples.05_state_modeling
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spc_core import (
    ExternalStateModel,
    Changed,
    UpdateCoordinator,
    UpdateSucceeded,
    UpdateFailed,
    MessageBus,
    handles,
    Event,
)
from spc_core.mapping import StateMapper, FieldMapper
from pydantic import Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# External State Models (from device API)
# ============================================================================

class ToolState(ExternalStateModel):
    """External representation of a tool"""
    temperature: float = 0.0
    target: float = 0.0
    name: str = "Tool 0"


class BedState(ExternalStateModel):
    """External representation of heated bed"""
    temperature: float = 0.0
    target: float = 0.0


class JobState(ExternalStateModel):
    """External representation of print job"""
    file_name: str = ""
    progress: float = 0.0
    time_elapsed: int = 0
    time_remaining: int = 0


class PrinterExternalState(ExternalStateModel):
    """
    Complete external state from printer API.

    Demonstrates nested models, lists, and metadata fields.
    """
    # Status
    status: str = "idle"
    error_message: str = ""

    # Temperature
    tools: List[ToolState] = Field(default_factory=list)
    bed: BedState = Field(default_factory=BedState)

    # Job
    job: Optional[JobState] = None

    # System
    firmware_version: str = "1.0.0"
    config: Dict[str, Any] = Field(default_factory=dict, json_schema_extra={"atomic": True})

    # Metadata (never updated from external)
    last_update_time: float = Field(default=0.0, json_schema_extra={"metadata": True})
    update_count: int = Field(default=0, json_schema_extra={"metadata": True})


# ============================================================================
# Internal State Models (application domain)
# ============================================================================

class InternalTemperature(ExternalStateModel):
    """Internal temperature representation"""
    current: float
    target: float
    is_heating: bool


class InternalJob(ExternalStateModel):
    """Internal job representation"""
    name: str
    progress_percent: float
    eta_seconds: int


class PrinterInternalState(ExternalStateModel):
    """
    Internal application state.

    Transformed from external state with business logic.
    """
    # Simplified status
    is_printing: bool
    is_paused: bool
    is_error: bool
    error: Optional[str] = None

    # Temperatures
    hotend: InternalTemperature
    bed: InternalTemperature

    # Job
    current_job: Optional[InternalJob] = None


# ============================================================================
# State Mapper
# ============================================================================

def create_printer_mapper() -> StateMapper[PrinterExternalState, PrinterInternalState]:
    """
    Create mapper to transform external state to internal state.

    Demonstrates complex transformations and business logic.
    """
    return StateMapper(
        internal_type=PrinterInternalState,
        field_mappings={
            # Status mapping
            "is_printing": FieldMapper(
                extract=lambda e: e.status in ["printing", "resuming"]
            ),
            "is_paused": FieldMapper(
                extract=lambda e: e.status == "paused"
            ),
            "is_error": FieldMapper(
                extract=lambda e: e.status == "error"
            ),
            "error": FieldMapper(
                extract=lambda e: e.error_message if e.error_message else None
            ),

            # Hotend temperature (first tool)
            "hotend": FieldMapper(
                extract=lambda e: e.tools[0] if e.tools else ToolState(),
                transform=lambda tool: InternalTemperature(
                    current=tool.temperature,
                    target=tool.target,
                    is_heating=abs(tool.temperature - tool.target) > 2.0
                )
            ),

            # Bed temperature
            "bed": FieldMapper(
                extract=lambda e: e.bed,
                transform=lambda bed: InternalTemperature(
                    current=bed.temperature,
                    target=bed.target,
                    is_heating=abs(bed.temperature - bed.target) > 2.0
                )
            ),

            # Job mapping
            "current_job": FieldMapper(
                extract=lambda e: e.job,
                transform=lambda job: InternalJob(
                    name=job.file_name,
                    progress_percent=job.progress,
                    eta_seconds=job.time_remaining
                ) if job else None
            ),
        },
        strict=True
    )


# ============================================================================
# Event Handlers
# ============================================================================

@handles(UpdateSucceeded)
class ChangeLogger:
    """
    Logs all state changes with detailed information.

    Demonstrates using change tracking for debugging.
    """

    def handle_update_succeeded(self, event: UpdateSucceeded):
        if not event.changes:
            logger.debug(f"[{event.coordinator_id}] No changes")
            return

        logger.info(f"[{event.coordinator_id}] State updated:")

        for field, change in event.changes.items():
            self._log_change(field, change, indent=2)

    def _log_change(self, name: str, change, indent: int = 0):
        """Recursively log changes"""
        prefix = " " * indent

        if isinstance(change, Changed):
            if change.old != change.new:
                logger.info(f"{prefix}{name}: {change.old} -> {change.new}")
        elif isinstance(change, dict):
            logger.info(f"{prefix}{name}:")
            for sub_name, sub_change in change.items():
                self._log_change(sub_name, sub_change, indent + 2)


@handles(UpdateSucceeded)
class TemperatureMonitor:
    """
    Monitors temperature changes and alerts on thresholds.

    Demonstrates selective field handling.
    """

    def __init__(self):
        self.hotend_alerts = 0
        self.bed_alerts = 0

    def handle_update_succeeded(self, event: UpdateSucceeded):
        state: PrinterExternalState = event.state

        # Check hotend temperature
        if state.tools:
            tool = state.tools[0]
            if tool.temperature > 250:
                self.hotend_alerts += 1
                logger.warning(
                    f"⚠️  High hotend temperature: {tool.temperature}°C "
                    f"(alert #{self.hotend_alerts})"
                )

        # Check bed temperature
        if state.bed.temperature > 100:
            self.bed_alerts += 1
            logger.warning(
                f"⚠️  High bed temperature: {state.bed.temperature}°C "
                f"(alert #{self.bed_alerts})"
            )


@handles(UpdateSucceeded)
class JobProgressTracker:
    """
    Tracks print job progress and milestones.

    Demonstrates job-specific logic.
    """

    def __init__(self):
        self.milestones = [25, 50, 75, 100]
        self.reached = set()

    def handle_update_succeeded(self, event: UpdateSucceeded):
        state: PrinterExternalState = event.state

        if not state.job:
            return

        progress = state.job.progress

        for milestone in self.milestones:
            if milestone not in self.reached and progress >= milestone:
                self.reached.add(milestone)
                logger.info(
                    f"🎯 Milestone: {milestone}% complete! "
                    f"({state.job.file_name})"
                )


@handles(UpdateFailed)
class ErrorHandler:
    """Handles update failures"""

    def handle_update_failed(self, event: UpdateFailed):
        logger.error(
            f"❌ Update failed (attempt {event.attempt}): {event.error}"
        )


# ============================================================================
# Simulated Printer API
# ============================================================================

class SimulatedPrinterAPI:
    """
    Simulates a printer API for demo purposes.

    In production, this would be replaced with real HTTP/WebSocket client.
    """

    def __init__(self):
        self.cycle = 0
        self.base_temp = 25.0
        self.printing = False
        self.progress = 0.0

    async def fetch_state(self) -> PrinterExternalState:
        """
        Simulate fetching state from printer API.

        Models realistic state transitions and temperature changes.
        """
        await asyncio.sleep(0.1)  # Simulate network delay

        self.cycle += 1

        # Simulate state transitions
        if self.cycle == 5:
            # Start heating
            logger.info("📝 Simulating: Start heating")
            self.base_temp = 200.0

        elif self.cycle == 10:
            # Start printing
            logger.info("📝 Simulating: Start print job")
            self.printing = True

        elif self.cycle == 25:
            # Pause
            logger.info("📝 Simulating: Pause print")
            self.printing = False

        elif self.cycle == 30:
            # Resume
            logger.info("📝 Simulating: Resume print")
            self.printing = True

        # Update progress
        if self.printing:
            self.progress = min(100.0, self.progress + 4.0)

        # Build state
        status = "printing" if self.printing else "idle"
        if self.cycle >= 25 and self.cycle < 30:
            status = "paused"

        tools = [
            ToolState(
                temperature=self.base_temp + (self.cycle % 5) * 0.5,
                target=self.base_temp,
                name="Tool 0"
            )
        ]

        bed = BedState(
            temperature=60.0 + (self.cycle % 3) * 0.3,
            target=60.0
        )

        job = None
        if self.printing or (self.cycle >= 25 and self.cycle < 30):
            job = JobState(
                file_name="test_print.gcode",
                progress=self.progress,
                time_elapsed=self.cycle * 10,
                time_remaining=int((100 - self.progress) * 5)
            )

        return PrinterExternalState(
            status=status,
            error_message="",
            tools=tools,
            bed=bed,
            job=job,
            firmware_version="1.2.3",
            config={"key": "value"}
        )


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("  Example 5: Advanced State Modeling")
    print("=" * 70)
    print()

    # Create message bus
    bus = MessageBus()

    # Subscribe handlers
    change_logger = ChangeLogger()
    temp_monitor = TemperatureMonitor()
    job_tracker = JobProgressTracker()
    error_handler = ErrorHandler()

    bus.subscribe(change_logger)
    bus.subscribe(temp_monitor)
    bus.subscribe(job_tracker)
    bus.subscribe(error_handler)

    print("--- Basic State Tracking ---\n")

    # Create simulated API
    api = SimulatedPrinterAPI()

    # Create coordinator
    coordinator = UpdateCoordinator(
        name="printer",
        update_method=api.fetch_state,
        update_interval=timedelta(seconds=1),
        bus=bus,
        max_retries=3,
        retry_backoff=2.0
    )

    # Start coordinator
    await coordinator.start()

    # Run for 35 seconds to see full cycle
    logger.info("Running simulation for 35 seconds...")
    await asyncio.sleep(35)

    # Stop coordinator
    await coordinator.stop()

    print("\n--- State Mapping ---\n")

    # Get final external state
    external_state = coordinator.state

    if external_state:
        logger.info("External State:")
        logger.info(f"  Status: {external_state.status}")
        if external_state.tools:
            logger.info(f"  Hotend: {external_state.tools[0].temperature}°C")
        logger.info(f"  Bed: {external_state.bed.temperature}°C")
        if external_state.job:
            logger.info(f"  Job: {external_state.job.file_name} ({external_state.job.progress}%)")

        # Transform to internal state
        mapper = create_printer_mapper()
        internal_state = mapper.map(external_state)

        logger.info("\nInternal State:")
        logger.info(f"  Is Printing: {internal_state.is_printing}")
        logger.info(f"  Is Paused: {internal_state.is_paused}")
        logger.info(f"  Hotend: {internal_state.hotend.current}°C "
                   f"(heating: {internal_state.hotend.is_heating})")
        logger.info(f"  Bed: {internal_state.bed.current}°C "
                   f"(heating: {internal_state.bed.is_heating})")
        if internal_state.current_job:
            logger.info(f"  Job: {internal_state.current_job.name} "
                       f"({internal_state.current_job.progress_percent}%)")

    print("\n--- Statistics ---\n")

    logger.info(f"Temperature alerts: {temp_monitor.hotend_alerts} hotend, "
               f"{temp_monitor.bed_alerts} bed")
    logger.info(f"Milestones reached: {sorted(job_tracker.reached)}")

    print("\n--- Manual Update ---\n")

    # Restart coordinator for manual update demo
    coordinator = UpdateCoordinator(
        name="printer",
        update_method=api.fetch_state,
        bus=bus
    )

    await coordinator.start()

    # Manual updates
    logger.info("Triggering manual update...")
    success = await coordinator.refresh()
    logger.info(f"Update {'succeeded' if success else 'failed'}")

    await asyncio.sleep(0.5)

    # Non-blocking refresh request
    logger.info("Requesting refresh (non-blocking)...")
    coordinator.request_refresh()

    await asyncio.sleep(0.5)

    await coordinator.stop()

    print("\n--- Change Detection ---\n")

    # Demonstrate change detection
    state1 = PrinterExternalState(
        status="idle",
        tools=[ToolState(temperature=25.0, target=0.0)],
        bed=BedState(temperature=25.0, target=0.0)
    )

    state2 = PrinterExternalState(
        status="printing",
        tools=[ToolState(temperature=200.0, target=200.0)],
        bed=BedState(temperature=60.0, target=60.0),
        job=JobState(
            file_name="test.gcode",
            progress=50.0,
            time_remaining=600
        )
    )

    changes = state1.update_from(state2)

    logger.info("Changes detected:")
    logger.info(f"  Status changed: {state1.has_changed(changes, 'status')}")
    logger.info(f"  Temperature changed: {state1.has_changed(changes, 'tools')}")
    logger.info(f"  Job changed: {state1.has_changed(changes, 'job')}")

    # Get specific change
    status_change = state1.get_change(changes, "status")
    if status_change:
        logger.info(f"  Status: {status_change.old} -> {status_change.new}")

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  ✓ ExternalStateModel tracks changes automatically")
    print("  ✓ UpdateCoordinator handles device updates with retries")
    print("  ✓ StateMapper transforms external to internal models")
    print("  ✓ Full integration with event system")
    print("  ✓ Type-safe throughout")


if __name__ == "__main__":
    asyncio.run(main())
