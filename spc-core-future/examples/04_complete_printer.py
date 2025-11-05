"""
Example 4: Complete Printer Integration

Demonstrates a full printer implementation using all features:
- Dependency injection
- Event-driven architecture
- Multiple protocols
- Scheduled tasks
- Lifecycle management

Run: python -m examples.04_complete_printer
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spc_core import (
    Container, Injected, Service,
    MessageBus, handles, emits, Event,
    Scheduled, interval, on_startup, on_shutdown,
    Protocol, Printer, atomic,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PrinterConfig:
    """Printer configuration"""
    name: str
    host: str
    port: int = 80
    api_key: str = ""


# ============================================================================
# Events
# ============================================================================

@dataclass
class TemperatureChanged(Event):
    tool: int
    actual: float
    target: float


@dataclass
class StatusChanged(Event):
    status: str


@dataclass
class JobProgress(Event):
    percent: float
    time_remaining: int


@dataclass
class GcodeCommand(Event):
    commands: list[str]


# ============================================================================
# Printer State
# ============================================================================

class PrinterState:
    """Printer state with transaction support"""

    def __init__(self):
        self.temperature = 0.0
        self.target_temp = 0.0
        self.status = "idle"
        self.progress = 0.0

    def __repr__(self):
        return (
            f"PrinterState(temp={self.temperature:.1f}/{self.target_temp:.1f}, "
            f"status={self.status}, progress={self.progress:.1f}%)"
        )


# ============================================================================
# Protocols
# ============================================================================

@emits(TemperatureChanged, StatusChanged)
class HTTPPollingProtocol(Protocol, Scheduled):
    """
    Protocol: Poll printer via HTTP REST API

    Demonstrates:
    - Protocol implementation
    - Scheduled polling
    - Event emission
    """

    def __init__(self, config: PrinterConfig = Injected, bus: MessageBus = Injected):
        Scheduled.__init__(self)
        self.config = config
        self.bus = bus
        self._running = False

    async def start(self):
        logger.info(f"[HTTP] Connecting to {self.config.host}:{self.config.port}")
        await asyncio.sleep(0.2)
        self._running = True
        logger.info("[HTTP] Connected")

        # Start scheduled polling
        await self.start_scheduled()

    async def stop(self):
        logger.info("[HTTP] Disconnecting...")
        self._running = False

        # Stop scheduled polling
        await self.stop_scheduled()
        logger.info("[HTTP] Disconnected")

    @interval(2)
    async def poll_temperature(self):
        """Poll temperature every 2 seconds"""
        if not self._running:
            return

        # Simulate HTTP GET /api/temperature
        await asyncio.sleep(0.1)

        # Simulate response
        import random
        actual = 20 + random.uniform(0, 200)
        target = 200 if actual < 190 else 0

        # Emit event
        await self.bus.publish(TemperatureChanged(
            tool=0,
            actual=actual,
            target=target
        ))

    @interval(5)
    async def poll_status(self):
        """Poll status every 5 seconds"""
        if not self._running:
            return

        # Simulate HTTP GET /api/status
        await asyncio.sleep(0.1)

        # Emit status
        statuses = ["idle", "printing", "heating"]
        import random
        status = random.choice(statuses)

        await self.bus.publish(StatusChanged(status=status))


@emits(JobProgress)
class WebSocketProtocol(Protocol):
    """
    Protocol: Real-time updates via WebSocket

    Demonstrates:
    - Async protocol
    - Real-time event streaming
    """

    def __init__(self, config: PrinterConfig = Injected, bus: MessageBus = Injected):
        self.config = config
        self.bus = bus
        self._task = None
        self._running = False

    async def start(self):
        logger.info(f"[WS] Connecting to ws://{self.config.host}/websocket")
        await asyncio.sleep(0.2)
        self._running = True
        logger.info("[WS] Connected")

        # Start listening task
        self._task = asyncio.create_task(self._listen())

    async def stop(self):
        logger.info("[WS] Disconnecting...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("[WS] Disconnected")

    async def _listen(self):
        """Simulate WebSocket message loop"""
        try:
            while self._running:
                # Simulate receiving message
                await asyncio.sleep(3)

                if not self._running:
                    break

                # Simulate job progress update
                import random
                progress = random.uniform(0, 100)
                time_remaining = int((100 - progress) * 10)

                await self.bus.publish(JobProgress(
                    percent=progress,
                    time_remaining=time_remaining
                ))

        except asyncio.CancelledError:
            logger.debug("[WS] Listen task cancelled")
            raise


# ============================================================================
# Printer Implementation
# ============================================================================

@handles(TemperatureChanged, StatusChanged, JobProgress, GcodeCommand)
class VirtualPrinter(Printer[PrinterConfig], Scheduled):
    """
    Complete printer implementation

    Demonstrates:
    - Dependency injection (protocols, bus, state)
    - Event handling (updates state)
    - Multiple protocols (HTTP + WebSocket)
    - Scheduled tasks (heartbeat, reporting)
    - Lifecycle management
    - Transactions (atomic state updates)
    """

    def __init__(
        self,
        config: PrinterConfig = Injected,
        http: HTTPPollingProtocol = Injected,
        ws: WebSocketProtocol = Injected,
        bus: MessageBus = Injected,
    ):
        Scheduled.__init__(self)
        self.config = config
        self.http = http
        self.ws = ws
        self.bus = bus
        self.state = PrinterState()

    async def start(self):
        """Start printer"""
        logger.info(f"[PRINTER] Starting '{self.config.name}'...")

        # Subscribe to events
        self.bus.subscribe(self)

        # Start protocols
        await self.http.start()
        await self.ws.start()

        # Start scheduled tasks
        await self.start_scheduled()

        logger.info(f"[PRINTER] '{self.config.name}' started")

    async def stop(self):
        """Stop printer"""
        logger.info(f"[PRINTER] Stopping '{self.config.name}'...")

        # Stop scheduled tasks
        await self.stop_scheduled()

        # Stop protocols
        await self.ws.stop()
        await self.http.stop()

        # Unsubscribe from events
        self.bus.unsubscribe(self)

        logger.info(f"[PRINTER] '{self.config.name}' stopped")

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def handle_temperature_changed(self, event: TemperatureChanged):
        """Handle temperature updates"""
        # Use atomic transaction for state update
        with atomic(self.state):
            self.state.temperature = event.actual
            self.state.target_temp = event.target

        logger.info(
            f"[PRINTER] Temperature: {event.actual:.1f}°C / {event.target:.1f}°C"
        )

    def handle_status_changed(self, event: StatusChanged):
        """Handle status updates"""
        old_status = self.state.status
        self.state.status = event.status

        if old_status != event.status:
            logger.info(f"[PRINTER] Status changed: {old_status} -> {event.status}")

    def handle_job_progress(self, event: JobProgress):
        """Handle job progress updates"""
        self.state.progress = event.percent

        logger.info(
            f"[PRINTER] Job progress: {event.percent:.1f}% "
            f"(~{event.time_remaining}s remaining)"
        )

    async def handle_gcode_command(self, event: GcodeCommand):
        """Handle gcode commands"""
        logger.info(f"[PRINTER] Executing gcode: {event.commands}")

        for cmd in event.commands:
            await asyncio.sleep(0.1)
            logger.info(f"[PRINTER]   > {cmd}")

    # ========================================================================
    # Scheduled Tasks
    # ========================================================================

    @on_startup
    async def initialize(self):
        """Called once on startup"""
        logger.info(f"[PRINTER] Initializing '{self.config.name}'...")
        await asyncio.sleep(0.3)
        logger.info("[PRINTER] Initialization complete")

    @interval(10, run_immediately=True)
    async def send_heartbeat(self):
        """Send periodic heartbeat"""
        logger.info(f"[PRINTER] ❤ Heartbeat - {self.state}")

    @interval(timedelta(seconds=15))
    async def report_status(self):
        """Periodic status report"""
        logger.info(
            f"[PRINTER] 📊 Status Report\n"
            f"  Name: {self.config.name}\n"
            f"  State: {self.state}\n"
            f"  Temperature: {self.state.temperature:.1f}°C "
            f"(target: {self.state.target_temp:.1f}°C)\n"
            f"  Status: {self.state.status}\n"
            f"  Progress: {self.state.progress:.1f}%"
        )

    @on_shutdown
    async def finalize(self):
        """Called once on shutdown"""
        logger.info(f"[PRINTER] Finalizing '{self.config.name}'...")
        await asyncio.sleep(0.2)
        logger.info("[PRINTER] Finalize complete")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("  Example 4: Complete Printer Integration")
    print("=" * 70)
    print()

    # Create container
    container = Container()

    # Register config
    config = PrinterConfig(
        name="Virtual Printer",
        host="printer.local",
        port=80,
        api_key="secret123"
    )
    container.register_instance(PrinterConfig, config)

    # Register infrastructure
    container.register(MessageBus)

    # Register protocols
    container.register(HTTPPollingProtocol)
    container.register(WebSocketProtocol)

    # Register printer
    container.register(VirtualPrinter)

    print("--- Starting Printer ---\n")

    # Resolve and start printer
    printer = await container.resolve(VirtualPrinter)
    bus = await container.resolve(MessageBus)

    await printer.start()

    print("\n--- Running (30 seconds) ---\n")

    # Let it run for 30 seconds
    await asyncio.sleep(30)

    print("\n--- Sending Gcode Command ---\n")

    # Send a gcode command
    await bus.publish(GcodeCommand(commands=["G28", "G0 Z10", "M104 S200"]))

    await asyncio.sleep(5)

    print("\n--- Stopping Printer ---\n")

    # Stop printer
    await printer.stop()

    # Shutdown container
    await container.shutdown()

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
