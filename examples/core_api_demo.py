"""
Complete demonstration of the new Core API design.

This example shows:
- Dependency injection with Depends()
- Event-driven protocol composition
- @tick scheduling
- inbox/outbox contracts
- Type-safe event routing

Run with: python -m examples.core_api_demo
"""

import asyncio
import logging
from datetime import timedelta
from typing import Optional
from dataclasses import dataclass

# Core API imports
from simplyprint_ws_client.core.di import Depends, DIContainer, Injectable
from simplyprint_ws_client.core.protocols import inbox, outbox, Protocol, EventRouter
from simplyprint_ws_client.core.decorators import tick, on_init, on_destroy, SchedulerMixin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Define Events (typed Pydantic-style models)
# ============================================================================

@dataclass
class TemperatureChangedEvent:
    """Event: Temperature has changed"""

    tool: int
    actual: float
    target: float


@dataclass
class StatusChangedEvent:
    """Event: Printer status has changed"""

    status: str


@dataclass
class GcodeCommandEvent:
    """Event: Gcode command to execute"""

    commands: list[str]


@dataclass
class JobProgressEvent:
    """Event: Print job progress update"""

    progress: float  # 0.0 to 100.0
    time_remaining: Optional[int] = None


# ============================================================================
# 2. Define Infrastructure (HTTP, MQTT clients - simulated)
# ============================================================================


class HTTPClient(Injectable):
    """Simulated HTTP client"""

    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        logger.info(f"HTTPClient initialized with base_url={base_url}")

    async def get(self, path: str) -> dict:
        """Simulate HTTP GET"""
        await asyncio.sleep(0.1)  # Simulate network delay
        return {"status": "ok", "data": {"temperature": 25.0}}


class MQTTClient(Injectable):
    """Simulated MQTT client"""

    def __init__(self, host: str = "localhost", port: int = 1883):
        self.host = host
        self.port = port
        self._callbacks: dict[str, callable] = {}
        logger.info(f"MQTTClient initialized: {host}:{port}")

    async def connect(self):
        """Simulate MQTT connection"""
        await asyncio.sleep(0.1)
        logger.info("MQTT connected")

    async def subscribe(self, topic: str, callback: callable):
        """Simulate MQTT subscribe"""
        self._callbacks[topic] = callback
        logger.info(f"MQTT subscribed to {topic}")

    async def publish(self, topic: str, payload: str):
        """Simulate MQTT publish"""
        logger.info(f"MQTT publish to {topic}: {payload}")

    async def simulate_message(self, topic: str, payload: str):
        """Simulate receiving a message"""
        if topic in self._callbacks:
            await self._callbacks[topic](topic, payload)


# ============================================================================
# 3. Define Protocols (consume external events, emit internal events)
# ============================================================================


@outbox(TemperatureChangedEvent, StatusChangedEvent)
class HTTPPollingProtocol(Protocol, SchedulerMixin):
    """
    Protocol: Poll printer via HTTP REST API

    Emits: TemperatureChangedEvent, StatusChangedEvent
    """

    http: HTTPClient = Depends()
    event_router: EventRouter = Depends()

    def __init__(self):
        SchedulerMixin.__init__(self)
        self._last_temp = 0.0

    async def start(self):
        logger.info("HTTPPollingProtocol starting...")
        await self.start_scheduled_tasks()

    async def stop(self):
        logger.info("HTTPPollingProtocol stopping...")
        await self.stop_scheduled_tasks()

    @tick(2, immediate=True)  # Poll every 2 seconds
    async def poll_temperature(self):
        """Poll temperature from HTTP API"""
        response = await self.http.get("/api/temperature")
        temp = response["data"]["temperature"]

        # Only emit if changed
        if temp != self._last_temp:
            logger.info(f"Temperature changed: {self._last_temp} -> {temp}")
            self._last_temp = temp

            await self.event_router.emit(
                TemperatureChangedEvent(tool=0, actual=temp, target=200.0),
                source=self.__class__.__name__,
            )

    @tick(5)  # Poll every 5 seconds
    async def poll_status(self):
        """Poll status from HTTP API"""
        logger.info("Polling status...")
        await self.event_router.emit(
            StatusChangedEvent(status="printing"),
            source=self.__class__.__name__,
        )


@outbox(TemperatureChangedEvent, StatusChangedEvent, JobProgressEvent)
class MQTTProtocol(Protocol):
    """
    Protocol: Receive real-time updates via MQTT

    Emits: TemperatureChangedEvent, StatusChangedEvent, JobProgressEvent
    """

    mqtt: MQTTClient = Depends()
    event_router: EventRouter = Depends()

    async def start(self):
        logger.info("MQTTProtocol starting...")
        await self.mqtt.connect()
        await self.mqtt.subscribe("printer/updates", self.on_mqtt_message)

    async def stop(self):
        logger.info("MQTTProtocol stopping...")

    async def on_mqtt_message(self, topic: str, payload: str):
        """Handle incoming MQTT message"""
        import json

        logger.info(f"MQTT message received: {topic}")
        data = json.loads(payload)

        # Parse and emit typed events
        if "temperature" in data:
            await self.event_router.emit(
                TemperatureChangedEvent(
                    tool=data.get("tool", 0),
                    actual=data["temperature"],
                    target=data.get("target", 0),
                ),
                source=self.__class__.__name__,
            )

        if "progress" in data:
            await self.event_router.emit(
                JobProgressEvent(
                    progress=data["progress"],
                    time_remaining=data.get("time_remaining"),
                ),
                source=self.__class__.__name__,
            )


# ============================================================================
# 4. Define Printer (business logic layer)
# ============================================================================


@inbox(TemperatureChangedEvent, StatusChangedEvent, JobProgressEvent, GcodeCommandEvent)
@outbox()  # Could emit events to other systems
class DemoPrinter(SchedulerMixin):
    """
    Demo printer that:
    - Receives events from protocols
    - Updates internal state
    - Emits events to SimplyPrint (not shown)

    Composes:
    - HTTPPollingProtocol (for status)
    - MQTTProtocol (for real-time updates)
    """

    # Dependencies injected
    http_protocol: HTTPPollingProtocol = Depends()
    mqtt_protocol: MQTTProtocol = Depends()

    def __init__(self):
        SchedulerMixin.__init__(self)
        self.temperature = 0.0
        self.status = "idle"
        self.job_progress = 0.0

    async def start(self):
        """Start the printer"""
        logger.info("=== DemoPrinter starting ===")

        # Start protocols
        await self.http_protocol.start()
        await self.mqtt_protocol.start()

        # Start own scheduled tasks
        await self.start_scheduled_tasks()

        logger.info("=== DemoPrinter started ===")

    async def stop(self):
        """Stop the printer"""
        logger.info("=== DemoPrinter stopping ===")

        await self.stop_scheduled_tasks()

        await self.http_protocol.stop()
        await self.mqtt_protocol.stop()

        logger.info("=== DemoPrinter stopped ===")

    # ========================================================================
    # Event Handlers (inbox)
    # ========================================================================

    async def on_temperature_changed(self, event: TemperatureChangedEvent):
        """Handle temperature change from any protocol"""
        self.temperature = event.actual
        logger.info(
            f"[PRINTER] Temperature updated: tool={event.tool}, "
            f"actual={event.actual}, target={event.target}"
        )

    async def on_status_changed(self, event: StatusChangedEvent):
        """Handle status change"""
        self.status = event.status
        logger.info(f"[PRINTER] Status updated: {event.status}")

    async def on_job_progress(self, event: JobProgressEvent):
        """Handle job progress update"""
        self.job_progress = event.progress
        logger.info(f"[PRINTER] Job progress: {event.progress}%")

    async def on_gcode_command(self, event: GcodeCommandEvent):
        """Handle gcode command (e.g., from SimplyPrint)"""
        logger.info(f"[PRINTER] Executing gcode: {event.commands}")
        # Would send to printer here

    # ========================================================================
    # Scheduled Tasks
    # ========================================================================

    @tick(10)
    async def send_heartbeat(self):
        """Send periodic heartbeat to SimplyPrint"""
        logger.info(f"[PRINTER] Heartbeat - temp={self.temperature}, status={self.status}")

    @on_init
    async def on_printer_init(self):
        """Called after all scheduled tasks start"""
        logger.info("[PRINTER] Initialization complete!")

    @on_destroy
    async def on_printer_destroy(self):
        """Called before scheduled tasks stop"""
        logger.info("[PRINTER] Cleanup in progress...")


# ============================================================================
# 5. Main Application
# ============================================================================


async def main():
    """Main application entry point"""

    logger.info("========================================")
    logger.info("  Core API Demo")
    logger.info("========================================")

    # 1. Create DI container
    container = DIContainer()

    # 2. Register dependencies
    container.register(HTTPClient)
    container.register(MQTTClient)
    container.register(EventRouter, singleton=True)
    container.register(HTTPPollingProtocol)
    container.register(MQTTProtocol)
    container.register(DemoPrinter)

    # 3. Resolve printer (this resolves entire dependency graph)
    logger.info("\n--- Resolving Dependencies ---")
    printer = await container.resolve(DemoPrinter)
    event_router = await container.resolve(EventRouter)

    # 4. Register components with event router
    logger.info("\n--- Registering Event Handlers ---")
    event_router.register(printer.http_protocol, name="HTTPPollingProtocol")
    event_router.register(printer.mqtt_protocol, name="MQTTProtocol")
    event_router.register(printer, name="DemoPrinter")

    # 5. Start printer
    logger.info("\n--- Starting Printer ---")
    await printer.start()

    # 6. Simulate some MQTT messages
    logger.info("\n--- Simulating MQTT Messages ---")
    await asyncio.sleep(3)

    mqtt_client = await container.resolve(MQTTClient)
    await mqtt_client.simulate_message(
        "printer/updates",
        '{"temperature": 210.5, "tool": 0, "target": 220.0}',
    )

    await asyncio.sleep(2)

    await mqtt_client.simulate_message(
        "printer/updates",
        '{"progress": 45.2, "time_remaining": 600}',
    )

    # 7. Let it run for a bit
    logger.info("\n--- Running for 15 seconds ---")
    await asyncio.sleep(15)

    # 8. Stop printer
    logger.info("\n--- Stopping Printer ---")
    await printer.stop()

    # 9. Cleanup
    await container.destroy_all()

    logger.info("\n========================================")
    logger.info("  Demo Complete")
    logger.info("========================================")


if __name__ == "__main__":
    asyncio.run(main())
