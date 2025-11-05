"""
Example 3: Scheduling

Demonstrates @interval, @on_startup, @on_shutdown decorators.

Run: python -m examples.03_scheduling
"""

import asyncio
import logging
from datetime import timedelta, datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spc_core import Scheduled, interval, on_startup, on_shutdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Interval
# ============================================================================

class HealthChecker(Scheduled):
    """Checks health at regular intervals"""

    def __init__(self):
        super().__init__()
        self.checks = 0

    @interval(2)  # Every 2 seconds
    async def check_health(self):
        self.checks += 1
        logger.info(f"[HEALTH] Check #{self.checks} - System healthy")


# ============================================================================
# Example 2: Multiple Intervals
# ============================================================================

class StatusMonitor(Scheduled):
    """Monitors different aspects at different intervals"""

    def __init__(self):
        super().__init__()
        self.temperature = 25.0
        self.memory_mb = 512

    @interval(1)
    async def monitor_temperature(self):
        """Fast polling for temperature"""
        import random
        self.temperature += random.uniform(-1, 1)
        logger.info(f"[MONITOR] Temperature: {self.temperature:.1f}°C")

    @interval(5)
    async def monitor_memory(self):
        """Slower polling for memory"""
        import random
        self.memory_mb += random.randint(-50, 50)
        logger.info(f"[MONITOR] Memory: {self.memory_mb}MB")

    @interval(10)
    async def report_summary(self):
        """Periodic summary report"""
        logger.info(
            f"[MONITOR] Summary: Temp={self.temperature:.1f}°C, "
            f"Memory={self.memory_mb}MB"
        )


# ============================================================================
# Example 3: Lifecycle Hooks
# ============================================================================

class DatabaseService(Scheduled):
    """Service with startup and shutdown hooks"""

    def __init__(self):
        super().__init__()
        self.connected = False
        self.queries = 0

    @on_startup
    async def initialize(self):
        """Run once on startup"""
        logger.info("[DB] Initializing...")
        await asyncio.sleep(0.5)
        self.connected = True
        logger.info("[DB] Connected to database")

    @interval(3)
    async def sync_data(self):
        """Periodic data sync"""
        if not self.connected:
            logger.warning("[DB] Not connected!")
            return

        self.queries += 1
        logger.info(f"[DB] Syncing data... (query #{self.queries})")

    @on_shutdown
    async def cleanup(self):
        """Run once on shutdown"""
        logger.info("[DB] Closing connections...")
        await asyncio.sleep(0.3)
        self.connected = False
        logger.info(f"[DB] Shutdown complete (executed {self.queries} queries)")


# ============================================================================
# Example 4: Immediate Execution
# ============================================================================

class HeartbeatService(Scheduled):
    """Sends heartbeat immediately and periodically"""

    def __init__(self):
        super().__init__()
        self.heartbeats = 0
        self.start_time = None

    @on_startup
    async def record_start_time(self):
        self.start_time = datetime.now()
        logger.info(f"[HEARTBEAT] Started at {self.start_time}")

    @interval(timedelta(seconds=4), run_immediately=True)
    async def send_heartbeat(self):
        """Send heartbeat immediately, then every 4 seconds"""
        self.heartbeats += 1

        if self.start_time:
            uptime = (datetime.now() - self.start_time).seconds
            logger.info(f"[HEARTBEAT] Beat #{self.heartbeats} (uptime: {uptime}s)")
        else:
            logger.info(f"[HEARTBEAT] Beat #{self.heartbeats}")


# ============================================================================
# Example 5: Error Handling
# ============================================================================

class FlakyService(Scheduled):
    """Service that occasionally fails"""

    def __init__(self):
        super().__init__()
        self.attempts = 0
        self.failures = 0

    @interval(2)
    async def unreliable_task(self):
        """Task that sometimes fails"""
        self.attempts += 1

        # Fail every 3rd attempt
        if self.attempts % 3 == 0:
            self.failures += 1
            logger.error(f"[FLAKY] Attempt #{self.attempts} FAILED!")
            raise RuntimeError("Simulated failure")

        logger.info(f"[FLAKY] Attempt #{self.attempts} succeeded")

    @on_shutdown
    async def report_stats(self):
        logger.info(
            f"[FLAKY] Stats: {self.attempts} attempts, "
            f"{self.failures} failures, "
            f"{self.attempts - self.failures} successes"
        )


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("  Example 3: Scheduling")
    print("=" * 70)
    print()

    print("--- Basic Interval ---\n")

    # Start health checker
    health = HealthChecker()
    await health.start_scheduled()

    # Run for 5 seconds
    await asyncio.sleep(5)

    # Stop
    await health.stop_scheduled()
    logger.info(f"Completed {health.checks} health checks\n")

    print("\n--- Multiple Intervals ---\n")

    # Start status monitor (multiple intervals)
    monitor = StatusMonitor()
    await monitor.start_scheduled()

    # Run for 12 seconds to see different intervals
    await asyncio.sleep(12)

    await monitor.stop_scheduled()

    print("\n--- Lifecycle Hooks ---\n")

    # Start database service with lifecycle
    db = DatabaseService()
    await db.start_scheduled()

    # Run for 10 seconds
    await asyncio.sleep(10)

    # Stop (will call cleanup)
    await db.stop_scheduled()

    print("\n--- Immediate Execution ---\n")

    # Start heartbeat (runs immediately)
    heartbeat = HeartbeatService()
    await heartbeat.start_scheduled()

    # Run for 10 seconds
    await asyncio.sleep(10)

    await heartbeat.stop_scheduled()

    print("\n--- Error Handling ---\n")

    # Start flaky service (handles errors gracefully)
    flaky = FlakyService()
    await flaky.start_scheduled()

    # Run for 8 seconds (will have some failures)
    await asyncio.sleep(8)

    await flaky.stop_scheduled()

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
