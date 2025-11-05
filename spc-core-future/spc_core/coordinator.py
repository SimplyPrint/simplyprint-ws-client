"""
Update Coordinator

Inspired by Home Assistant's DataUpdateCoordinator pattern.
Provides centralized device updates with error handling and change tracking.
"""

import asyncio
import logging
from datetime import timedelta
from typing import TypeVar, Generic, Callable, Awaitable, Optional
from dataclasses import dataclass, field as dataclass_field
from enum import Enum, auto

from .state import ExternalStateModel, ChangedFields
from .events import MessageBus, Event

__all__ = [
    "CoordinatorStatus",
    "CoordinatorState",
    "UpdateFailed",
    "UpdateSucceeded",
    "UpdateCoordinator",
]

logger = logging.getLogger(__name__)

# ============================================================================
# Types
# ============================================================================

StateT = TypeVar("StateT", bound=ExternalStateModel)
UpdateMethod = Callable[[], Awaitable[StateT]]


class CoordinatorStatus(Enum):
    """Status of the coordinator"""
    INITIALIZING = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPED = auto()


# ============================================================================
# Events
# ============================================================================

@dataclass
class CoordinatorState(Event):
    """
    Event emitted when coordinator state changes.

    Example:
        >>> @handles(CoordinatorState)
        ... class StatusDisplay:
        ...     def handle_coordinator_state(self, event: CoordinatorState):
        ...         print(f"Status: {event.status}")
    """
    coordinator_id: str
    status: CoordinatorStatus
    error: Optional[Exception] = None


@dataclass
class UpdateFailed(Event):
    """
    Event emitted when an update fails.

    Example:
        >>> @handles(UpdateFailed)
        ... class ErrorLogger:
        ...     def handle_update_failed(self, event: UpdateFailed):
        ...         logger.error(f"Update failed: {event.error}")
    """
    coordinator_id: str
    error: Exception
    attempt: int


@dataclass
class UpdateSucceeded(Event):
    """
    Event emitted when an update succeeds.

    Includes the changed fields for efficient downstream processing.

    Example:
        >>> @handles(UpdateSucceeded)
        ... class ChangeHandler:
        ...     def handle_update_succeeded(self, event: UpdateSucceeded):
        ...         if event.changes:
        ...             print(f"Fields changed: {event.changes.keys()}")
    """
    coordinator_id: str
    changes: ChangedFields
    state: ExternalStateModel


# ============================================================================
# Update Coordinator
# ============================================================================

class UpdateCoordinator(Generic[StateT]):
    """
    Coordinates device updates with error handling and change tracking.

    Inspired by Home Assistant's DataUpdateCoordinator. Provides:
    - Centralized update logic
    - Automatic retries with exponential backoff
    - Change tracking and event emission
    - Manual refresh support
    - Error recovery

    Example:
        >>> async def fetch_printer_state() -> PrinterState:
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get("http://printer/state") as resp:
        ...             data = await resp.json()
        ...             return PrinterState(**data)
        ...
        >>> coordinator = UpdateCoordinator(
        ...     name="printer",
        ...     update_method=fetch_printer_state,
        ...     update_interval=timedelta(seconds=5),
        ...     bus=message_bus
        ... )
        ...
        >>> await coordinator.start()
        >>> # Updates happen automatically every 5 seconds
        >>> # Access current state
        >>> print(coordinator.state.status)
        >>> # Force immediate update
        >>> await coordinator.refresh()
    """

    def __init__(
        self,
        name: str,
        update_method: UpdateMethod[StateT],
        update_interval: Optional[timedelta] = None,
        bus: Optional[MessageBus] = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize coordinator.

        Args:
            name: Unique identifier for this coordinator
            update_method: Async function that fetches new state
            update_interval: How often to update (None = manual only)
            bus: Optional message bus for event emission
            max_retries: Maximum retry attempts on failure
            retry_backoff: Backoff multiplier for retries (exponential)
        """
        self.name = name
        self._update_method = update_method
        self._update_interval = update_interval
        self._bus = bus
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

        self._state: Optional[StateT] = None
        self._status = CoordinatorStatus.INITIALIZING
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._refresh_event = asyncio.Event()
        self._consecutive_failures = 0

        self.logger = logger.getChild(name)

    @property
    def state(self) -> Optional[StateT]:
        """Current state (may be None if not initialized)"""
        return self._state

    @property
    def status(self) -> CoordinatorStatus:
        """Current coordinator status"""
        return self._status

    async def start(self) -> None:
        """
        Start the coordinator.

        Performs initial update and starts update loop if interval configured.
        """
        if self._task is not None:
            raise RuntimeError(f"Coordinator {self.name} already running")

        self.logger.info("Starting coordinator")
        self._stop_event.clear()

        # Perform initial update
        await self._perform_update()

        # Start update loop if interval configured
        if self._update_interval is not None:
            self._task = asyncio.create_task(self._update_loop())

        self.logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator"""
        if self._task is None:
            return

        self.logger.info("Stopping coordinator")
        self._stop_event.set()

        if self._task:
            await self._task
            self._task = None

        self._update_status(CoordinatorStatus.STOPPED)
        self.logger.info("Coordinator stopped")

    async def refresh(self) -> bool:
        """
        Force an immediate update.

        Returns:
            True if update succeeded, False if failed

        Example:
            >>> if await coordinator.refresh():
            ...     print("Update successful")
            ... else:
            ...     print("Update failed")
        """
        self.logger.debug("Manual refresh requested")
        return await self._perform_update()

    async def _update_loop(self) -> None:
        """Main update loop"""
        while not self._stop_event.is_set():
            try:
                # Wait for interval or manual refresh
                interval_seconds = self._update_interval.total_seconds()

                try:
                    await asyncio.wait_for(
                        self._refresh_event.wait(),
                        timeout=interval_seconds
                    )
                    self._refresh_event.clear()
                except asyncio.TimeoutError:
                    pass

                # Perform update
                await self._perform_update()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Unexpected error in update loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on errors

    async def _perform_update(self) -> bool:
        """
        Perform a single update with retry logic.

        Returns:
            True if update succeeded, False if all retries failed
        """
        attempt = 0
        last_error = None

        while attempt <= self._max_retries:
            try:
                # Fetch new state
                new_state = await self._update_method()

                # Handle first update
                if self._state is None:
                    self._state = new_state
                    self._update_status(CoordinatorStatus.RUNNING)
                    self._consecutive_failures = 0

                    # Emit success with no changes (initial state)
                    if self._bus:
                        await self._bus.publish(UpdateSucceeded(
                            coordinator_id=self.name,
                            changes={},
                            state=self._state
                        ))

                    self.logger.info("Initial state loaded")
                    return True

                # Update existing state and track changes
                changes = self._state.update_from(new_state)

                # Reset failure counter
                self._consecutive_failures = 0
                self._update_status(CoordinatorStatus.RUNNING)

                # Emit success event
                if self._bus:
                    await self._bus.publish(UpdateSucceeded(
                        coordinator_id=self.name,
                        changes=changes,
                        state=self._state
                    ))

                if changes:
                    self.logger.debug(f"State updated: {len(changes)} field(s) changed")
                else:
                    self.logger.debug("State unchanged")

                return True

            except Exception as e:
                last_error = e
                attempt += 1
                self._consecutive_failures += 1

                self.logger.warning(
                    f"Update failed (attempt {attempt}/{self._max_retries + 1}): {e}"
                )

                # Emit failure event
                if self._bus:
                    await self._bus.publish(UpdateFailed(
                        coordinator_id=self.name,
                        error=e,
                        attempt=attempt
                    ))

                # Exponential backoff for retries
                if attempt <= self._max_retries:
                    backoff = self._retry_backoff ** (attempt - 1)
                    await asyncio.sleep(backoff)

        # All retries failed
        self._update_status(CoordinatorStatus.ERROR, last_error)
        self.logger.error(f"Update failed after {self._max_retries + 1} attempts")
        return False

    def _update_status(self, status: CoordinatorStatus, error: Optional[Exception] = None) -> None:
        """Update status and emit event"""
        old_status = self._status
        self._status = status

        if old_status != status:
            self.logger.info(f"Status changed: {old_status} -> {status}")

            if self._bus:
                asyncio.create_task(self._bus.publish(CoordinatorState(
                    coordinator_id=self.name,
                    status=status,
                    error=error
                )))

    def request_refresh(self) -> None:
        """
        Request a refresh on next update cycle.

        Non-blocking alternative to refresh(). Useful when you want
        to trigger an update but don't want to wait for it.

        Example:
            >>> coordinator.request_refresh()
            >>> # Update will happen on next cycle
        """
        self._refresh_event.set()
