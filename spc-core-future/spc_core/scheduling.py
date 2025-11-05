"""
Scheduling - Simple Periodic Tasks

Clean API for scheduled execution.
"""

__all__ = ["Scheduled", "interval", "on_startup", "on_shutdown"]

from typing import Callable, Union, Dict
from functools import wraps
from datetime import timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


def interval(
    seconds: Union[int, float, timedelta],
    *,
    run_immediately: bool = False,
):
    """
    Decorator: Run method at regular intervals.

    Args:
        seconds: Interval in seconds (or timedelta)
        run_immediately: If True, run once immediately before first interval

    Usage:
        class MyService(Scheduled):
            @interval(5)
            async def check_status(self):
                # Runs every 5 seconds
                pass

            @interval(timedelta(minutes=1), run_immediately=True)
            async def heartbeat(self):
                # Runs immediately, then every minute
                pass
    """
    if isinstance(seconds, timedelta):
        seconds = seconds.total_seconds()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_interval = True  # type: ignore
        wrapper._interval_seconds = float(seconds)  # type: ignore
        wrapper._interval_immediate = run_immediately  # type: ignore

        return wrapper
    return decorator


def on_startup(func: Callable):
    """
    Decorator: Run method once on startup.

    Usage:
        class MyService(Scheduled):
            @on_startup
            async def initialize(self):
                # Runs once when service starts
                pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_startup = True  # type: ignore
    return wrapper


def on_shutdown(func: Callable):
    """
    Decorator: Run method once on shutdown.

    Usage:
        class MyService(Scheduled):
            @on_shutdown
            async def cleanup(self):
                # Runs once when service stops
                pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_shutdown = True  # type: ignore
    return wrapper


class Scheduled:
    """
    Mixin to enable @interval, @on_startup, @on_shutdown decorators.

    Usage:
        class MyService(Scheduled):
            async def start(self):
                await self.start_scheduled()

            async def stop(self):
                await self.stop_scheduled()

            @interval(5)
            async def poll(self):
                print("Polling...")

            @on_startup
            async def init(self):
                print("Starting!")

            @on_shutdown
            async def cleanup(self):
                print("Cleaning up!")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._scheduled_running = False

    async def start_scheduled(self):
        """Start all scheduled tasks"""
        if self._scheduled_running:
            logger.warning(f"{self.__class__.__name__}: Already running")
            return

        self._scheduled_running = True

        # Run startup hooks
        for hook in self._discover_hooks("_is_startup"):
            try:
                await self._call(hook)
            except Exception as e:
                logger.error(f"Error in startup hook {hook.__name__}: {e}", exc_info=True)

        # Start interval tasks
        for method, metadata in self._discover_intervals():
            task = asyncio.create_task(
                self._run_interval(
                    method,
                    metadata["seconds"],
                    metadata["immediate"]
                ),
                name=method.__name__
            )
            self._scheduled_tasks[method.__name__] = task

        logger.info(
            f"{self.__class__.__name__}: Started {len(self._scheduled_tasks)} task(s)"
        )

    async def stop_scheduled(self):
        """Stop all scheduled tasks"""
        if not self._scheduled_running:
            return

        self._scheduled_running = False

        # Run shutdown hooks
        for hook in self._discover_hooks("_is_shutdown"):
            try:
                await self._call(hook)
            except Exception as e:
                logger.error(f"Error in shutdown hook {hook.__name__}: {e}", exc_info=True)

        # Cancel tasks
        for task in self._scheduled_tasks.values():
            if not task.done():
                task.cancel()

        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks.values(), return_exceptions=True)

        self._scheduled_tasks.clear()

        logger.info(f"{self.__class__.__name__}: Stopped")

    async def _run_interval(self, method: Callable, seconds: float, immediate: bool):
        """Run an interval task"""
        try:
            if immediate:
                await self._call(method)

            while self._scheduled_running:
                await asyncio.sleep(seconds)

                if not self._scheduled_running:
                    break

                try:
                    await self._call(method)
                except Exception as e:
                    logger.error(
                        f"Error in interval {method.__name__}: {e}",
                        exc_info=True
                    )

        except asyncio.CancelledError:
            logger.debug(f"Interval {method.__name__} cancelled")
            raise

    async def _call(self, method: Callable):
        """Call method (async or sync)"""
        if asyncio.iscoroutinefunction(method):
            await method()
        else:
            method()

    def _discover_intervals(self):
        """Discover all @interval methods"""
        intervals = []

        for attr_name in dir(self):
            try:
                attr = getattr(self.__class__, attr_name, None)
            except AttributeError:
                continue

            if not hasattr(attr, "_is_interval"):
                continue

            method = getattr(self, attr_name)
            metadata = {
                "seconds": attr._interval_seconds,
                "immediate": attr._interval_immediate,
            }

            intervals.append((method, metadata))

        return intervals

    def _discover_hooks(self, marker: str):
        """Discover hooks by marker attribute"""
        hooks = []

        for attr_name in dir(self):
            try:
                attr = getattr(self.__class__, attr_name, None)
            except AttributeError:
                continue

            if not hasattr(attr, marker):
                continue

            hooks.append(getattr(self, attr_name))

        return hooks
