"""
Decorators for scheduling and lifecycle management.

Provides:
- @tick: Schedule periodic task execution
- @on_init: Run on component initialization
- @on_destroy: Run on component destruction
"""

__all__ = ["tick", "on_init", "on_destroy", "SchedulerMixin"]

from typing import Callable, Union, Optional, Dict, List
from functools import wraps
import asyncio
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


def tick(
    interval: Union[int, float, timedelta],
    *,
    immediate: bool = False,
    name: Optional[str] = None,
    retry_on_error: bool = True,
):
    """
    Schedule a method to run periodically.

    The method will be called every `interval` seconds/timedelta.
    If the method takes longer than `interval` to execute, the next call
    will be scheduled immediately after completion (no concurrent calls).

    Args:
        interval: Seconds between executions (int/float) or timedelta
        immediate: If True, run immediately on start (default: False)
        name: Optional name for the task (defaults to method name)
        retry_on_error: If True, continue scheduling even if method raises exception

    Usage:
        ```python
        class MyPrinter(SchedulerMixin):
            @tick(5)
            async def poll_status(self):
                # Runs every 5 seconds
                ...

            @tick(timedelta(minutes=1), immediate=True)
            async def send_heartbeat(self):
                # Runs every minute, immediately on start
                ...
        ```

    Note: The class must inherit from SchedulerMixin to activate scheduling.
    """

    # Convert to seconds
    if isinstance(interval, timedelta):
        interval_seconds = interval.total_seconds()
    else:
        interval_seconds = float(interval)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Mark function with scheduling metadata
        wrapper._is_scheduled = True  # type: ignore
        wrapper._schedule_interval = interval_seconds  # type: ignore
        wrapper._schedule_immediate = immediate  # type: ignore
        wrapper._schedule_name = name or func.__name__  # type: ignore
        wrapper._schedule_retry_on_error = retry_on_error  # type: ignore

        return wrapper

    return decorator


def on_init(func: Callable):
    """
    Mark a method to be called on component initialization.

    The method will be called by SchedulerMixin after all scheduled tasks are started.

    Usage:
        ```python
        class MyComponent(SchedulerMixin):
            @on_init
            async def initialize(self):
                # Called once on startup
                ...
        ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_init_hook = True  # type: ignore
    return wrapper


def on_destroy(func: Callable):
    """
    Mark a method to be called on component destruction.

    The method will be called by SchedulerMixin before scheduled tasks are cancelled.

    Usage:
        ```python
        class MyComponent(SchedulerMixin):
            @on_destroy
            async def cleanup(self):
                # Called once on shutdown
                ...
        ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_destroy_hook = True  # type: ignore
    return wrapper


class SchedulerMixin:
    """
    Mixin to enable @tick decorated methods.

    Usage:
        ```python
        class MyPrinter(Printer, SchedulerMixin):
            async def init(self):
                await self.start_scheduled_tasks()

            async def halt(self):
                await self.stop_scheduled_tasks()

            @tick(5)
            async def poll_temperature(self):
                ...
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._scheduled_running: bool = False

    async def start_scheduled_tasks(self):
        """
        Start all @tick decorated methods.

        This should be called from your init() or start() method.
        """
        if self._scheduled_running:
            logger.warning(f"{self.__class__.__name__}: Scheduled tasks already running")
            return

        self._scheduled_running = True

        # Discover and start scheduled methods
        scheduled_methods = self._discover_scheduled_methods()

        logger.info(
            f"{self.__class__.__name__}: Starting {len(scheduled_methods)} scheduled task(s)"
        )

        for method, metadata in scheduled_methods:
            task_name = metadata["name"]
            interval = metadata["interval"]
            immediate = metadata["immediate"]
            retry_on_error = metadata["retry_on_error"]

            logger.debug(
                f"Scheduling {self.__class__.__name__}.{task_name} "
                f"every {interval}s (immediate={immediate})"
            )

            task = asyncio.create_task(
                self._run_scheduled_task(method, interval, immediate, retry_on_error),
                name=task_name,
            )

            self._scheduled_tasks[task_name] = task

        # Call init hooks
        init_hooks = self._discover_init_hooks()
        for hook in init_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.error(
                    f"Error in init hook {hook.__name__}: {e}",
                    exc_info=True,
                )

    async def stop_scheduled_tasks(self):
        """
        Stop all scheduled tasks.

        This should be called from your halt() or stop() method.
        """
        if not self._scheduled_running:
            return

        self._scheduled_running = False

        # Call destroy hooks
        destroy_hooks = self._discover_destroy_hooks()
        for hook in destroy_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.error(
                    f"Error in destroy hook {hook.__name__}: {e}",
                    exc_info=True,
                )

        # Cancel all tasks
        logger.info(
            f"{self.__class__.__name__}: Stopping {len(self._scheduled_tasks)} scheduled task(s)"
        )

        for task_name, task in self._scheduled_tasks.items():
            if not task.done():
                logger.debug(f"Cancelling task {task_name}")
                task.cancel()

        # Wait for all tasks to complete
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks.values(), return_exceptions=True)

        self._scheduled_tasks.clear()

    async def _run_scheduled_task(
        self,
        method: Callable,
        interval: float,
        immediate: bool,
        retry_on_error: bool,
    ):
        """
        Run a scheduled task in a loop.

        Args:
            method: The method to call
            interval: Seconds between calls
            immediate: If True, call immediately before first sleep
            retry_on_error: If True, continue even if method raises exception
        """
        method_name = method.__name__

        try:
            # Immediate execution
            if immediate:
                try:
                    await self._call_method(method)
                except Exception as e:
                    logger.error(
                        f"Error in scheduled task {method_name} (immediate): {e}",
                        exc_info=True,
                    )

                    if not retry_on_error:
                        raise

            # Loop
            while self._scheduled_running:
                await asyncio.sleep(interval)

                if not self._scheduled_running:
                    break

                try:
                    await self._call_method(method)
                except Exception as e:
                    logger.error(
                        f"Error in scheduled task {method_name}: {e}",
                        exc_info=True,
                    )

                    if not retry_on_error:
                        raise

        except asyncio.CancelledError:
            logger.debug(f"Scheduled task {method_name} cancelled")
            raise

    async def _call_method(self, method: Callable):
        """Call a method (sync or async)"""
        if asyncio.iscoroutinefunction(method):
            await method()
        else:
            method()

    def _discover_scheduled_methods(self) -> List[tuple[Callable, dict]]:
        """
        Discover all @tick decorated methods.

        Returns:
            List of (method, metadata) tuples
        """
        scheduled = []

        for attr_name in dir(self):
            try:
                attr = getattr(self.__class__, attr_name, None)
            except AttributeError:
                continue

            if not hasattr(attr, "_is_scheduled"):
                continue

            method = getattr(self, attr_name)

            metadata = {
                "name": attr._schedule_name,
                "interval": attr._schedule_interval,
                "immediate": attr._schedule_immediate,
                "retry_on_error": attr._schedule_retry_on_error,
            }

            scheduled.append((method, metadata))

        return scheduled

    def _discover_init_hooks(self) -> List[Callable]:
        """Discover all @on_init decorated methods"""
        hooks = []

        for attr_name in dir(self):
            try:
                attr = getattr(self.__class__, attr_name, None)
            except AttributeError:
                continue

            if not hasattr(attr, "_is_init_hook"):
                continue

            hooks.append(getattr(self, attr_name))

        return hooks

    def _discover_destroy_hooks(self) -> List[Callable]:
        """Discover all @on_destroy decorated methods"""
        hooks = []

        for attr_name in dir(self):
            try:
                attr = getattr(self.__class__, attr_name, None)
            except AttributeError:
                continue

            if not hasattr(attr, "_is_destroy_hook"):
                continue

            hooks.append(getattr(self, attr_name))

        return hooks
