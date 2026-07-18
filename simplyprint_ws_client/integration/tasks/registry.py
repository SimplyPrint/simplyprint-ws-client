"""Task-spec inventory assembled explicitly by an application at startup."""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Dict, Optional, ValuesView

from simplyprint_ws_client.integration.tasks.spec import TaskFn, TaskSpec


class TaskRegistry:
    """A passive, ordered inventory of :class:`TaskSpec` s.

    Each application owns an instance and passes it to its task registrars and
    scheduler. Registries therefore cannot leak tasks across hosts or test runs.
    """

    def __init__(self) -> None:
        self._specs: Dict[str, TaskSpec] = {}

    def periodic(
        self,
        *,
        name: Optional[str] = None,
        interval: Optional[timedelta] = None,
        cron: Optional[str] = None,
        timeout: Optional[float] = None,
        max_instances: int = 1,
        coalesce: bool = True,
        misfire_grace_time: Optional[int] = 30,
        enabled: bool = True,
        run_at_startup: bool = False,
    ) -> Callable[[TaskFn], TaskFn]:
        """Decorator: register the wrapped coroutine as a task.

        ``interval`` / ``cron`` set the schedule -- omit both for an
        on-demand-only task. ``name`` defaults to the function's qualified name.
        Returns the original function unchanged, so the callable can still be
        imported and unit-tested directly.
        """

        def decorator(fn: TaskFn) -> TaskFn:
            self.register(
                TaskSpec(
                    name=name or fn.__qualname__,
                    fn=fn,
                    interval=interval,
                    cron=cron,
                    timeout=timeout,
                    max_instances=max_instances,
                    coalesce=coalesce,
                    misfire_grace_time=misfire_grace_time,
                    enabled=enabled,
                    run_at_startup=run_at_startup,
                )
            )
            return fn

        return decorator

    def register(self, spec: TaskSpec) -> None:
        """Add a fully-built spec. Raises on a duplicate name -- task names are a
        flat global namespace so on-demand triggers and logs are unambiguous."""
        if spec.name in self._specs:
            raise ValueError(f"duplicate task name: {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> TaskSpec:
        """The spec registered under ``name`` (``KeyError`` if absent)."""
        return self._specs[name]

    def specs(self) -> ValuesView[TaskSpec]:
        """Every registered spec, in registration order."""
        return self._specs.values()
