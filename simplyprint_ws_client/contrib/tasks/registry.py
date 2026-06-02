"""The task registry: declare once at import time, wire once at startup.

Every recurring or triggerable task is declared with the
:meth:`REGISTRY.periodic <TaskRegistry.periodic>` decorator next to the coroutine
that implements it. The registry is a passive inventory -- it collects
:class:`~simplyprint_ws_client.contrib.tasks.spec.TaskSpec` s and never runs
anything. At startup the app's single
:class:`~simplyprint_ws_client.contrib.tasks.scheduler.SchedulerService` walks the
registry and arranges the firing; that service is the *only* code that touches
the scheduler backend.

This inverts today's pain (``add_job(...)`` calls and interval literals scattered
through startup code, with brand-specific jobs living inside a shared scheduler):
a task is declared where it lives -- a brand's task module under
``printers/<brand>/`` for brand work, an app module for app work -- so the shared
scheduler never names a brand. The difference between brands is purely *which
specs get registered*, which is the hook-not-branch rule.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Dict, Optional, ValuesView

from simplyprint_ws_client.contrib.tasks.spec import TaskFn, TaskSpec


class TaskRegistry:
    """A passive, ordered inventory of :class:`TaskSpec` s.

    Registration happens at import time (a decorator runs when its task module is
    imported), so the app assembles its full task set simply by importing the
    task modules it wants -- a brand contributes tasks by being imported, never
    by editing shared code.
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


#: The process-wide registry. Task modules decorate against this at import time;
#: the app's SchedulerService reads it once at startup. Registration is global --
#: it is import-time and stateless, like every task framework -- but runtime
#: state lives on the wired StatusRegistry, never a global.
REGISTRY = TaskRegistry()
