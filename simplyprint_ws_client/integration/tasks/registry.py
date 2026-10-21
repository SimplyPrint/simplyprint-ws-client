"""Task-spec inventory assembled explicitly by an application at startup."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Dict, Optional, ValuesView

from simplyprint_ws_client.integration.tasks.spec import TaskFn, TaskSpec

if TYPE_CHECKING:
    from simplyprint_ws_client.core.scope import Scope
    from simplyprint_ws_client.integration.tasks.scheduler import SchedulerService

__all__ = ["ScopedTaskRegistry", "TaskRegistry"]


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
        # Copy-on-write: the scheduler iterates specs() while a live add or
        # removal may land from another thread.
        specs = dict(self._specs)
        specs[spec.name] = spec
        self._specs = specs

    def remove(self, spec: TaskSpec) -> bool:
        """Drop ``spec`` if it is the one registered under its name."""
        if self._specs.get(spec.name) is not spec:
            return False
        specs = dict(self._specs)
        del specs[spec.name]
        self._specs = specs
        return True

    def get(self, name: str) -> TaskSpec:
        """The spec registered under ``name`` (``KeyError`` if absent)."""
        return self._specs[name]

    def specs_by_name(self) -> Dict[str, TaskSpec]:
        """A snapshot of ``{name: spec}``."""
        return dict(self._specs)

    def specs(self) -> ValuesView[TaskSpec]:
        """Every registered spec, in registration order."""
        return self._specs.values()


class ScopedTaskRegistry(TaskRegistry):
    """The registrar one mounted domain sees.

    Every task registered through it goes straight onto ``scheduler`` under
    ``scope`` (live, before or after start) and is removed when the scope
    closes. Names are qualified as ``"<prefix>.<name>"`` so two domains cannot
    collide; a name that already starts with ``"<prefix>."`` is kept as is.
    """

    def __init__(
        self, scheduler: "SchedulerService", scope: "Scope", *, prefix: str
    ) -> None:
        super().__init__()
        self._scheduler = scheduler
        self._scope = scope
        self._prefix = prefix

    def qualify(self, name: str) -> str:
        head = self._prefix + "."
        return name if name.startswith(head) else head + name

    def register(self, spec: TaskSpec) -> None:
        scoped = replace(spec, name=self.qualify(spec.name))
        self._scheduler.add(scoped, self._scope)
        super().register(scoped)
