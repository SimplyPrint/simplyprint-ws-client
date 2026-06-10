"""The declarative unit of recurring / triggerable work: :class:`TaskSpec`.

A ``TaskSpec`` is *what* a task is and *how* it should run -- its name, the
coroutine that does the work, and the run policy (interval or cron, timeout,
overlap guards). It deliberately knows nothing about *when* it actually fires or
*who* fires it: a :class:`~simplyprint_ws_client.integration.tasks.scheduler.SchedulerService`
reads the spec and arranges the firing, and the same spec can be fired on demand.

This is the split APScheduler v4 itself adopts (Task vs Schedule vs Job);
modelling it here means our own code never re-couples them and a future v4 move
is confined to the scheduler. A spec names no brand and holds no live object --
it is pure data plus a coroutine, registered once at import time via
:meth:`~simplyprint_ws_client.integration.tasks.registry.TaskRegistry.periodic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Optional

#: A task body. Receives the runtime
#: :class:`~simplyprint_ws_client.integration.tasks.context.TaskContext` and performs
#: one unit of work; its return value is ignored -- a task reports by publishing
#: to the status registry, not by returning.
TaskFn = Callable[..., Awaitable[object]]


@dataclass(frozen=True)
class TaskSpec:
    """One declarative, schedulable / triggerable unit of work.

    At most one scheduling mode may be set: ``interval`` (fixed cadence) or
    ``cron`` (calendar expression). If *both* are ``None`` the task is
    **on-demand only** -- it never fires on a timer but can be triggered by name
    (e.g. "re-check this one printer" from the onboarding debugger).
    """

    name: str
    """Stable, unique identifier. Becomes the scheduler job id, the key for
    on-demand triggers, and the label for per-task logging. Convention:
    ``"<area>.<verb>"`` -- e.g. ``"health.compute"``, ``"ota.check"``,
    ``"discovery.sweep"``."""

    fn: TaskFn
    """The coroutine that does the work. Invoked as ``fn(ctx)`` with a
    :class:`~simplyprint_ws_client.integration.tasks.context.TaskContext`."""

    interval: Optional[timedelta] = None
    """Fixed-cadence schedule. Mutually exclusive with ``cron``."""

    cron: Optional[str] = None
    """Crontab-style schedule. Mutually exclusive with ``interval``."""

    timeout: Optional[float] = None
    """Hard wall-clock budget (seconds) for a single run; ``None`` = no timeout.
    A run that exceeds it is cancelled and recorded as a failure."""

    max_instances: int = 1
    """Concurrent runs of *this* task the scheduler permits. ``1`` (the default)
    means a still-running scheduled run makes the next tick misfire rather than
    overlap -- the primary anti-pileup guard."""

    coalesce: bool = True
    """If several scheduled runs were missed (process paused / asleep), run the
    task once on catch-up instead of N times back-to-back."""

    misfire_grace_time: Optional[int] = 30
    """Seconds a scheduled run may be late before it is skipped. Keep *low* for
    liveness-style polls (a stale poll is worthless) and high / ``None`` for
    must-run housekeeping."""

    enabled: bool = True
    """A disabled spec is registered but not put on a timer; still triggerable on
    demand. Lets a brand ship a task dark and flip it on via config."""

    run_at_startup: bool = False
    """Fire once shortly after the scheduler starts (staggered with other
    startup tasks), then continue on the normal schedule -- for housekeeping that
    should run soon after boot rather than wait a full interval. Ignored for
    on-demand-only tasks."""

    def __post_init__(self) -> None:
        if self.interval is not None and self.cron is not None:
            raise ValueError(
                f"task {self.name!r}: set at most one of interval / cron, not both"
            )

    @property
    def on_demand_only(self) -> bool:
        """True when the task has no timer and only runs when triggered by name."""
        return self.interval is None and self.cron is None
