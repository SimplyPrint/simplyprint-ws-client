"""Declarative recurring & on-demand tasks.

Register task specs into an application-owned :class:`TaskRegistry`, then pass
that registry to the app's single :class:`SchedulerService`. A task reports by
publishing to a
:class:`~simplyprint_ws_client.core.status.registry.StatusRegistry` (see
:mod:`simplyprint_ws_client.core.status`), so periodic work and the status
endpoint are decoupled: tasks compute on a schedule, the endpoint serves the
cached snapshot.

The same spec can fire two ways -- on a timer (continuous, service-wide) and on
demand (``SchedulerService.trigger(name, key=printer_id)``, e.g. the onboarding
debugger re-checking one printer) -- with a single-flight guard merging an
on-demand trigger into an in-flight run of the same task + entity.

Example -- a producer codes against this contract::

    import time
    from datetime import timedelta

    from simplyprint_ws_client.integration.tasks import TaskContext, TaskRegistry
    from simplyprint_ws_client.core.status import StatusEntry, StatusState

    def register_tasks(registry: TaskRegistry) -> None:
        @registry.periodic(interval=timedelta(hours=1), name="ota.check")
        async def check_for_update(ctx: TaskContext) -> None:
            ...  # do the work
            ctx.status.publish(StatusEntry(
                section="ota",
                state=StatusState.OK,
                detail={"has_update": False},
                updated_at=time.time(),
                ttl=3600,
            ))
"""

from simplyprint_ws_client.integration.tasks.context import TaskContext
from simplyprint_ws_client.integration.tasks.registry import TaskRegistry
from simplyprint_ws_client.integration.tasks.scheduler import SchedulerService
from simplyprint_ws_client.integration.tasks.spec import TaskFn, TaskSpec

__all__ = [
    "TaskSpec",
    "TaskFn",
    "TaskRegistry",
    "TaskContext",
    "SchedulerService",
]
