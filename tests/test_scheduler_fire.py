"""SchedulerService.fire() -- the sync, fire-and-forget sibling of trigger()."""

import asyncio
import threading

import pytest

from simplyprint_ws_client.core.status.registry import StatusRegistry
from simplyprint_ws_client.integration.tasks import (
    SchedulerService,
    TaskRegistry,
    TaskSpec,
)


def test_fire_runs_an_on_demand_task():
    registry = TaskRegistry()
    ran = threading.Event()

    async def _task(_ctx):
        ran.set()

    # On-demand: no interval/cron, so the scheduler never times it -- only fire()
    # (or trigger()) can run it.
    registry.register(TaskSpec(name="t.fire", fn=_task))

    scheduler = SchedulerService(status=StatusRegistry(), registry=registry)
    scheduler.start()
    try:
        future = scheduler.fire("t.fire")
        assert ran.wait(timeout=3.0)
        # fire() returns the scheduling Future without awaiting it.
        future.result(timeout=3.0)
    finally:
        scheduler.shutdown()


def test_fire_before_start_raises():
    scheduler = SchedulerService(status=StatusRegistry(), registry=TaskRegistry())
    with pytest.raises(RuntimeError):
        scheduler.fire("anything")


def test_shutdown_lets_running_tasks_cleanup():
    registry = TaskRegistry()
    started = threading.Event()
    cleaned = threading.Event()

    async def _task(_ctx):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    registry.register(TaskSpec(name="t.cleanup", fn=_task))
    scheduler = SchedulerService(status=StatusRegistry(), registry=registry)
    scheduler.start()
    try:
        scheduler.fire("t.cleanup")
        assert started.wait(timeout=3.0)
    finally:
        scheduler.shutdown()

    assert cleaned.wait(timeout=3.0)


def test_scheduler_restarts_after_shutdown():
    registry = TaskRegistry()
    ran = threading.Event()

    async def _task(_ctx):
        ran.set()

    registry.register(TaskSpec(name="t.restart", fn=_task))
    scheduler = SchedulerService(status=StatusRegistry(), registry=registry)

    scheduler.start()
    scheduler.shutdown()

    scheduler.start()
    try:
        scheduler.fire("t.restart").result(timeout=3.0)
        assert ran.is_set()
    finally:
        scheduler.shutdown()
