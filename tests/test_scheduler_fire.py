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


def test_owned_registries_and_status_do_not_leak_between_schedulers():
    registry_a = TaskRegistry()
    registry_b = TaskRegistry()
    status_a = StatusRegistry()
    status_b = StatusRegistry()

    async def _task(_ctx):
        return None

    registry_a.register(TaskSpec(name="a.only", fn=_task))
    registry_b.register(TaskSpec(name="b.only", fn=_task))

    with pytest.raises(KeyError):
        registry_a.get("b.only")
    with pytest.raises(KeyError):
        registry_b.get("a.only")

    scheduler_a = SchedulerService(status=status_a, registry=registry_a)
    scheduler_b = SchedulerService(status=status_b, registry=registry_b)
    scheduler_a.start()
    scheduler_b.start()
    try:
        scheduler_a.fire("a.only").result(timeout=3.0)
        scheduler_b.fire("b.only").result(timeout=3.0)
    finally:
        scheduler_b.shutdown()
        scheduler_a.shutdown()

    assert set(status_a.snapshot()["tasks"].detail) == {"a.only"}
    assert set(status_b.snapshot()["tasks"].detail) == {"b.only"}


def test_cross_thread_commands_have_a_lost_self_pipe_wake_backstop(monkeypatch):
    real_new_event_loop = asyncio.new_event_loop

    def new_event_loop_without_self_pipe_wake():
        loop = real_new_event_loop()
        monkeypatch.setattr(loop, "_write_to_self", lambda: None)
        return loop

    monkeypatch.setattr(
        asyncio, "new_event_loop", new_event_loop_without_self_pipe_wake
    )
    registry = TaskRegistry()
    ran = threading.Event()

    async def _task(_ctx):
        ran.set()

    registry.register(TaskSpec(name="t.lost-wake", fn=_task))
    scheduler = SchedulerService(status=StatusRegistry(), registry=registry)
    scheduler.start()
    try:
        scheduler.fire("t.lost-wake").result(timeout=3.0)
        assert ran.is_set()
    finally:
        # shutdown() is another cross-thread callback and therefore exercises
        # the same backstop before joining the scheduler thread.
        scheduler.shutdown()


@pytest.mark.asyncio
async def test_trigger_await_has_a_lost_caller_wake_backstop(monkeypatch):
    registry = TaskRegistry()

    async def _task(_ctx):
        return "done"

    registry.register(TaskSpec(name="t.async-lost-wake", fn=_task))
    scheduler = SchedulerService(status=StatusRegistry(), registry=registry)
    scheduler.start()
    try:
        caller_loop = asyncio.get_running_loop()
        monkeypatch.setattr(caller_loop, "_write_to_self", lambda: None)
        assert await scheduler.trigger("t.async-lost-wake") == "done"
    finally:
        scheduler.shutdown()
