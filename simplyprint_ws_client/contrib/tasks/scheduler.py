"""The one place that drives a scheduler backend: :class:`SchedulerService`.

Reads the :data:`~simplyprint_ws_client.contrib.tasks.registry.REGISTRY` once at
startup and arranges every task's firing on APScheduler (3.x; the v4 ``add_job``
-> ``add_schedule`` rename is confined to this file). It is also the single entry
point for the two firing paths a task supports:

* **scheduled** -- interval / cron specs fire on a timer, with the spec's
  ``max_instances`` / ``coalesce`` / ``misfire_grace_time`` taken straight through
  to prevent pile-up, and ``run_at_startup`` specs fired once shortly after boot;
* **on demand** -- :meth:`trigger` runs a task now (e.g. "re-check this printer"
  from the onboarding debugger), keyed per entity.

A single-flight guard keyed by ``(task name, key)`` merges a second run of the
same task + entity into the one already in flight instead of starting a duplicate
-- so the debugger hammering "retry" never piles work onto a running sweep, while
distinct printers never block each other. Every run's outcome (and any failure)
is recorded under the reserved :data:`TASKS_SECTION` of the status registry, so a
wedged task becomes visible instead of silently vanishing.

The scheduler runs an ``AsyncIOScheduler`` on its **own event loop in its own
daemon thread**: periodic work stays isolated from the client loop (as the old
thread-based scheduler was) while remaining ``async`` -- tasks are coroutines,
blocking work offloads via ``asyncio.to_thread``, and on-demand triggers from
another loop marshal across with ``run_coroutine_threadsafe``. APScheduler is
imported lazily so importing this module stays dependency-light.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from simplyprint_ws_client.contrib.status.registry import (
    StatusEntry,
    StatusRegistry,
    StatusState,
)
from simplyprint_ws_client.contrib.tasks.context import TaskContext
from simplyprint_ws_client.contrib.tasks.registry import REGISTRY, TaskRegistry
from simplyprint_ws_client.contrib.tasks.spec import TaskSpec

logger = logging.getLogger("simplyprint.tasks")

#: Reserved status section the scheduler publishes its own per-task runtime
#: health into (last run, ok/error, duration). Producers own every other section.
TASKS_SECTION = "tasks"

# Startup-fire timing, preserving the old JobScheduler behaviour: a
# run_at_startup task first fires shortly after boot, each staggered a minute
# apart so they don't all land at once.
_STARTUP_DELAY = timedelta(seconds=10)
_STARTUP_STAGGER = timedelta(minutes=1)


class SchedulerService:
    """Owns the scheduler lifecycle and the scheduled / on-demand firing paths."""

    def __init__(
        self,
        status: StatusRegistry,
        registry: TaskRegistry = REGISTRY,
    ) -> None:
        self._status = status
        self._registry = registry
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._scheduler: Optional[Any] = None  # apscheduler AsyncIOScheduler
        self._ready = threading.Event()
        self._started = False
        # Single-flight: the in-flight run per (task name, key). Only ever touched
        # on the scheduler loop, so a plain dict needs no extra locking.
        self._inflight: Dict[Tuple[str, str], asyncio.Future] = {}
        # Per-task runtime health, republished under TASKS_SECTION on each run.
        self._health: Dict[str, Dict[str, Any]] = {}


    def start(self) -> None:
        """Spin up the scheduler loop/thread, register every enabled scheduled
        spec, and begin firing. Idempotent; blocks until the scheduler is up."""
        if self._started:
            return
        self._started = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="task-scheduler", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def shutdown(self) -> None:
        """Stop firing and tear down the loop/thread. Idempotent. In-flight runs
        are cancelled past a short join grace."""
        if not self._started or self._loop is None:
            return
        self._started = False
        loop = self._loop

        def _stop() -> None:
            if self._scheduler is not None and self._scheduler.running:
                self._scheduler.shutdown(wait=False)
            loop.stop()

        loop.call_soon_threadsafe(_stop)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run_loop(self) -> None:
        from apscheduler.events import EVENT_JOB_ERROR
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        # Quiet APScheduler's executor: it logs a full traceback on every job
        # error, which would double up with our own recording below.
        logging.getLogger("apscheduler.executors.default").setLevel(logging.CRITICAL)

        self._scheduler = AsyncIOScheduler(event_loop=self._loop)
        self._schedule_all()
        self._scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)
        self._scheduler.start()
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            for task in asyncio.all_tasks(self._loop):
                task.cancel()
            self._loop.close()


    def _schedule_all(self) -> None:
        assert self._scheduler is not None
        startup_index = 0
        for spec in self._registry.specs():
            if not spec.enabled or spec.on_demand_only:
                continue

            trigger, trigger_kwargs = self._trigger_for(spec)
            job_kwargs: Dict[str, Any] = {
                "id": spec.name,
                "name": spec.name,
                "args": [spec.name],
                "max_instances": spec.max_instances,
                "coalesce": spec.coalesce,
                "misfire_grace_time": spec.misfire_grace_time,
            }
            if spec.run_at_startup:
                job_kwargs["next_run_time"] = (
                    datetime.now() + _STARTUP_DELAY + _STARTUP_STAGGER * startup_index
                )
                startup_index += 1

            self._scheduler.add_job(self._fire, trigger, **trigger_kwargs, **job_kwargs)

    @staticmethod
    def _trigger_for(spec: TaskSpec) -> Tuple[Any, Dict[str, Any]]:
        if spec.cron is not None:
            from apscheduler.triggers.cron import CronTrigger

            return CronTrigger.from_crontab(spec.cron), {}
        # on_demand_only specs are filtered before we get here, so interval is set
        assert spec.interval is not None
        return "interval", {"seconds": spec.interval.total_seconds()}


    async def _fire(self, name: str) -> None:
        """The APScheduler entry point for a scheduled run (key = service-wide)."""
        await self._run(self._registry.get(name), key="")

    async def trigger(self, name: str, *, key: str = "") -> object:
        """Run task ``name`` now, scoped to ``key`` (e.g. a printer id).

        Coalesces under the single-flight guard: if a run of ``(name, key)`` is
        already in flight, the caller joins it instead of starting a second, and
        receives that run's result. Safe to call from any loop/thread; marshals
        the run onto the scheduler loop.
        """
        if not self._started or self._loop is None:
            raise RuntimeError("SchedulerService is not started")
        spec = self._registry.get(name)
        fut = asyncio.run_coroutine_threadsafe(self._run(spec, key=key), self._loop)
        return await asyncio.wrap_future(fut)

    async def _run(self, spec: TaskSpec, key: str) -> object:
        """Single-flight wrapper -- one in-flight run per (name, key); late
        callers join the running one rather than starting a duplicate."""
        ident = (spec.name, key)
        existing = self._inflight.get(ident)
        if existing is not None:
            return await asyncio.shield(existing)

        fut = asyncio.ensure_future(self._execute(spec, key))
        self._inflight[ident] = fut
        try:
            return await fut
        finally:
            self._inflight.pop(ident, None)

    async def _execute(self, spec: TaskSpec, key: str) -> object:
        ctx = TaskContext(status=self._status, key=key)
        started = time.time()
        try:
            coro = spec.fn(ctx)
            if spec.timeout is not None:
                result = await asyncio.wait_for(coro, spec.timeout)
            else:
                result = await coro
        except Exception as exc:
            self._record(spec.name, started, error=repr(exc))
            raise
        self._record(spec.name, started, error=None)
        return result


    def _record(self, name: str, started: float, *, error: Optional[str]) -> None:
        now = time.time()
        self._health[name] = {
            "last_run": now,
            "ok": error is None,
            "error": error,
            "duration_s": round(now - started, 3),
        }
        failing = any(not h["ok"] for h in self._health.values())
        self._status.publish(
            StatusEntry(
                section=TASKS_SECTION,
                state=StatusState.DEGRADED if failing else StatusState.OK,
                detail=dict(self._health),
                updated_at=now,
            )
        )

    def _on_job_error(self, event: Any) -> None:
        # _execute already recorded the failure into the status registry; this is
        # just the propagated-exception breadcrumb (kept so swallowing inside a
        # task is never required to stay quiet).
        logger.debug("task %s raised: %r", event.job_id, event.exception)
