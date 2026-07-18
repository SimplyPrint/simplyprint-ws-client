"""Tests for :class:`CoalescingTask` -- the dirty-flag, trailing-edge
single-flight runner.

Pins the contract every storm-collapsing consumer relies on: a burst of triggers
becomes one run (1-2 with no debounce, exactly 1 within a debounce window); a
trigger that lands while the job runs causes exactly one trailing run; triggers
are thread-safe; ``aclose`` drains any pending work once and leaves no task; and
a trigger raised with no live loop is kept dirty rather than lost or raised.
"""

import asyncio
import threading

import pytest

from simplyprint_ws_client.common.asyncio.coalescing_task import CoalescingTask


@pytest.mark.asyncio
async def test_trigger_storm_collapses_to_one_or_two_runs():
    runs = 0

    async def fn():
        nonlocal runs
        runs += 1
        await asyncio.sleep(0)

    ct = CoalescingTask(fn, loop=asyncio.get_running_loop())
    for _ in range(100):
        ct.trigger()
    await asyncio.sleep(0.05)
    await ct.aclose()

    assert 1 <= runs <= 2


@pytest.mark.asyncio
async def test_trigger_during_run_causes_exactly_one_trailing_run():
    runs = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def fn():
        nonlocal runs
        runs += 1
        if runs == 1:
            started.set()
            await release.wait()  # hold the first run open

    ct = CoalescingTask(fn, loop=asyncio.get_running_loop())
    ct.trigger()
    await started.wait()
    ct.trigger()  # lands while the first run is in flight
    ct.trigger()
    release.set()
    await asyncio.sleep(0.05)
    await ct.aclose()

    assert runs == 2


@pytest.mark.asyncio
async def test_debounce_collapses_a_burst_to_one_run():
    runs = 0

    async def fn():
        nonlocal runs
        runs += 1

    ct = CoalescingTask(fn, delay=0.05, loop=asyncio.get_running_loop())
    for _ in range(10):
        ct.trigger()
        await asyncio.sleep(0.005)  # all within the 50ms window
    await asyncio.sleep(0.1)
    await ct.aclose()

    assert runs == 1


@pytest.mark.asyncio
async def test_trigger_from_a_foreign_thread():
    runs = 0

    async def fn():
        nonlocal runs
        runs += 1

    ct = CoalescingTask(fn, loop=asyncio.get_running_loop())
    thread = threading.Thread(target=ct.trigger)
    thread.start()
    thread.join()
    await asyncio.sleep(0.05)
    await ct.aclose()

    assert runs == 1


@pytest.mark.asyncio
async def test_aclose_drains_pending_work_and_leaves_no_task():
    runs = 0

    async def fn():
        nonlocal runs
        runs += 1

    before = asyncio.all_tasks()
    ct = CoalescingTask(fn, delay=0.05, loop=asyncio.get_running_loop())
    ct.trigger()
    await ct.aclose()  # drains the pending run (debounce skipped)

    assert runs == 1
    leftover = asyncio.all_tasks() - before - {asyncio.current_task()}
    assert not leftover


@pytest.mark.asyncio
async def test_trigger_after_aclose_is_a_noop():
    runs = 0

    async def fn():
        nonlocal runs
        runs += 1

    ct = CoalescingTask(fn, loop=asyncio.get_running_loop())
    await ct.aclose()
    ct.trigger()
    await asyncio.sleep(0.02)

    assert runs == 0
    assert ct.dirty is False


def test_trigger_with_a_dead_loop_keeps_dirty_without_raising():
    async def fn():  # pragma: no cover -- never runs (no live loop)
        return None

    loop = asyncio.new_event_loop()
    loop.close()  # a dead loop: call_soon_threadsafe will raise, caught internally

    ct = CoalescingTask(fn, loop=loop)
    ct.trigger()  # must not raise

    assert ct.dirty is True
