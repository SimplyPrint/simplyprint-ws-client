import asyncio

import pytest

from simplyprint_ws_client.common.asyncio.bounded_dispatch import (
    BoundedAsyncDispatcher,
    map_concurrently,
)


@pytest.mark.asyncio
async def test_map_concurrently_bounds_task_count_and_preserves_order():
    active = 0
    peak = 0
    owner_tasks = set()

    async def handle(item):
        nonlocal active, peak
        owner_tasks.add(asyncio.current_task())
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return item * 2
        finally:
            active -= 1

    results = await map_concurrently(handle, range(100), concurrency=3)

    assert results == [item * 2 for item in range(100)]
    assert peak == 3
    assert len(owner_tasks) == 3


@pytest.mark.asyncio
async def test_dispatch_uses_fixed_workers_and_rejects_queue_overflow():
    gate = asyncio.Event()
    two_started = asyncio.Event()
    running = 0
    peak = 0
    overflows = 0

    async def handle(_item):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == 2:
            two_started.set()
        try:
            await gate.wait()
        finally:
            running -= 1

    def overflow():
        nonlocal overflows
        overflows += 1

    dispatch = BoundedAsyncDispatcher(
        handle,
        workers=2,
        maxsize=2,
        on_overflow=overflow,
    )
    assert dispatch.submit(1)
    assert dispatch.submit(2)
    await asyncio.wait_for(two_started.wait(), 1.0)

    assert dispatch.submit(3)
    assert dispatch.submit(4)
    assert not dispatch.submit(5)
    assert not dispatch.submit(6)
    assert dispatch.active_workers == 2
    assert dispatch.pending == 2
    assert overflows == 1

    gate.set()
    await asyncio.wait_for(dispatch.join(), 1.0)
    assert peak == 2
    dispatch.close()


@pytest.mark.asyncio
async def test_dispatch_contains_item_failure_and_retires_when_idle():
    handled = []
    errors = []

    async def handle(item):
        handled.append(item)
        if item == "bad":
            raise ValueError("bad item")

    dispatch = BoundedAsyncDispatcher(
        handle,
        workers=1,
        maxsize=4,
        idle_timeout=0.02,
        on_error=errors.append,
    )
    assert dispatch.submit("bad")
    assert dispatch.submit("good")
    await asyncio.wait_for(dispatch.join(), 1.0)

    assert handled == ["bad", "good"]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)

    await asyncio.sleep(0.05)
    assert dispatch.active_workers == 0
    dispatch.close()
