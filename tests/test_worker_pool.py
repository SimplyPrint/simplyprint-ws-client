"""Brand-free tests for the generic :class:`WorkerPool`.

A fake producer (just byte frames -- no camera, no OpenCV, no network) is run
through all three execution contexts. The point: the execution context is
transparent -- the same producer yields the same frames to an ``on_item`` sink
that always fires on the consumer loop, whether it ran inline, in a thread, or in
a subprocess across a zero-copy shared-memory channel.
"""

import asyncio
import multiprocessing as mp
import threading
import time

import pytest

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.worker import ExecutionContext, OverflowPolicy
from simplyprint_ws_client.common.worker.pool import WorkerPool

from tests._loop_heartbeat import LoopHeartbeat


def _frame(i, size):
    return bytes([i % 256]) * size


def _wedged_producer(emit, is_stopped, block):
    """Emit one frame then ignore the stop event -- models a camera worker stuck
    in a hung read. ``WorkerHandle.stop`` must not block on its join."""
    emit(b"x", 0.0)
    time.sleep(block)


def _sync_producer(emit, is_stopped, count, size):
    """A synchronous producer (the only kind PROCESS allows)."""
    for i in range(count):
        if is_stopped():
            return
        emit(_frame(i, size), float(i))
        time.sleep(0.001)


def _cooperative_producer(emit, is_stopped):
    emit(b"started", 0.0)
    while not is_stopped():
        time.sleep(0.01)


def _oneshot_producer(emit, is_stopped):
    emit(b"final", 42.0)


async def _async_producer(emit, is_stopped, count, size):
    """An async producer (INLINE / THREAD)."""
    for i in range(count):
        if is_stopped():
            return
        emit(_frame(i, size), float(i))
        await asyncio.sleep(0.001)


async def _drain(got, count, attempts=400):
    for _ in range(attempts):
        await asyncio.sleep(0.005)
        if len(got) >= count:
            break


@pytest.mark.asyncio
async def test_inline_async_producer_delivers_on_the_loop():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
    got = []

    handle = pool.allocate(
        ExecutionContext.INLINE,
        _async_producer,
        lambda data, ts: got.append((bytes(data), ts)),
        args=(5, 32),
        overflow=OverflowPolicy.UNBOUNDED,
    )
    try:
        await _drain(got, 5)
    finally:
        handle.stop()
        pool.stop()

    assert [d for d, _ in got][:5] == [_frame(i, 32) for i in range(5)]


@pytest.mark.asyncio
async def test_thread_async_producer_couriers_home():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
    got = []

    handle = pool.allocate(
        ExecutionContext.THREAD,
        _async_producer,
        lambda data, ts: got.append(bytes(data)),
        args=(6, 64),
        overflow=OverflowPolicy.UNBOUNDED,
    )
    try:
        await _drain(got, 6)
    finally:
        handle.stop()
        pool.stop()

    assert got[:6] == [_frame(i, 64) for i in range(6)]


@pytest.mark.asyncio
async def test_thread_sync_producer_couriers_home():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
    got = []

    handle = pool.allocate(
        ExecutionContext.THREAD,
        _sync_producer,
        lambda data, ts: got.append(bytes(data)),
        args=(6, 64),
        overflow=OverflowPolicy.UNBOUNDED,
    )
    try:
        await _drain(got, 6)
    finally:
        handle.stop()
        pool.stop()

    assert got[:6] == [_frame(i, 64) for i in range(6)]


@pytest.mark.asyncio
async def test_process_producer_delivers_zero_copy_frames():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=loop),
        n_slabs=16,
        slab_size=4096,
    )
    got = []

    handle = pool.allocate(
        ExecutionContext.PROCESS,
        _sync_producer,
        lambda data, ts: got.append(bytes(data)),
        args=(8, 1024),
        overflow=OverflowPolicy.UNBOUNDED,
    )
    try:
        await _drain(got, 8)
    finally:
        handle.stop()
        pool.stop()

    assert got[:8] == [_frame(i, 1024) for i in range(8)]


@pytest.mark.asyncio
async def test_process_delivers_frame_when_producer_exits_immediately():
    """An immediately exiting producer must not reap its channel first."""
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
    delivered = loop.create_future()

    def on_item(data, timestamp):
        delivered.set_result((bytes(data), timestamp))

    handle = pool.allocate(
        ExecutionContext.PROCESS,
        _oneshot_producer,
        on_item,
        overflow=OverflowPolicy.UNBOUNDED,
    )
    try:
        assert await asyncio.wait_for(delivered, 2.0) == (b"final", 42.0)
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_bounded_process_lane_reopens_after_oneshot_exit():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=loop), max_process_workers=1
    )

    async def run_once():
        delivered = loop.create_future()
        handle = await pool.allocate_async(
            ExecutionContext.PROCESS,
            _oneshot_producer,
            lambda data, timestamp: delivered.set_result((bytes(data), timestamp)),
            overflow=OverflowPolicy.UNBOUNDED,
        )
        return handle, await delivered

    first, first_frame = await run_once()
    try:
        second, second_frame = await asyncio.wait_for(run_once(), 2.0)
        assert first_frame == second_frame == (b"final", 42.0)
    finally:
        first.stop()
        if "second" in locals():
            second.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_inline_rejects_a_sync_producer():
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    with pytest.raises(ValueError):
        pool.allocate(ExecutionContext.INLINE, _sync_producer, lambda d, t: None)


@pytest.mark.asyncio
async def test_process_rejects_an_async_producer():
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    with pytest.raises(ValueError):
        pool.allocate(ExecutionContext.PROCESS, _async_producer, lambda d, t: None)


@pytest.mark.asyncio
async def test_bounded_process_lane_waits_instead_of_spawning_past_limit(monkeypatch):
    loop = asyncio.get_running_loop()
    provider = EventLoopProvider(loop=loop)
    pool = WorkerPool(
        event_loop_provider=provider,
        max_process_workers=1,
    )
    allocated = []

    class FakeHandle:
        def __init__(self, release_capacity):
            self._release_capacity = release_capacity

        def stop(self):
            self._release_capacity()

    def fake_allocate(*_args, _release_capacity=None, **_kwargs):
        handle = FakeHandle(_release_capacity)
        allocated.append(handle)
        return handle

    monkeypatch.setattr(pool, "allocate", fake_allocate)

    first = await pool.allocate_async(
        ExecutionContext.PROCESS,
        _cooperative_producer,
        lambda _data, _ts: None,
    )
    assert len(allocated) == 1

    second_allocation = asyncio.create_task(
        pool.allocate_async(
            ExecutionContext.PROCESS,
            _cooperative_producer,
            lambda _data, _ts: None,
        )
    )
    await asyncio.sleep(0.05)
    assert not second_allocation.done()
    assert len(allocated) == 1

    first.stop()
    second = await asyncio.wait_for(second_allocation, 1.0)
    assert len(allocated) == 2

    second.stop()
    pool.stop()


@pytest.mark.asyncio
async def test_bounded_process_lane_rejects_sync_allocation_bypass():
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop()),
        max_process_workers=1,
    )
    with pytest.raises(RuntimeError, match="allocate_async"):
        pool.allocate(
            ExecutionContext.PROCESS,
            _sync_producer,
            lambda _data, _ts: None,
            args=(1, 1),
        )
    pool.stop()


# -- reaper: stops never block the loop; teardown leaks nothing -------------- #


@pytest.mark.asyncio
async def test_handle_stop_is_nonblocking_with_a_wedged_producer():
    """handle.stop() signals + enqueues on the reaper and returns at once -- the
    up-to-JOIN_TIMEOUT join happens off the loop, so the loop keeps beating."""
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
    handle = pool.allocate(
        ExecutionContext.THREAD,
        _wedged_producer,
        lambda d, t: None,
        args=(3.0,),  # exceeds JOIN_TIMEOUT (2.0) so the join WOULD block
        overflow=OverflowPolicy.UNBOUNDED,
    )
    await asyncio.sleep(0.05)  # let it start and emit

    async with LoopHeartbeat(interval=0.01) as hb:
        start = time.perf_counter()
        handle.stop()
        elapsed = time.perf_counter() - start

    assert elapsed < 0.2  # did not block on the 2s join
    assert hb.max_gap_ms < 200  # the loop never stalled
    pool.stop()  # the sanctioned blocking-join site (drains the reaper)


def test_pool_stop_drains_the_reaper_and_leaks_no_threads():
    base = {t.name for t in threading.enumerate()}

    async def go():
        loop = asyncio.get_running_loop()
        pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
        handles = [
            pool.allocate(
                ExecutionContext.THREAD,
                _sync_producer,
                lambda d, t: None,
                args=(200, 32),  # still running when we stop them
                overflow=OverflowPolicy.UNBOUNDED,
            )
            for _ in range(3)
        ]
        await asyncio.sleep(0.05)
        for handle in handles:
            handle.stop()
        return pool

    pool = asyncio.run(go())
    pool.stop()

    names = {t.name for t in threading.enumerate()}
    assert "sp-worker-reaper" not in names  # the reaper was joined, not leaked
    assert len(threading.enumerate()) <= len(base)  # worker threads joined too


@pytest.mark.asyncio
async def test_pool_stop_terminates_a_wedged_process():
    base = len(mp.active_children())
    loop = asyncio.get_running_loop()
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=loop),
        n_slabs=8,
        slab_size=4096,
    )
    handle = pool.allocate(
        ExecutionContext.PROCESS,
        _wedged_producer,
        lambda d, t: None,
        args=(30.0,),  # ignores its stop event -> requires terminate escalation
        overflow=OverflowPolicy.UNBOUNDED,
    )
    await asyncio.sleep(0.2)  # let the subprocess spawn
    handle.stop()
    pool.stop()  # proc.join(2.0) times out -> terminate() -> joined, reaped

    assert len(mp.active_children()) <= base  # no leaked subprocess
