"""Brand-free tests for the generic :class:`WorkerPool`.

A fake producer (just byte frames -- no camera, no OpenCV, no network) is run
through all three execution contexts. The point: the execution context is
transparent -- the same producer yields the same frames to an ``on_item`` sink
that always fires on the consumer loop, whether it ran inline, in a thread, or in
a subprocess across a zero-copy shared-memory channel.
"""

import asyncio
import logging
import multiprocessing as mp
import os
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
async def test_inline_rejects_a_sync_producer():
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    with pytest.raises(ValueError):
        pool.allocate(ExecutionContext.INLINE, _sync_producer, lambda d, t: None)


@pytest.mark.asyncio
async def test_process_runs_an_async_producer():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop))
    got = []

    handle = pool.allocate(
        ExecutionContext.PROCESS,
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


# -- the process group: bounded processes, unbounded producers --------------- #


def _tagged_stream(emit, is_stopped, tag):
    """A continuous producer, like a live camera: it never ends on its own."""
    while not is_stopped():
        emit(tag, time.time())
        time.sleep(0.01)


def _crash_after_first_frame(emit, is_stopped):
    emit(b"last words", time.time())
    time.sleep(0.2)
    os._exit(3)  # takes the whole worker process down


def _logging_producer(emit, is_stopped):
    logging.getLogger("worker.test").warning("hello from %s", "the worker")
    emit(b"logged", 0.0)


async def _first_items(pool, count, producer, args_for, *, attempts=1000):
    """Allocate ``count`` producers and wait until each delivered an item."""
    seen = {}
    handles = []
    for i in range(count):
        handles.append(
            pool.allocate(
                ExecutionContext.PROCESS,
                producer,
                lambda data, ts, i=i: seen.setdefault(i, bytes(data)),
                args=args_for(i),
            )
        )
    for _ in range(attempts):
        if len(seen) == count:
            break
        await asyncio.sleep(0.01)
    return handles, seen


@pytest.mark.asyncio
async def test_producers_are_not_limited_by_the_process_count():
    """Six never-ending producers on two processes all deliver: the process
    count bounds CPU parallelism, never how many producers run."""
    loop = asyncio.get_running_loop()
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=loop),
        max_processes=2,
        producers_per_process=1,
    )
    handles, seen = await _first_items(
        pool, 6, _tagged_stream, lambda i: (f"cam{i}".encode(),)
    )
    try:
        assert seen == {i: f"cam{i}".encode() for i in range(6)}
        assert sorted(pool.process_loads()) == [3, 3]
    finally:
        for handle in handles:
            handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_a_worker_fills_before_the_next_one_starts():
    """Memory follows load: six producers need two workers, not one per core."""
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop), max_processes=4)
    handles, seen = await _first_items(pool, 6, _tagged_stream, lambda i: (b"x",))
    try:
        assert len(seen) == 6
        assert pool.process_loads() == [4, 2]
    finally:
        for handle in handles:
            handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_producers_spread_over_processes_before_sharing_one():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=loop),
        max_processes=3,
        producers_per_process=1,
    )
    handles, seen = await _first_items(pool, 3, _tagged_stream, lambda i: (b"x",))
    try:
        assert len(seen) == 3
        assert pool.process_loads() == [1, 1, 1]
    finally:
        for handle in handles:
            handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_oneshot_producers_reuse_a_live_worker():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop), max_processes=1)
    children = set()
    try:
        for _ in range(3):
            delivered = loop.create_future()
            handle = pool.allocate(
                ExecutionContext.PROCESS,
                _oneshot_producer,
                lambda data, ts, done=delivered: (
                    done.done() or done.set_result((bytes(data), ts))
                ),
            )
            assert await asyncio.wait_for(delivered, 5.0) == (b"final", 42.0)
            children.update(p.pid for p in mp.active_children())
            for _ in range(200):  # completion retires the handle, not the worker
                if handle.stopped:
                    break
                await asyncio.sleep(0.01)
            assert handle.stopped
        assert pool.process_loads() == [0]
        assert len(children) == 1
    finally:
        pool.stop()


@pytest.mark.asyncio
async def test_a_stopped_producer_delivers_nothing_more():
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop), max_processes=1)
    got = []
    handle = pool.allocate(
        ExecutionContext.PROCESS,
        _tagged_stream,
        lambda data, ts: got.append(data),
        args=(b"f",),
    )
    try:
        await _drain(got, 3)
        handle.stop()
        settled = len(got)
        await asyncio.sleep(0.2)
        assert len(got) == settled
        assert pool.process_loads() == [0]
    finally:
        pool.stop()


@pytest.mark.asyncio
async def test_a_crashed_worker_fails_every_producer_on_it():
    """A worker that dies takes its producers down; each gets a failed (None)
    item and completes, and the pool starts a fresh worker for the next one."""
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop), max_processes=1)
    items = {"steady": [], "crash": []}
    steady = pool.allocate(
        ExecutionContext.PROCESS,
        _tagged_stream,
        lambda data, ts: items["steady"].append(data),
        args=(b"s",),
    )
    crash = pool.allocate(
        ExecutionContext.PROCESS,
        _crash_after_first_frame,
        lambda data, ts: items["crash"].append(data),
    )
    try:
        for _ in range(500):
            if steady.stopped and crash.stopped:
                break
            await asyncio.sleep(0.01)
        assert steady.stopped and crash.stopped
        assert items["crash"][0] == b"last words"
        assert items["steady"][-1] is None
        assert items["crash"][-1] is None
        assert pool.process_loads() == []

        handles, seen = await _first_items(
            pool, 1, _tagged_stream, lambda i: (b"again",)
        )
        assert seen == {0: b"again"}
        handles[0].stop()
    finally:
        pool.stop()


@pytest.mark.asyncio
async def test_an_idle_worker_shuts_down():
    loop = asyncio.get_running_loop()
    base = len(mp.active_children())
    pool = WorkerPool(
        event_loop_provider=EventLoopProvider(loop=loop),
        max_processes=1,
        idle_timeout=0.1,
    )
    try:
        handles, seen = await _first_items(pool, 1, _tagged_stream, lambda i: (b"x",))
        assert len(seen) == 1
        assert len(mp.active_children()) == base + 1
        handles[0].stop()
        for _ in range(300):
            if not pool.process_loads() and len(mp.active_children()) == base:
                break
            await asyncio.sleep(0.01)
        assert pool.process_loads() == []
        assert len(mp.active_children()) == base
    finally:
        pool.stop()


@pytest.mark.asyncio
async def test_worker_log_records_reach_the_parent(caplog):
    loop = asyncio.get_running_loop()
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=loop), max_processes=1)
    delivered = loop.create_future()
    with caplog.at_level(logging.WARNING, logger="worker.test"):
        handle = pool.allocate(
            ExecutionContext.PROCESS,
            _logging_producer,
            lambda data, ts: delivered.done() or delivered.set_result(bytes(data)),
        )
        try:
            assert await asyncio.wait_for(delivered, 5.0) == b"logged"
        finally:
            handle.stop()
            pool.stop()
    assert "hello from the worker" in caplog.text


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
