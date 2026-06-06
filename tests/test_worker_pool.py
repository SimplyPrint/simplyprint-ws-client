"""Brand-free tests for the generic :class:`WorkerPool`.

A fake producer (just byte frames -- no camera, no OpenCV, no network) is run
through all three execution contexts. The point: the execution context is
transparent -- the same producer yields the same frames to an ``on_item`` sink
that always fires on the consumer loop, whether it ran inline, in a thread, or in
a subprocess across a zero-copy shared-memory channel.
"""

import asyncio
import time

import pytest

from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.worker import ExecutionContext, OverflowPolicy
from simplyprint_ws_client.shared.worker.pool import WorkerPool


def _frame(i, size):
    return bytes([i % 256]) * size


def _sync_producer(emit, is_stopped, count, size):
    """A synchronous producer (the only kind PROCESS allows)."""
    for i in range(count):
        if is_stopped():
            return
        emit(_frame(i, size), float(i))
        time.sleep(0.001)


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
