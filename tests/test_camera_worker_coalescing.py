"""Tests for ``CameraWorkerBackend`` desired-state coalescing.

A storm of start/pause/poll commands (from device handlers, the pause timer, the
loop) must collapse to one reconcile and at most one live worker -- not a worker
allocate/stop per command. Frames from a retired worker must be dropped, and
``stop`` must free the pool slot at once.
"""

import asyncio
import threading

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.pool import CameraPool


class _Cam(BaseCameraProtocol):
    is_async = True
    polling_mode = CameraProtocolPollingMode.CONTINUOUS

    @staticmethod
    def test(uri):
        return True

    async def read(self):
        while True:
            yield b"f"
            await asyncio.sleep(0.005)


class _Snap(BaseCameraProtocol):
    is_async = True
    polling_mode = CameraProtocolPollingMode.ON_DEMAND

    @staticmethod
    def test(uri):
        return True

    async def read(self):
        yield b"snap"


def _pool():
    return CameraPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )


def _count_allocations(pool):
    calls = {"n": 0}
    original = pool._workers.allocate

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    pool._workers.allocate = counting
    return calls


@pytest.mark.asyncio
async def test_start_pause_storm_collapses_to_one_worker():
    pool = _pool()
    pool.protocols.append(_Cam)
    handle = pool.create(URL("x://cam"))
    driver = handle._driver
    allocs = _count_allocations(pool)

    def hammer():
        for i in range(25):
            (driver.start if i % 2 == 0 else driver.pause)()

    thread = threading.Thread(target=hammer)  # some commands from a foreign thread
    thread.start()
    hammer()
    thread.join()
    driver.start()  # final desired state: RUNNING
    await asyncio.sleep(0.2)  # let the reconcile settle

    try:
        assert driver._worker is not None  # exactly one live worker
        assert allocs["n"] <= 3  # not one allocation per command (<< 50)
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_oneshot_poll_burst_is_bounded():
    pool = _pool()
    pool.protocols.append(_Snap)
    handle = pool.create(URL("x://cam"))
    driver = handle._driver
    allocs = _count_allocations(pool)

    for _ in range(20):
        driver.poll()
    await asyncio.sleep(0.2)

    try:
        assert allocs["n"] <= 3  # a poll burst collapses (<< 20)
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_frames_stop_after_pause_settles():
    pool = _pool()
    pool.protocols.append(_Cam)
    handle = pool.create(URL("x://cam"))

    count = {"n": 0}
    original_set = handle._set_frame

    def counting_set(data, ts):
        count["n"] += 1
        original_set(data, ts)

    handle._set_frame = counting_set

    handle.start()
    await asyncio.sleep(0.1)
    assert count["n"] > 0  # streaming

    handle.pause()
    await asyncio.sleep(0.1)  # reconcile retires the worker; generation bumped
    settled = count["n"]
    await asyncio.sleep(0.1)

    try:
        assert count["n"] == settled  # no frames delivered after pause settled
        assert handle._driver._worker is None
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_stop_releases_the_pool_slot_immediately():
    pool = _pool()
    pool.protocols.append(_Cam)
    handle = pool.create(URL("x://cam"))
    handle.start()
    await asyncio.sleep(0.1)

    assert handle.id in pool.allocations
    handle.stop()
    assert handle.id not in pool.allocations  # released at once

    await asyncio.sleep(0.1)
    try:
        assert handle._driver._worker is None  # worker retired by the reconcile
    finally:
        pool.stop()
