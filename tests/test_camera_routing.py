"""Brand-free tests for camera execution-context routing + async cameras.

The pool used to raise ``NotImplementedError`` for any async protocol. Now it
routes: async -> INLINE (a task on the consumer loop, no process, no pickle),
sync -> PROCESS (the proven CPU-isolated path), and an explicit
``execution_context`` override wins (e.g. an async camera in its own THREAD, or a
light sync camera off the main loop). These tests use fake protocols -- no
OpenCV, no network -- to pin routing and to prove INLINE/THREAD cameras stream
frames to a handle exactly like a process camera would.
"""

import asyncio
import multiprocessing as mp
import os
import time

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.pool import CameraPool
from simplyprint_ws_client.common.worker.context import ExecutionContext


def _shm_count():
    if not os.path.isdir("/dev/shm"):
        return None
    return len([n for n in os.listdir("/dev/shm") if n.startswith("psm_")])


def _restore_start_method(method):
    mp.set_start_method(method, force=True)


async def _assert_process_camera_delivers_via_shared_memory():
    before = _shm_count()
    pool = _pool(asyncio.get_running_loop())
    pool.protocols.append(_ProcessSnapshot)
    handle = pool.create(URL("x://cam"))
    try:
        frame = await asyncio.wait_for(handle.receive_frame(), 10.0)
        assert frame == b"PROCFRAME"
    finally:
        handle.stop()
        pool.stop()  # joins the reader thread, which closes+unlinks the segment

    if before is not None:
        assert _shm_count() == before  # the worker's channel was not leaked


class _ProcessSnapshot(BaseCameraProtocol):
    """A sync ON_DEMAND protocol -> routes to a worker PROCESS (must be picklable)."""

    is_async = False
    polling_mode = CameraProtocolPollingMode.ON_DEMAND

    @staticmethod
    def test(uri):
        return True

    def read(self):
        yield b"PROCFRAME"


class _AsyncStream(BaseCameraProtocol):
    is_async = True
    polling_mode = CameraProtocolPollingMode.CONTINUOUS

    @staticmethod
    def test(uri):
        return True

    async def read(self):
        i = 0
        while True:
            yield bytes([i % 256]) * 8
            i += 1
            await asyncio.sleep(0.005)


class _AsyncSnapshot(BaseCameraProtocol):
    is_async = True
    polling_mode = CameraProtocolPollingMode.ON_DEMAND

    @staticmethod
    def test(uri):
        return True

    async def read(self):
        yield b"SNAP"


class _SyncStreamThread(BaseCameraProtocol):
    is_async = False
    polling_mode = CameraProtocolPollingMode.CONTINUOUS
    execution_context = ExecutionContext.THREAD  # explicit: off the main loop

    @staticmethod
    def test(uri):
        return True

    def read(self):
        i = 0
        while True:
            yield bytes([i % 256]) * 8
            i += 1
            time.sleep(0.005)


class _PlainSync(BaseCameraProtocol):
    @staticmethod
    def test(uri):
        return True

    def read(self):
        return iter(())


def _pool(loop):
    return CameraPool(pool_size=1, event_loop_provider=EventLoopProvider(loop=loop))


# -- routing (pure) ---------------------------------------------------------- #


def test_async_protocol_routes_to_inline():
    assert CameraPool._route(_AsyncStream) is ExecutionContext.INLINE


def test_sync_protocol_routes_to_process():
    assert CameraPool._route(_PlainSync) is ExecutionContext.PROCESS


def test_explicit_execution_context_wins():
    assert CameraPool._route(_SyncStreamThread) is ExecutionContext.THREAD


# -- async camera execution -------------------------------------------------- #


@pytest.mark.asyncio
async def test_inline_async_stream_delivers_frames():
    pool = _pool(asyncio.get_running_loop())
    pool.protocols.append(_AsyncStream)
    handle = pool.create(URL("x://cam"))
    try:
        handle.start()
        frame = await asyncio.wait_for(handle.receive_frame(), 2.0)
        assert len(frame) == 8
        # a second frame keeps coming on its own (continuous)
        frame2 = await asyncio.wait_for(handle.receive_frame(), 2.0)
        assert len(frame2) == 8
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_inline_async_snapshot_reads_one():
    pool = _pool(asyncio.get_running_loop())
    pool.protocols.append(_AsyncSnapshot)
    handle = pool.create(URL("x://cam"))
    try:
        frame = await asyncio.wait_for(handle.receive_frame(), 2.0)
        assert frame == b"SNAP"
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_thread_camera_couriers_frames_home():
    pool = _pool(asyncio.get_running_loop())
    pool.protocols.append(_SyncStreamThread)
    handle = pool.create(URL("x://cam"))
    try:
        handle.start()
        frame = await asyncio.wait_for(handle.receive_frame(), 2.0)
        assert len(frame) == 8
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_no_protocol_match_raises():
    pool = _pool(asyncio.get_running_loop())
    with pytest.raises(ValueError):
        pool.create(URL("x://cam"))
    pool.stop()


@pytest.mark.asyncio
async def test_process_camera_delivers_via_shared_memory():
    # End-to-end PROCESS path: a real subprocess produces a frame that crosses the
    # zero-copy SharedSlabChannel and resolves the handle's receive_frame future.
    await _assert_process_camera_delivers_via_shared_memory()


@pytest.mark.asyncio
async def test_process_camera_starts_under_spawn():
    # Spawn pickles the Process object. The parent-owned SharedSlabChannel contains
    # a threading.Lock, so it must not be attached to the process until after start.
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    try:
        await _assert_process_camera_delivers_via_shared_memory()
    finally:
        _restore_start_method(previous)
