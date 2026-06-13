"""Regression: stopping a wedged camera worker must not block the app loop.

This is the structural inverse of the benchmark's measured 6,011 ms stall. A
camera worker whose read is wedged (a hung stream) used to make the synchronous
``camera_uri`` setter join the worker on the loop -- up to ~2s per THREAD worker,
~6s per PROCESS worker, and additive across cameras. With the WorkerPool reaper,
``handle.stop()`` only signals + enqueues; the join happens off the loop. A
heartbeat proves the loop keeps beating while a wedged worker is stopped.
"""

import asyncio
import time

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.worker.context import ExecutionContext
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.pool import CameraPool

from tests._loop_heartbeat import LoopHeartbeat


class _WedgedThreadCam(BaseCameraProtocol):
    """Emits one frame, then its read wedges -- the worker thread ignores its
    stop event, so a join WOULD block. Routed to THREAD for a deterministic,
    fast test (the PROCESS path is the full ~6s; THREAD is the clean ~2s join)."""

    is_async = False
    polling_mode = CameraProtocolPollingMode.CONTINUOUS
    execution_context = ExecutionContext.THREAD

    @staticmethod
    def test(uri):
        return True

    def read(self):
        yield b"frame"
        time.sleep(30)  # ignore the stop event -> the worker thread will not exit
        yield b"never"


@pytest.mark.asyncio
async def test_camera_handle_stop_does_not_block_the_loop():
    loop = asyncio.get_running_loop()
    pool = CameraPool(event_loop_provider=EventLoopProvider(loop=loop))
    pool.protocols.append(_WedgedThreadCam)
    handle = pool.create(URL("x://cam"))

    handle.start()
    await asyncio.wait_for(handle.receive_frame(), 2.0)  # one frame; then it wedges

    async with LoopHeartbeat(interval=0.01) as hb:
        start = time.perf_counter()
        handle.stop()  # the camera_uri=None path
        elapsed = time.perf_counter() - start

    assert elapsed < 0.2  # stop returned at once (no on-loop join)
    assert hb.max_gap_ms < 200  # the loop kept beating (was ~2000+ before the reaper)

    pool.stop()  # the wedged join is drained here, on the reaper -- off the loop
