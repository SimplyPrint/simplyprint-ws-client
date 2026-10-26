"""A demand-driven camera that nobody watches still stops when its pool stops it.

Such a camera reads its stream (RTSP demux, a WebRTC session) and yields
nothing while no frame is wanted. The producer only regains control when the
camera yields, so the stop has to reach the camera while it waits: through
the :class:`~simplyprint_sdk.camera.FrameDemand` it consults between packets
and while idle. Without that, every view-then-pause leaks a stream session
(Bambu printers then refuse the next one) and the worker never goes idle.
"""

import asyncio
import threading
import time

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.worker import ExecutionContext
from simplyprint_ws_client.common.worker.pool import WorkerSignal
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.pool import (
    CameraPool,
    _sync_camera_producer,
)


class _IdleCam(BaseCameraProtocol):
    """Reads a 'packet' every 5 ms and asks the demand about each one, as
    ``DemandDecoder.push`` does. Nobody ever requests, so it yields nothing."""

    polling_mode = CameraProtocolPollingMode.CONTINUOUS
    demand_driven = True
    #: Runs in a thread of this process, so the test sees ``closed``.
    execution_context = ExecutionContext.THREAD
    #: Set when ``read`` cleaned up (its stream session closed).
    closed: threading.Event = threading.Event()

    @staticmethod
    def test(uri):
        return True

    def read(self):
        try:
            while True:
                time.sleep(0.005)
                if self.demand is not None and self.demand.pending:
                    if self.demand.take():
                        yield b"frame"
        finally:
            type(self).closed.set()


class _SleepingCam(_IdleCam):
    """Sleeps on ``FrameDemand.wait()`` without a timeout between frames."""

    def read(self):
        try:
            while True:
                if self.demand.wait() and self.demand.take():
                    yield b"frame"
        finally:
            type(self).closed.set()


@pytest.mark.parametrize("protocol", [_IdleCam, _SleepingCam])
def test_a_stopped_producer_ends_a_camera_that_yields_nothing(protocol):
    protocol.closed = threading.Event()
    stop = threading.Event()
    emitted = []
    producer = threading.Thread(
        target=_sync_camera_producer,
        args=(
            lambda *item: emitted.append(item),
            stop.is_set,
            protocol(URL("x://cam")),
            True,
            WorkerSignal(),
        ),
        daemon=True,
    )
    producer.start()
    time.sleep(0.1)  # running, idle: no request ever came
    stop.set()
    producer.join(timeout=3)

    assert not producer.is_alive()  # the worker goes idle
    assert protocol.closed.is_set()  # the camera's session was closed
    assert emitted == []  # a stop is not a failed read


@pytest.mark.asyncio
async def test_pausing_an_unwatched_demand_driven_camera_closes_its_stream():
    _IdleCam.closed = threading.Event()
    pool = CameraPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    pool.protocols.append(_IdleCam)
    handle = pool.create(URL("x://cam"), pause_timeout=60)
    try:
        handle.start()  # stream on, nobody waits for a frame
        await asyncio.sleep(0.2)
        assert not _IdleCam.closed.is_set()
        handle.pause()
        for _ in range(300):
            if _IdleCam.closed.is_set():
                break
            await asyncio.sleep(0.01)
        assert _IdleCam.closed.is_set()
    finally:
        handle.stop()
        pool.stop()
