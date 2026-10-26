"""The camera demand signal: a demand-driven camera makes a frame only when a
client waits for one.

``CameraHandle.receive_frame`` polls the backend; for a ``demand_driven``
continuous protocol each poll is one request, carried to the producer by the
worker pool's signal (over the command pipe for a worker process) and handed
to the protocol as a :class:`simplyprint_sdk.camera.FrameDemand`.
"""

import asyncio
import time

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.worker import ExecutionContext, OverflowPolicy
from simplyprint_ws_client.common.worker.pool import WorkerPool, WorkerSignal
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.pool import CameraPool

# --- the pool's signal ----------------------------------------------------------


def test_a_signal_set_before_listening_is_replayed_once():
    signal = WorkerSignal()
    signal.set(1)
    signal.set(2)  # only the latest counts
    heard = []
    signal.listen(heard.append)
    signal.set(3)
    assert heard == [2, 3]


def _echo_signals(emit, is_stopped, delay, *, signal):
    """Emit every signal value; start listening only after ``delay``."""
    time.sleep(delay)
    signal.listen(lambda value: emit(str(value).encode(), 0.0))
    while not is_stopped():
        time.sleep(0.005)


async def _until(predicate, attempts=1000):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [ExecutionContext.THREAD, ExecutionContext.PROCESS])
async def test_the_owner_signals_a_running_producer(context):
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop()))
    got = []
    handle = pool.allocate(
        context,
        _echo_signals,
        lambda data, ts: got.append(bytes(data)),
        args=(0.05,),
        overflow=OverflowPolicy.UNBOUNDED,
        signal=True,
    )
    try:
        handle.signal(7)  # before the producer listens: kept for it
        await _until(lambda: got)
        handle.signal(8)
        await _until(lambda: got[-1:] == [b"8"])
    finally:
        handle.stop()
        pool.stop()

    assert got == [b"7", b"8"]


def _plain(emit, is_stopped):
    emit(b"plain", 0.0)


@pytest.mark.asyncio
async def test_a_producer_that_asked_for_no_signal_gets_no_keyword():
    pool = WorkerPool(event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop()))
    got = []
    handle = pool.allocate(ExecutionContext.THREAD, _plain, lambda d, ts: got.append(d))
    try:
        await _until(lambda: got)
        handle.signal(1)  # ignored, not an error
    finally:
        handle.stop()
        pool.stop()
    assert got == [b"plain"]


# --- demand-driven cameras ------------------------------------------------------


class _AsyncDemandCam(BaseCameraProtocol):
    """A stream that reads a 'packet' every 5 ms and makes a frame only when
    one is wanted."""

    is_async = True
    polling_mode = CameraProtocolPollingMode.CONTINUOUS
    demand_driven = True

    @staticmethod
    def test(uri):
        return True

    async def read(self):
        made = 0
        while True:
            await asyncio.sleep(0.005)
            if self.demand is None or self.demand.take():
                made += 1
                yield f"f{made}".encode()


class _SyncDemandCam(BaseCameraProtocol):
    """The same, blocking: it runs in a worker process."""

    polling_mode = CameraProtocolPollingMode.CONTINUOUS
    demand_driven = True

    @staticmethod
    def test(uri):
        return True

    def read(self):
        made = 0
        while True:
            time.sleep(0.005)
            if self.demand is None or self.demand.take():
                made += 1
                yield f"f{made}".encode()


def _counting(handle):
    delivered = []
    original = handle.deliver_frame

    def deliver(data, ts):
        delivered.append(data)
        original(data, ts)

    handle.deliver_frame = deliver
    return delivered


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", [_AsyncDemandCam, _SyncDemandCam])
async def test_a_demand_driven_camera_makes_one_frame_per_request(protocol):
    pool = CameraPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    pool.protocols.append(protocol)
    handle = pool.create(URL("x://cam"), pause_timeout=60)
    delivered = _counting(handle)
    try:
        handle.start()  # running (stream on), but nobody waits yet
        await asyncio.sleep(0.3)
        assert delivered == []  # no demand: no frames

        frames = [
            await asyncio.wait_for(handle.receive_frame(), timeout=10)
            for _ in range(3)
        ]
        assert frames == [b"f1", b"f2", b"f3"]  # fresh each time, one per request

        await asyncio.sleep(0.3)
        assert delivered == [b"f1", b"f2", b"f3"]  # and nothing in between
    finally:
        handle.stop()
        pool.stop()


@pytest.mark.asyncio
async def test_a_request_before_the_worker_runs_reaches_it():
    pool = CameraPool(
        event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    pool.protocols.append(_AsyncDemandCam)
    handle = pool.create(URL("x://cam"), pause_timeout=60)
    try:
        # The first poll starts the worker and requests in one go: the request
        # is counted before the worker exists, and sent once it does.
        assert await asyncio.wait_for(handle.receive_frame(), timeout=5) == b"f1"
    finally:
        handle.stop()
        pool.stop()
