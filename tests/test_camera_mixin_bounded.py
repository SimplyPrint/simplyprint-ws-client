"""The owned camera worker bounds setup and frame waits and survives failures."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.cancelable_lock import CancelableLock
from simplyprint_ws_client.integration.camera.mixin import ClientCameraMixin
from simplyprint_ws_client.core.protocol.messages import WebcamSnapshotDemandData


def _bare_mixin() -> ClientCameraMixin:
    """A mixin instance with just the camera state -- no full Client boot."""
    mixin = ClientCameraMixin.__new__(ClientCameraMixin)
    mixin._camera_handle = None
    mixin._stream_setup = asyncio.Event()
    mixin._stream_lock = CancelableLock()
    mixin._request_count = 3
    mixin._camera_max_cache_age = None
    mixin._camera_logger = logging.getLogger("test.camera")
    # Tight bounds so a regression to unbounded waits fails fast.
    mixin._CAMERA_SETUP_TIMEOUT = 0.02
    mixin._CAMERA_FRAME_TIMEOUT = 0.02
    return mixin


class _FakeCameraHandle:
    id = 7

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeCameraPool:
    def __init__(self):
        self.handle = _FakeCameraHandle()

    def create(self, uri, *, pause_timeout=None):
        return self.handle


def _camera_mixin_with_pool() -> ClientCameraMixin:
    mixin = _bare_mixin()
    mixin._camera_pool = _FakeCameraPool()
    mixin._camera_uri = None
    mixin._camera_status = "ok"
    mixin._camera_pause_timeout = 10
    mixin._request_count = 0
    mixin.event_loop = SimpleNamespace(
        call_soon_threadsafe=lambda callback, *args: callback(*args)
    )
    mixin.printer = SimpleNamespace(webcam_info=SimpleNamespace(connected=False))
    return mixin


def test_clearing_camera_uri_disconnects_webcam_and_stops_handle():
    mixin = _camera_mixin_with_pool()

    mixin.camera_uri = URL("http://camera.local/stream")
    handle = mixin._camera_handle
    assert handle is not None
    assert mixin.printer.webcam_info.connected is True

    mixin.camera_uri = None

    assert handle.stopped is True
    assert mixin._camera_handle is None
    assert mixin.camera_uri is None
    assert mixin.camera_status == "ok"
    assert mixin.printer.webcam_info.connected is False


def test_camera_uri_without_pool_does_not_advertise_connected_webcam():
    mixin = _bare_mixin()
    mixin._camera_pool = None
    mixin._camera_uri = None
    mixin._camera_status = "ok"
    mixin.printer = SimpleNamespace(webcam_info=SimpleNamespace(connected=True))

    mixin.camera_uri = URL("http://camera.local/stream")

    assert mixin.camera_status == "err"
    assert mixin.printer.webcam_info.connected is False


@pytest.mark.asyncio
async def test_snapshot_demand_without_camera_returns_bounded():
    """No camera ever configured: the demand gives up within the setup bound
    instead of waiting on ``_stream_setup`` forever."""
    mixin = _bare_mixin()
    await asyncio.wait_for(mixin.on_webcam_snapshot(), timeout=1.0)


@pytest.mark.asyncio
async def test_stream_toggles_without_camera_return_bounded():
    mixin = _bare_mixin()
    await asyncio.wait_for(mixin.on_stream_on(), timeout=1.0)
    await asyncio.wait_for(mixin.on_stream_off(), timeout=1.0)
    # stream-off still resets its request bookkeeping without a handle.
    assert mixin._request_count == 0


@pytest.mark.asyncio
async def test_dead_camera_worker_counts_as_failed_attempts():
    """A handle whose frame future nobody resolves (dead worker, frozen
    source) is a bounded sequence of failed attempts, not a hang."""
    mixin = _bare_mixin()

    async def never_resolves(allow_cache_age=None):
        await asyncio.Event().wait()

    mixin._camera_handle = SimpleNamespace(receive_frame=never_resolves, id=1, fps=None)
    mixin._stream_setup.set()

    frame = await asyncio.wait_for(
        mixin._receive_frame_with_retries(
            WebcamSnapshotDemandData(), attempt=0, retry_timeout=0
        ),
        timeout=2.0,
    )
    assert frame is None


@pytest.mark.asyncio
async def test_stream_demands_coalesce_behind_one_camera_worker():
    mixin = _bare_mixin()
    mixin._request_count = 0
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def process(work):
        calls.append(work.kind)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        mixin._request_count -= 1
        return True

    mixin._process_camera_work = process
    data = WebcamSnapshotDemandData()
    for _ in range(3):
        await mixin.on_webcam_snapshot(data)

    await asyncio.wait_for(entered.wait(), 1.0)
    assert mixin._camera_work_queue.empty()
    assert mixin._request_count == 3

    release.set()
    await asyncio.wait_for(mixin._camera_work_queue.join(), 1.0)
    assert calls == ["stream", "stream", "stream"]
    assert mixin._request_count == 0
    assert mixin._camera_stream_pending is False
    await mixin.shutdown_camera_mixin()


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_kill_fifo_camera_worker():
    mixin = _bare_mixin()
    mixin._request_count = 0
    calls = []

    async def process(work):
        calls.append(work.data.id)
        if work.data.id == "bad":
            raise RuntimeError("upload failed")
        return True

    mixin._process_camera_work = process
    await mixin.on_webcam_snapshot(WebcamSnapshotDemandData(id="bad"))
    await mixin.on_webcam_snapshot(WebcamSnapshotDemandData(id="good"))

    await asyncio.wait_for(mixin._camera_work_queue.join(), 1.0)
    assert calls == ["bad", "good"]
    assert mixin._camera_worker_task is not None
    assert not mixin._camera_worker_task.done()
    assert mixin._camera_snapshot_backlog == 0
    await mixin.shutdown_camera_mixin()


@pytest.mark.asyncio
async def test_snapshot_queue_has_a_hard_entry_limit(caplog):
    mixin = _bare_mixin()
    mixin._request_count = 0
    mixin._camera_closing = True  # keep the owner stopped so the queue stays full

    for index in range(mixin._CAMERA_QUEUE_MAXSIZE + 1):
        await mixin.on_webcam_snapshot(
            WebcamSnapshotDemandData(id=f"snapshot-{index}")
        )

    assert mixin._camera_work_queue.qsize() == mixin._CAMERA_QUEUE_MAXSIZE
    assert mixin._camera_snapshot_backlog == mixin._CAMERA_QUEUE_MAXSIZE
    assert "queue is full" in caplog.text


@pytest.mark.asyncio
async def test_stream_off_preempts_stream_but_preserves_snapshot():
    mixin = _bare_mixin()
    mixin._camera_handle = SimpleNamespace(pause=lambda: None)
    stream_entered = asyncio.Event()
    snapshot_finished = asyncio.Event()

    async def process(work):
        if work.kind == "stream":
            stream_entered.set()
            await asyncio.Event().wait()
        snapshot_finished.set()
        return True

    mixin._process_camera_work = process
    stream = WebcamSnapshotDemandData()
    await mixin.on_webcam_snapshot(stream)
    await mixin.on_webcam_snapshot(WebcamSnapshotDemandData(id="snapshot"))
    await asyncio.wait_for(stream_entered.wait(), 1.0)

    await mixin.on_stream_off()
    await asyncio.wait_for(snapshot_finished.wait(), 1.0)
    await asyncio.wait_for(mixin._camera_work_queue.join(), 1.0)
    assert mixin._request_count == 0
    await mixin.shutdown_camera_mixin()
