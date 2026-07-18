"""The owned camera worker bounds setup and frame waits and survives failures."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from yarl import URL

from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.integration.camera.controller import CameraController
from simplyprint_ws_client.core.protocol.messages import WebcamSnapshotDemandData


async def _send_stream(_message) -> None:
    pass


def _controller(camera_pool=None) -> CameraController:
    """Build the camera component without booting a cloud client."""
    printer = SimpleNamespace(
        config=SimpleNamespace(unique_id="camera-test"),
        webcam_info=SimpleNamespace(connected=False),
    )
    provider = SimpleNamespace(
        event_loop=SimpleNamespace(
            call_soon_threadsafe=lambda callback, *args: callback(*args)
        )
    )
    controller = CameraController(
        printer=printer,
        logger=logging.getLogger("test"),
        event_loop_provider=provider,
        send_stream=_send_stream,
        context=ClientContext(camera_pool=camera_pool),
        pause_timeout=10,
    )
    controller._request_count = 3
    # Tight bounds so a regression to unbounded waits fails fast.
    controller._CAMERA_SETUP_TIMEOUT = 0.02
    controller._CAMERA_FRAME_TIMEOUT = 0.02
    return controller


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


def _camera_with_pool() -> CameraController:
    controller = _controller(_FakeCameraPool())
    controller._request_count = 0
    return controller


def test_clearing_camera_uri_disconnects_webcam_and_stops_handle():
    camera = _camera_with_pool()

    assert camera.set_uri(URL("http://camera.local/stream"))
    handle = camera._camera_handle
    assert handle is not None
    assert camera._printer.webcam_info.connected is True

    assert camera.set_uri(None)

    assert handle.stopped is True
    assert camera._camera_handle is None
    assert camera.uri is None
    assert camera._printer.webcam_info.connected is False


def test_camera_uri_without_pool_does_not_advertise_connected_webcam():
    camera = _controller()
    camera._printer.webcam_info.connected = True

    assert not camera.set_uri(URL("http://camera.local/stream"))

    assert camera._printer.webcam_info.connected is False


@pytest.mark.asyncio
async def test_snapshot_demand_without_camera_returns_bounded():
    """No camera ever configured: the demand gives up within the setup bound
    instead of waiting on ``_stream_setup`` forever."""
    camera = _controller()
    await asyncio.wait_for(camera.snapshot(), timeout=1.0)


@pytest.mark.asyncio
async def test_stream_toggles_without_camera_return_bounded():
    camera = _controller()
    await asyncio.wait_for(camera.stream_on(), timeout=1.0)
    await asyncio.wait_for(camera.stream_off(), timeout=1.0)
    # stream-off still resets its request bookkeeping without a handle.
    assert camera._request_count == 0


@pytest.mark.asyncio
async def test_dead_camera_worker_counts_as_failed_attempts():
    """A handle whose frame future nobody resolves (dead worker, frozen
    source) is a bounded sequence of failed attempts, not a hang."""
    camera = _controller()

    async def never_resolves(allow_cache_age=None):
        await asyncio.Event().wait()

    camera._camera_handle = SimpleNamespace(
        receive_frame=never_resolves, id=1, fps=None
    )
    camera._stream_setup.set()

    frame = await asyncio.wait_for(
        camera._receive_frame_with_retries(
            WebcamSnapshotDemandData(), attempt=0, retry_timeout=0
        ),
        timeout=2.0,
    )
    assert frame is None


@pytest.mark.asyncio
async def test_stream_demands_coalesce_behind_one_camera_worker():
    camera = _controller()
    camera._request_count = 0
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def process(work):
        calls.append(work.data.id)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        camera._request_count -= 1
        return True

    camera._process_camera_work = process
    data = WebcamSnapshotDemandData()
    for _ in range(3):
        await camera.snapshot(data)

    await asyncio.wait_for(entered.wait(), 1.0)
    assert camera._camera_work_queue.empty()
    assert camera._request_count == 3

    release.set()
    await asyncio.wait_for(camera._camera_work_queue.join(), 1.0)
    assert calls == [None, None, None]
    assert camera._request_count == 0
    assert camera._camera_stream_pending is False
    await camera.close()


@pytest.mark.asyncio
async def test_snapshot_failure_does_not_kill_fifo_camera_worker():
    camera = _controller()
    camera._request_count = 0
    calls = []

    async def process(work):
        calls.append(work.data.id)
        if work.data.id == "bad":
            raise RuntimeError("upload failed")
        return True

    camera._process_camera_work = process
    await camera.snapshot(WebcamSnapshotDemandData(id="bad"))
    await camera.snapshot(WebcamSnapshotDemandData(id="good"))

    await asyncio.wait_for(camera._camera_work_queue.join(), 1.0)
    assert calls == ["bad", "good"]
    assert camera._camera_worker_task is not None
    assert not camera._camera_worker_task.done()
    assert camera._camera_snapshot_backlog == 0
    await camera.close()


@pytest.mark.asyncio
async def test_snapshot_queue_has_a_hard_entry_limit(caplog):
    camera = _controller()
    camera._request_count = 0
    camera._camera_closing = True  # keep the owner stopped so the queue stays full

    for index in range(camera._CAMERA_QUEUE_MAXSIZE + 1):
        await camera.snapshot(WebcamSnapshotDemandData(id=f"snapshot-{index}"))

    assert camera._camera_work_queue.qsize() == camera._CAMERA_QUEUE_MAXSIZE
    assert camera._camera_snapshot_backlog == camera._CAMERA_QUEUE_MAXSIZE
    assert "queue is full" in caplog.text


@pytest.mark.asyncio
async def test_stream_off_preempts_stream_but_preserves_snapshot():
    camera = _controller()
    camera._camera_handle = SimpleNamespace(pause=lambda: None, stop=lambda: None, id=1)
    stream_entered = asyncio.Event()
    snapshot_finished = asyncio.Event()

    async def process(work):
        if work.data.id is None:
            stream_entered.set()
            await asyncio.Event().wait()
        snapshot_finished.set()
        return True

    camera._process_camera_work = process
    stream = WebcamSnapshotDemandData()
    await camera.snapshot(stream)
    await camera.snapshot(WebcamSnapshotDemandData(id="snapshot"))
    await asyncio.wait_for(stream_entered.wait(), 1.0)

    await camera.stream_off()
    await asyncio.wait_for(snapshot_finished.wait(), 1.0)
    await asyncio.wait_for(camera._camera_work_queue.join(), 1.0)
    assert camera._request_count == 0
    await camera.close()
