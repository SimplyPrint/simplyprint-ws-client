"""Camera demand handlers must be bounded.

These handlers run inline on the connection's dispatch chain: an unbounded
await here wedges the wire's supervise task -- recv never runs again, the
websockets keepalive kills the socket, and the transport sits CONNECTED on a
dead wire (the production keepalive-wedge bug). The mixin therefore bounds
both waits: camera-handle setup and each frame read.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.cancelable_lock import CancelableLock
from simplyprint_ws_client.core.api.simplyprint_api import SimplyPrintApiError
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
    await asyncio.wait_for(mixin._run_webcam_snapshot(), timeout=1.0)


@pytest.mark.asyncio
async def test_snapshot_demand_listener_returns_immediately():
    """The websocket dispatch listener must only queue camera work.

    Actual frame capture/upload runs in the background so slow cameras or
    snapshot uploads cannot stall all inbound SimplyPrint messages.
    """
    mixin = _bare_mixin()

    mixin.on_webcam_snapshot()

    assert mixin._webcam_snapshot_task is not None
    await asyncio.wait_for(mixin._webcam_snapshot_task, timeout=1.0)


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
async def test_failed_snapshot_upload_does_not_escape_dispatch(monkeypatch):
    mixin = _bare_mixin()

    async def fail_post_snapshot(*args, **kwargs):
        raise SimplyPrintApiError("boom")

    monkeypatch.setattr(
        "simplyprint_ws_client.integration.camera.mixin.SimplyPrintApi.post_snapshot",
        fail_post_snapshot,
    )

    keep_streaming = await mixin._publish_frame(
        WebcamSnapshotDemandData(id="snapshot-id"),
        b"jpeg",
    )

    assert keep_streaming is False
