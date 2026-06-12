"""Stream demands serve the newest *unseen* frame with zero latency.

The cloud paces a live stream by sending one ``webcam_snapshot`` demand per
frame it wants (~15s apart at the panel's idle cadence). Blocking each demand
on the camera's *next* frame adds a full frame period of latency (1.3s+ on
slow chamber cams) -- and pausing the worker between demands added a full
respawn + connect on top. The contract pinned here:

- ``CameraHandle.receive_frame`` judges the cache by the *frame's own* arrival
  time (a dead worker must never serve an ancient frame just because polls
  kept coming).
- The mixin lets a stream request use any cached frame that arrived after the
  last published one (unseen by the server), falling back to ``max_cache_age``
  for the first frame of a session; snapshot events keep ``max_cache_age``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time

import pytest

from simplyprint_ws_client.common.asyncio.cancelable_lock import CancelableLock
from simplyprint_ws_client.integration.camera.handle import CameraHandle
from simplyprint_ws_client.integration.camera.mixin import ClientCameraMixin
from simplyprint_ws_client.core.protocol.messages import WebcamSnapshotDemandData


class _NoopDriver:
    def poll(self):
        pass

    def start(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass


def _handle() -> CameraHandle:
    return CameraHandle(pool=None, camera_id=1, driver=_NoopDriver())


@pytest.mark.asyncio
async def test_receive_frame_serves_cached_frame_by_frame_age():
    handle = _handle()
    handle._set_frame(b"frame-1", time.time() - 0.2)

    frame = await asyncio.wait_for(
        handle.receive_frame(allow_cache_age=datetime.timedelta(seconds=1)),
        timeout=1.0,
    )
    assert frame == b"frame-1"


@pytest.mark.asyncio
async def test_receive_frame_rejects_stale_cached_frame():
    """An old frame is not served just because polling was recent -- the
    request blocks for a live frame instead."""
    handle = _handle()
    handle._set_frame(b"ancient", time.time() - 30.0)

    fut = asyncio.ensure_future(
        handle.receive_frame(allow_cache_age=datetime.timedelta(seconds=1))
    )
    await asyncio.sleep(0.05)
    assert not fut.done()

    handle._set_frame(b"fresh", time.time())
    assert await asyncio.wait_for(fut, timeout=1.0) == b"fresh"


def _bare_mixin() -> ClientCameraMixin:
    mixin = ClientCameraMixin.__new__(ClientCameraMixin)
    mixin._camera_handle = None
    mixin._stream_setup = asyncio.Event()
    mixin._stream_lock = CancelableLock()
    mixin._request_count = 0
    mixin._camera_max_cache_age = datetime.timedelta(seconds=1)
    mixin._camera_logger = logging.getLogger("test.camera")
    return mixin


def test_snapshot_event_uses_max_cache_age():
    mixin = _bare_mixin()
    mixin._last_stream_frame_at = time.time() - 5.0

    allowed = mixin._allowed_cache_age(WebcamSnapshotDemandData(id="abc"))
    assert allowed == mixin._camera_max_cache_age


def test_first_stream_frame_uses_max_cache_age():
    mixin = _bare_mixin()
    mixin._last_stream_frame_at = None

    allowed = mixin._allowed_cache_age(WebcamSnapshotDemandData())
    assert allowed == mixin._camera_max_cache_age


def test_stream_frame_accepts_anything_newer_than_last_publish():
    mixin = _bare_mixin()
    mixin._last_stream_frame_at = time.time() - 14.0

    allowed = mixin._allowed_cache_age(WebcamSnapshotDemandData())
    assert allowed is not None
    # Tolerance covers the test's own runtime; the point is the window tracks
    # the time since the last publish, not a fixed cap.
    assert 13.5 <= allowed.total_seconds() <= 14.5
