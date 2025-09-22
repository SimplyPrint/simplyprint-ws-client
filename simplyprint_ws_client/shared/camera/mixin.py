import asyncio
import base64
import datetime
import logging
import os
from typing import Optional, Literal, TypeVar

from yarl import URL

from .handle import CameraHandle
from .pool import CameraPool
from ..asyncio.cancelable_lock import CancelableLock
from ..sp.simplyprint_api import SimplyPrintApi
from ... import DemandMsgType
from ...core.client import Client, configure
from ...core.config import PrinterConfig
from ...core.ws_protocol.messages import (
    WebcamSnapshotDemandData,
    StreamMsg,
)

_T = TypeVar("_T", bound=PrinterConfig)


class ClientCameraMixin(Client[_T]):
    _camera_pool: Optional[CameraPool] = None
    _camera_uri: Optional[URL] = None
    _camera_handle: Optional[CameraHandle] = None
    _camera_status: Literal["ok", "new", "err"] = "ok"
    _camera_max_cache_age: Optional[datetime.timedelta] = None
    _camera_pause_timeout: Optional[int] = None
    _camera_debug: bool = False
    _camera_logger: logging.Logger = logging.getLogger(__name__)
    _stream_lock: CancelableLock
    _stream_setup: asyncio.Event
    _request_count: int = 0

    def initialize_camera_mixin(
        self,
        camera_pool: Optional[CameraPool] = None,
        pause_timeout: Optional[int] = None,
        max_cache_age: Optional[datetime.timedelta] = None,
        camera_debug: Optional[bool] = None,
        **_kwargs,
    ):
        self._camera_pool = camera_pool
        self._stream_lock = CancelableLock()
        self._stream_setup = asyncio.Event()
        self._camera_pause_timeout = pause_timeout
        self._camera_max_cache_age = max_cache_age
        self._camera_debug = (
            "SIMPLYPRINT_DEBUG_CAMERA" in os.environ
            if camera_debug is None
            else camera_debug
        )
        self._camera_logger = self.logger.getChild("camera")
        self._camera_logger.setLevel(
            logging.DEBUG if self._camera_debug else logging.INFO
        )

    @property
    def camera_status(self) -> Literal["ok", "new", "err"]:
        """Get the camera status."""
        return self._camera_status

    @property
    def camera_uri(self) -> Optional[URL]:
        """Get the camera URI."""
        return self._camera_uri

    @camera_uri.setter
    def camera_uri(self, uri: Optional[URL] = None):
        """Returns whether it has changed the camera URI"""
        if self._camera_pool is None:
            self._camera_status = "err"
            self._camera_logger.debug(
                f"Dropped camera URI {uri} because no camera pool is available."
            )
            return

        # If the camera URI is the same, don't recreate the camera.
        if self._camera_uri == uri and self._camera_handle:
            self._camera_status = "ok"
            self._camera_logger.debug(
                f"Camera URI {uri} is the same as the current one, not changing."
            )
            return

        self._camera_uri = uri

        # Clear out the previous camera (if URI is different)
        if self._camera_handle:
            self._camera_handle.stop()
            self._camera_logger.debug(
                f"Cleared previous camera handle ID {self._camera_handle.id}."
            )
            self._camera_handle = None
            self.event_loop.call_soon_threadsafe(self._stream_setup.clear)

        # Create a new camera handle
        if self._camera_uri and self._camera_pool:
            self._camera_handle = self._camera_pool.create(
                self._camera_uri, pause_timeout=self._camera_pause_timeout
            )
            self.event_loop.call_soon_threadsafe(self._stream_setup.set)

        # Check if we left off with a request that needs to be sent.
        if self._request_count > 0:
            asyncio.run_coroutine_threadsafe(
                self.on_webcam_snapshot(), loop=self.event_loop
            )

        # Mark the webcam as connected if it's not already.
        if not self.printer.webcam_info.connected:
            self.printer.webcam_info.connected = True

        self._camera_status = "new"
        self._camera_logger.debug(
            f"Set new camera URI to {uri} with handle ID {self._camera_handle.id if self._camera_handle else 'N/A'}, status is now {self._camera_status}."
        )

    def __del__(self):
        self.camera_uri = None

    async def on_stream_on(self):
        if not self._camera_handle:
            await self._stream_setup.wait()

        self._camera_handle.start()

    async def on_stream_off(self):
        if not self._camera_handle:
            await self._stream_setup.wait()

        self._camera_handle.pause()
        self._stream_lock.cancel()
        self._request_count = 0

    async def on_test_webcam(self):
        await self.on_webcam_snapshot()

    @configure(DemandMsgType.WEBCAM_SNAPSHOT, priority=2)
    def _before_webcam_snapshot(self, data: WebcamSnapshotDemandData):
        # Pure stream request, not a snapshot event.
        if data.id is None:
            self._request_count += 1

    async def on_webcam_snapshot(
        self,
        data: WebcamSnapshotDemandData = WebcamSnapshotDemandData(),
        attempt=0,
        retry_timeout=5,
    ):
        if not self._camera_handle:
            await self._stream_setup.wait()

        is_snapshot_event = data.id is not None

        st = datetime.datetime.now()

        # Block until the camera is ready, but we will sometimes allow snapshot events
        # to use existing images if they are new enough but only once.
        frame = await self._camera_handle.receive_frame(
            allow_cache_age=self._camera_max_cache_age if is_snapshot_event else None
        )

        # Empty frame or none.
        if not frame:
            if attempt <= 3:
                self._camera_logger.debug(
                    f"Failed to get frame, retrying in {retry_timeout} seconds"
                )
                await asyncio.sleep(retry_timeout)
                await self.on_webcam_snapshot(data, attempt + 1, retry_timeout)
            else:
                self._camera_logger.debug(
                    f"Failed to get frame, giving up. Used camera handle id {self._camera_handle.id}."
                )

            return

        self._camera_logger.debug(
            f"Received frame from camera with size {len(frame) if frame else 0} bytes "
            f"with an fps of {self._camera_handle.fps or 'N/A'} in "
            f"{datetime.datetime.now() - st} from camera handle id {self._camera_handle.id}."
        )

        # Capture snapshot events and send them to the API
        if is_snapshot_event:
            await SimplyPrintApi.post_snapshot(data.id, frame, endpoint=data.endpoint)
            self._camera_logger.debug(f"Posted snapshot to API with id {data.id}")
            return

        # Mark the webcam as connected if it's not already.
        if not self.printer.webcam_info.connected:
            self.printer.webcam_info.connected = True

        async with self._stream_lock:
            # Prevent racing between receiving frames and sending them.
            await self.printer.intervals.wait_for("webcam")
            b64frame = base64.b64encode(frame).decode("utf-8")
            await self.send(StreamMsg(b64frame))

            if self._request_count > 0:
                self._request_count -= 1

        # Keep sending frames until the request count is 0.
        if len(self._stream_lock) == 0 and self._request_count > 0:
            await self.on_webcam_snapshot()
