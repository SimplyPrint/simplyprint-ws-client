import asyncio
import base64
import datetime
import logging
import os
from typing import Optional, Literal, TypeVar

from yarl import URL

from simplyprint_ws_client.integration.camera.handle import CameraHandle
from simplyprint_ws_client.integration.camera.pool import CameraPool
from simplyprint_ws_client.common.asyncio.cancelable_lock import CancelableLock
from simplyprint_ws_client.core.api.simplyprint_api import SimplyPrintApi
from simplyprint_ws_client import DemandMsgType
from simplyprint_ws_client.core.client import Client, configure
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.protocol.messages import (
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

    def teardown_camera_mixin(self) -> None:
        """Release the camera handle. Called from the client's ``teardown`` -
        explicit ownership, never ``__del__`` (a finalizer would re-enter the
        event loop during GC)."""
        self.camera_uri = None

    async def on_stream_on(self):
        if not self._camera_handle:
            await self._stream_setup.wait()

        if self._camera_handle is not None:
            self._camera_handle.start()

    async def on_stream_off(self):
        if not self._camera_handle:
            await self._stream_setup.wait()

        if self._camera_handle is not None:
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

    #: Frame-read attempts per snapshot request before giving up.
    _SNAPSHOT_MAX_ATTEMPTS = 5

    async def on_webcam_snapshot(
        self,
        data: Optional[WebcamSnapshotDemandData] = None,
        attempt=0,
        retry_timeout=5,
    ):
        data = data or WebcamSnapshotDemandData()

        # Both the retry path and the keep-streaming path are loops (recursion
        # here used to grow the stack under sustained streaming).
        while True:
            frame = await self._receive_frame_with_retries(data, attempt, retry_timeout)
            if frame is None:
                return

            if not await self._publish_frame(data, frame):
                return

            # Keep sending frames until the request count is 0.
            data = WebcamSnapshotDemandData()
            attempt = 0

    async def _receive_frame_with_retries(
        self, data: WebcamSnapshotDemandData, attempt: int, retry_timeout: float
    ) -> Optional[bytes]:
        if not self._camera_handle:
            await self._stream_setup.wait()

        is_snapshot_event = data.id is not None

        while True:
            handle = self._camera_handle
            if handle is None:
                return None

            st = datetime.datetime.now()

            # Block until the camera is ready, but we will sometimes allow
            # snapshot events to use existing images if they are new enough,
            # but only once.
            frame = await handle.receive_frame(
                allow_cache_age=self._camera_max_cache_age
                if is_snapshot_event
                else None
            )

            if frame:
                self._camera_logger.debug(
                    f"Received frame from camera with size {len(frame)} bytes "
                    f"with an fps of {handle.fps or 'N/A'} in "
                    f"{datetime.datetime.now() - st} from camera handle id {handle.id}."
                )
                return frame

            attempt += 1
            if attempt >= self._SNAPSHOT_MAX_ATTEMPTS:
                self._camera_logger.debug(
                    f"Failed to get frame, giving up. Used camera handle id {handle.id}."
                )
                return None

            self._camera_logger.debug(
                f"Failed to get frame, retrying in {retry_timeout} seconds"
            )
            await asyncio.sleep(retry_timeout)

    async def _publish_frame(
        self, data: WebcamSnapshotDemandData, frame: bytes
    ) -> bool:
        """Deliver one frame; True when the stream should keep going."""
        # Capture snapshot events and send them to the API
        if data.id is not None:
            await SimplyPrintApi.post_snapshot(data.id, frame, endpoint=data.endpoint)
            self._camera_logger.debug(f"Posted snapshot to API with id {data.id}")
            return False

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

        return len(self._stream_lock) == 0 and self._request_count > 0
