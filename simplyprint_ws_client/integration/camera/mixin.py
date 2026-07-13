import asyncio
import base64
import datetime
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Literal, TypeVar

from yarl import URL

from simplyprint_ws_client.integration.camera.handle import CameraHandle
from simplyprint_ws_client.integration.camera.pool import CameraPool
from simplyprint_ws_client.common.asyncio.cancelable_lock import CancelableLock
from simplyprint_ws_client.core.api.simplyprint_api import SimplyPrintApi
from simplyprint_ws_client.core.client import Client
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.protocol.messages import (
    WebcamSnapshotDemandData,
    StreamMsg,
)

_T = TypeVar("_T", bound=PrinterConfig)


@dataclass
class _CameraWork:
    kind: Literal["stream", "snapshot", "test"]
    data: WebcamSnapshotDemandData
    done: Optional[asyncio.Future] = None


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
    #: ``time.time()`` of the last published stream frame. A cached frame that
    #: arrived after this is *unseen* by the server and can be served with zero
    #: latency instead of blocking until the camera produces its next frame.
    _last_stream_frame_at: Optional[float] = None
    _camera_work_queue: asyncio.Queue
    _camera_worker_task: Optional[asyncio.Task] = None
    _camera_active_task: Optional[asyncio.Task] = None
    _camera_active_kind: Optional[Literal["stream", "snapshot", "test"]] = None
    _camera_stream_pending: bool = False
    _camera_snapshot_backlog: int = 0
    _camera_backlog_reported: bool = False
    _camera_closing: bool = False
    _camera_stream_cancel_requested: bool = False

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
        self._camera_work_queue = asyncio.Queue()
        self._camera_worker_task = None
        self._camera_active_task = None
        self._camera_active_kind = None
        self._camera_stream_pending = False
        self._camera_snapshot_backlog = 0
        self._camera_backlog_reported = False
        self._camera_closing = False
        self._camera_stream_cancel_requested = False

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
        if uri is None:
            if self._camera_handle:
                self._camera_handle.stop()
                self._camera_logger.debug(
                    f"Cleared previous camera handle ID {self._camera_handle.id}."
                )
                self._camera_handle = None
                self.event_loop.call_soon_threadsafe(self._stream_setup.clear)
            self._camera_uri = None
            self._camera_status = "ok"
            self._last_stream_frame_at = None
            try:
                self.printer.webcam_info.connected = False
            except AttributeError:
                pass
            return

        if self._camera_pool is None:
            self._camera_status = "err"
            self._camera_logger.debug(
                f"Dropped camera URI {uri} because no camera pool is available."
            )
            try:
                self.printer.webcam_info.connected = False
            except AttributeError:
                pass
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
                self._resume_webcam_stream(), loop=self.event_loop
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

    async def shutdown_camera_mixin(self) -> None:
        """Cancel and await the camera dispatch worker before releasing it."""
        self._camera_closing = True
        active = getattr(self, "_camera_active_task", None)
        worker = getattr(self, "_camera_worker_task", None)
        if active is not None:
            active.cancel()
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        queue = getattr(self, "_camera_work_queue", None)
        if queue is not None:
            while not queue.empty():
                work = queue.get_nowait()
                if work.done is not None and not work.done.done():
                    work.done.cancel()
                queue.task_done()

    #: How long a demand waits for a camera handle to be configured before
    #: giving up. Camera work is detached from connection dispatch, but it is
    #: still owned work and must not wait forever during shutdown or recovery.
    _CAMERA_SETUP_TIMEOUT = 10.0

    #: How long one frame read may take before it counts as a failed attempt.
    #: A camera worker that died or a frozen source must not hold the camera
    #: owner forever on a future nobody will resolve.
    _CAMERA_FRAME_TIMEOUT = 30.0

    async def _wait_for_camera(self) -> bool:
        """Wait (briefly, bounded) for a camera handle; False when none came."""
        if self._camera_handle:
            return True
        try:
            await asyncio.wait_for(
                self._stream_setup.wait(), self._CAMERA_SETUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            return False
        return self._camera_handle is not None

    async def on_stream_on(self):
        if await self._wait_for_camera():
            self._camera_handle.start()

    async def on_stream_off(self):
        if await self._wait_for_camera():
            self._camera_handle.pause()
        self._stream_lock.cancel()
        self._request_count = 0
        self._last_stream_frame_at = None
        # A stream read/upload is expendable once streaming is disabled. ID'd
        # snapshots and webcam tests remain lossless and are never preempted.
        if (
            getattr(self, "_camera_active_kind", None) == "stream"
            and getattr(self, "_camera_active_task", None) is not None
        ):
            self._camera_stream_cancel_requested = True
            self._camera_active_task.cancel()

    async def on_test_webcam(self):
        self._ensure_camera_worker()
        done = asyncio.get_running_loop().create_future()
        self._camera_work_queue.put_nowait(
            _CameraWork("test", WebcamSnapshotDemandData(), done)
        )
        await done

    #: Frame-read attempts per snapshot request before giving up.
    _SNAPSHOT_MAX_ATTEMPTS = 5

    async def on_webcam_snapshot(
        self,
        data: Optional[WebcamSnapshotDemandData] = None,
        attempt=0,
        retry_timeout=5,
    ):
        data = data or WebcamSnapshotDemandData()
        self._ensure_camera_worker()

        if data.id is None:
            # Stream demands have no identity and only represent outstanding
            # frame credit. Keep a single marker queued/active while retaining
            # every credit in ``_request_count``.
            self._request_count += 1
            self._queue_stream_marker()
            return
        else:
            # ID'd snapshots are individually meaningful and must not shed.
            kind = "snapshot"
            self._camera_snapshot_backlog += 1
            if (
                self._camera_snapshot_backlog >= 20
                and not self._camera_backlog_reported
            ):
                self._camera_backlog_reported = True
                self._camera_logger.warning(
                    "Camera snapshot backlog reached %d requests.",
                    self._camera_snapshot_backlog,
                )

        self._camera_work_queue.put_nowait(_CameraWork(kind, data))

    async def _resume_webcam_stream(self) -> None:
        """Resume existing stream credit after a camera handle appears."""
        self._ensure_camera_worker()
        self._queue_stream_marker()

    def _queue_stream_marker(self) -> None:
        if self._camera_stream_pending or self._request_count <= 0:
            return
        self._camera_stream_pending = True
        self._camera_work_queue.put_nowait(
            _CameraWork("stream", WebcamSnapshotDemandData())
        )

    def _ensure_camera_worker(self) -> None:
        """Lazily create the one owner of frame reads and uploads."""
        if not hasattr(self, "_camera_work_queue"):
            self._camera_work_queue = asyncio.Queue()
            self._camera_worker_task = None
            self._camera_active_task = None
            self._camera_active_kind = None
            self._camera_stream_pending = False
            self._camera_snapshot_backlog = 0
            self._camera_backlog_reported = False
            self._camera_closing = False
            self._camera_stream_cancel_requested = False

        if self._camera_closing:
            return
        if self._camera_worker_task is None or self._camera_worker_task.done():
            self._camera_worker_task = asyncio.get_running_loop().create_task(
                self._camera_worker(),
                name=f"sp-camera:{getattr(self, 'unique_id', 'unknown')}",
            )

    async def _camera_worker(self) -> None:
        """Serialize camera access while allowing cloud dispatch to continue."""
        while True:
            work = await self._camera_work_queue.get()
            result = False
            error = None
            worker_cancelled = False
            try:
                if work.kind == "stream" and self._request_count <= 0:
                    continue

                self._camera_active_kind = work.kind
                self._camera_active_task = asyncio.get_running_loop().create_task(
                    self._process_camera_work(work),
                    name=f"sp-camera-frame:{work.kind}",
                )
                try:
                    result = await self._camera_active_task
                except asyncio.CancelledError:
                    # Stream-off marks the one child cancellation that is safe
                    # to consume. Any other cancellation belongs to the owner
                    # task (teardown/loop shutdown) and must propagate.
                    if not (
                        work.kind == "stream"
                        and self._camera_stream_cancel_requested
                        and not self._camera_closing
                    ):
                        raise
            except asyncio.CancelledError:
                worker_cancelled = True
                raise
            except Exception as exc:  # keep one failed upload from killing owner
                error = exc
                self._camera_logger.warning(
                    "Camera %s request failed", work.kind, exc_info=True
                )
            finally:
                self._camera_active_task = None
                self._camera_active_kind = None
                self._camera_stream_cancel_requested = False

                if work.kind == "snapshot":
                    self._camera_snapshot_backlog = max(
                        0, self._camera_snapshot_backlog - 1
                    )
                    if (
                        self._camera_backlog_reported
                        and self._camera_snapshot_backlog < 10
                    ):
                        self._camera_backlog_reported = False
                elif work.kind == "stream":
                    if not result and self._request_count > 0:
                        self._request_count -= 1
                    if self._request_count > 0 and not self._camera_closing:
                        self._camera_work_queue.put_nowait(
                            _CameraWork("stream", WebcamSnapshotDemandData())
                        )
                    else:
                        self._camera_stream_pending = False

                if work.done is not None and not work.done.done():
                    if worker_cancelled:
                        work.done.cancel()
                    elif error is not None:
                        work.done.set_exception(error)
                    else:
                        work.done.set_result(result)
                self._camera_work_queue.task_done()

    async def _process_camera_work(self, work: _CameraWork) -> bool:
        frame = await self._receive_frame_with_retries(work.data, 0, 5)
        if frame is None:
            return False
        await self._publish_frame(work.data, frame)
        return True

    def _allowed_cache_age(
        self, data: WebcamSnapshotDemandData
    ) -> Optional[datetime.timedelta]:
        """How old a cached frame may be to satisfy this request.

        Snapshot events (``data.id``) tolerate ``max_cache_age``. Stream
        requests accept any frame newer than the last one we published --
        it is unseen by the server, so waiting for the camera's *next* frame
        only adds latency (a full frame period on slow chamber cams). The
        first frame of a stream session has no publish reference and falls
        back to ``max_cache_age``.
        """
        if data.id is not None:
            return self._camera_max_cache_age

        if self._last_stream_frame_at is None:
            return self._camera_max_cache_age

        return datetime.timedelta(seconds=time.time() - self._last_stream_frame_at)

    async def _receive_frame_with_retries(
        self, data: WebcamSnapshotDemandData, attempt: int, retry_timeout: float
    ) -> Optional[bytes]:
        if not await self._wait_for_camera():
            return None

        while True:
            handle = self._camera_handle
            if handle is None:
                return None

            st = datetime.datetime.now()

            # Block until the camera is ready, but serve an already-received
            # frame when it is acceptable: snapshot events may reuse a frame up
            # to ``max_cache_age`` old; stream requests may serve any frame the
            # server has not seen yet (zero-latency while the worker is hot).
            # Bounded: a dead camera worker counts as a failed attempt instead
            # of holding the dispatch hostage forever.
            try:
                frame = await asyncio.wait_for(
                    handle.receive_frame(allow_cache_age=self._allowed_cache_age(data)),
                    self._CAMERA_FRAME_TIMEOUT,
                )
            except asyncio.TimeoutError:
                frame = None

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
            self._last_stream_frame_at = time.time()

            if self._request_count > 0:
                self._request_count -= 1

        return len(self._stream_lock) == 0 and self._request_count > 0
