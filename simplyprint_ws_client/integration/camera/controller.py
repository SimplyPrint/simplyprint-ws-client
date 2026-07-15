import asyncio
import base64
import datetime
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional, final

from yarl import URL

from simplyprint_ws_client.integration.camera.handle import CameraHandle
from simplyprint_ws_client.integration.camera.pool import CameraPool
from simplyprint_ws_client.common.asyncio.cancelable_lock import CancelableLock
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.protocol.messages import (
    StreamOffDemandData,
    StreamOnDemandData,
    WebcamTestDemandData,
    WebcamSnapshotDemandData,
    StreamMsg,
)
from simplyprint_ws_client.core.state import Interval, PrinterState

StreamSender = Callable[[StreamMsg], Awaitable[None]]


class _CameraWorkKind(Enum):
    STREAM = "stream"
    SNAPSHOT = "snapshot"
    TEST = "test"


@dataclass
class _CameraWork:
    kind: _CameraWorkKind
    data: WebcamSnapshotDemandData
    done: Optional[asyncio.Future] = None


@final
class CameraController:
    """Final owner of one printer's camera handle and demand-work lane.

    Protocol routing stays on :class:`PrinterClient`; this component owns only
    camera state, bounded work admission, frame delivery and teardown.
    """

    #: Queue entries, not tasks. Stream demand is coalesced to one entry, while
    #: ID'd snapshots remain individual up to this hard memory bound.
    _CAMERA_QUEUE_MAXSIZE = 20
    _camera_pool: Optional[CameraPool]
    _camera_uri: Optional[URL]
    _camera_handle: Optional[CameraHandle]
    _camera_max_cache_age: Optional[datetime.timedelta]
    _camera_pause_timeout: Optional[int]
    _camera_debug: bool
    _camera_logger: logging.Logger
    _stream_lock: CancelableLock
    _stream_setup: asyncio.Event
    _request_count: int
    #: ``time.time()`` of the last published stream frame. A cached frame that
    #: arrived after this is *unseen* by the server and can be served with zero
    #: latency instead of blocking until the camera produces its next frame.
    _last_stream_frame_at: Optional[float]
    _camera_work_queue: asyncio.Queue
    _camera_worker_task: Optional[asyncio.Task]
    _camera_active_task: Optional[asyncio.Task]
    _camera_active_kind: Optional[_CameraWorkKind]
    _camera_stream_pending: bool
    _camera_snapshot_backlog: int
    _camera_backlog_reported: bool
    _camera_closing: bool
    _camera_stream_cancel_requested: bool

    def __init__(
        self,
        *,
        printer: PrinterState,
        logger: logging.Logger,
        event_loop_provider: EventLoopProvider,
        send_stream: StreamSender,
        context: ClientContext,
        pause_timeout: Optional[int] = None,
        max_cache_age: Optional[datetime.timedelta] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self._printer = printer
        self._event_loop_provider = event_loop_provider
        self._send_stream = send_stream
        self._camera_pool = context.camera_pool
        self._simplyprint_api = context.simplyprint_api
        self._camera_uri = None
        self._camera_handle = None
        self._stream_lock = CancelableLock()
        self._stream_setup = asyncio.Event()
        self._camera_pause_timeout = pause_timeout
        self._camera_max_cache_age = max_cache_age
        self._camera_debug = (
            "SIMPLYPRINT_DEBUG_CAMERA" in os.environ if debug is None else debug
        )
        self._camera_logger = logger.getChild("camera")
        self._camera_logger.setLevel(
            logging.DEBUG if self._camera_debug else logging.INFO
        )
        self._camera_work_queue = asyncio.Queue(maxsize=self._CAMERA_QUEUE_MAXSIZE)
        self._request_count = 0
        self._last_stream_frame_at = None
        self._camera_worker_task = None
        self._camera_active_task = None
        self._camera_active_kind = None
        self._camera_stream_pending = False
        self._camera_snapshot_backlog = 0
        self._camera_backlog_reported = False
        self._camera_closing = False
        self._camera_stream_cancel_requested = False

    @property
    def uri(self) -> Optional[URL]:
        """Get the camera URI."""
        return self._camera_uri

    def set_uri(self, uri: Optional[URL]) -> bool:
        """Replace or clear the camera source; return whether it was accepted."""
        if uri is None:
            if self._camera_handle:
                self._camera_handle.stop()
                self._camera_logger.debug(
                    f"Cleared previous camera handle ID {self._camera_handle.id}."
                )
                self._camera_handle = None
                self._event_loop_provider.event_loop.call_soon_threadsafe(
                    self._stream_setup.clear
                )
            self._camera_uri = None
            self._last_stream_frame_at = None
            self._printer.webcam_info.connected = False
            return True

        if self._camera_pool is None:
            self._camera_logger.debug(
                f"Dropped camera URI {uri} because no camera pool is available."
            )
            self._printer.webcam_info.connected = False
            return False

        # If the camera URI is the same, don't recreate the camera.
        if self._camera_uri == uri and self._camera_handle:
            self._camera_logger.debug(
                f"Camera URI {uri} is the same as the current one, not changing."
            )
            return True

        self._camera_uri = uri

        # Clear out the previous camera (if URI is different)
        if self._camera_handle:
            self._camera_handle.stop()
            self._camera_logger.debug(
                f"Cleared previous camera handle ID {self._camera_handle.id}."
            )
            self._camera_handle = None
            self._event_loop_provider.event_loop.call_soon_threadsafe(
                self._stream_setup.clear
            )

        # Create a new camera handle
        if self._camera_uri and self._camera_pool:
            self._camera_handle = self._camera_pool.create(
                self._camera_uri, pause_timeout=self._camera_pause_timeout
            )
            self._event_loop_provider.event_loop.call_soon_threadsafe(
                self._stream_setup.set
            )

        # Check if we left off with a request that needs to be sent.
        if self._request_count > 0:
            asyncio.run_coroutine_threadsafe(
                self._resume_webcam_stream(),
                loop=self._event_loop_provider.event_loop,
            )

        # Mark the webcam as connected if it's not already.
        if not self._printer.webcam_info.connected:
            self._printer.webcam_info.connected = True

        self._camera_logger.debug(
            f"Set new camera URI to {uri} with handle ID {self._camera_handle.id if self._camera_handle else 'N/A'}."
        )
        return True

    async def close(self) -> None:
        """Cancel owned work, then release the camera handle."""
        self._camera_closing = True
        active = self._camera_active_task
        worker = self._camera_worker_task
        if active is not None:
            active.cancel()
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        while not self._camera_work_queue.empty():
            work = self._camera_work_queue.get_nowait()
            if work.done is not None and not work.done.done():
                work.done.cancel()
            self._camera_work_queue.task_done()

        # Explicit ownership, never ``__del__``: a finalizer would re-enter the
        # event loop during garbage collection.
        self.set_uri(None)

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

    async def stream_on(self, _data: Optional[StreamOnDemandData] = None) -> None:
        if await self._wait_for_camera():
            self._camera_handle.start()

    async def stream_off(self, _data: Optional[StreamOffDemandData] = None) -> None:
        if await self._wait_for_camera():
            self._camera_handle.pause()
        self._stream_lock.cancel()
        self._request_count = 0
        self._last_stream_frame_at = None
        # A stream read/upload is expendable once streaming is disabled. ID'd
        # snapshots and webcam tests remain lossless and are never preempted.
        if (
            self._camera_active_kind is _CameraWorkKind.STREAM
            and self._camera_active_task is not None
        ):
            self._camera_stream_cancel_requested = True
            self._camera_active_task.cancel()

    async def test_webcam(self, _data: Optional[WebcamTestDemandData] = None) -> None:
        self._ensure_camera_worker()
        done = asyncio.get_running_loop().create_future()
        try:
            self._camera_work_queue.put_nowait(
                _CameraWork(_CameraWorkKind.TEST, WebcamSnapshotDemandData(), done)
            )
        except asyncio.QueueFull:
            raise RuntimeError("Camera work queue is full") from None
        await done

    #: Frame-read attempts per snapshot request before giving up.
    _SNAPSHOT_MAX_ATTEMPTS = 5

    async def snapshot(
        self,
        data: Optional[WebcamSnapshotDemandData] = None,
    ) -> None:
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
            kind = _CameraWorkKind.SNAPSHOT
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

        try:
            self._camera_work_queue.put_nowait(_CameraWork(kind, data))
        except asyncio.QueueFull:
            self._camera_snapshot_backlog = max(0, self._camera_snapshot_backlog - 1)
            self._camera_logger.warning(
                "Dropped camera snapshot %s because the %d-entry queue is full.",
                data.id,
                self._CAMERA_QUEUE_MAXSIZE,
            )

    async def _resume_webcam_stream(self) -> None:
        """Resume existing stream credit after a camera handle appears."""
        self._ensure_camera_worker()
        self._queue_stream_marker()

    def _queue_stream_marker(self) -> None:
        if self._camera_stream_pending or self._request_count <= 0:
            return
        try:
            self._camera_work_queue.put_nowait(
                _CameraWork(_CameraWorkKind.STREAM, WebcamSnapshotDemandData())
            )
        except asyncio.QueueFull:
            # Credits remain in _request_count, so a later completed snapshot or
            # camera reconfiguration can enqueue the single stream marker.
            return
        self._camera_stream_pending = True

    def _ensure_camera_worker(self) -> None:
        """Lazily create the one owner of frame reads and uploads."""
        if self._camera_closing:
            return
        if self._camera_worker_task is None or self._camera_worker_task.done():
            self._camera_worker_task = asyncio.get_running_loop().create_task(
                self._camera_worker(),
                name=f"sp-camera:{self._printer.config.unique_id}",
            )

    async def _camera_worker(self) -> None:
        """Serialize camera access while allowing cloud dispatch to continue."""
        while True:
            work = await self._camera_work_queue.get()
            result = False
            error = None
            worker_cancelled = False
            try:
                if work.kind is _CameraWorkKind.STREAM and self._request_count <= 0:
                    continue

                self._camera_active_kind = work.kind
                self._camera_active_task = asyncio.get_running_loop().create_task(
                    self._process_camera_work(work),
                    name=f"sp-camera-frame:{work.kind.value}",
                )
                try:
                    result = await self._camera_active_task
                except asyncio.CancelledError:
                    # Stream-off marks the one child cancellation that is safe
                    # to consume. Any other cancellation belongs to the owner
                    # task (teardown/loop shutdown) and must propagate.
                    if not (
                        work.kind is _CameraWorkKind.STREAM
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
                    "Camera %s request failed", work.kind.value, exc_info=True
                )
            finally:
                self._camera_active_task = None
                self._camera_active_kind = None
                self._camera_stream_cancel_requested = False

                if work.kind is _CameraWorkKind.SNAPSHOT:
                    self._camera_snapshot_backlog = max(
                        0, self._camera_snapshot_backlog - 1
                    )
                    if (
                        self._camera_backlog_reported
                        and self._camera_snapshot_backlog < 10
                    ):
                        self._camera_backlog_reported = False
                elif work.kind is _CameraWorkKind.STREAM:
                    if not result and self._request_count > 0:
                        self._request_count -= 1
                    self._camera_stream_pending = False
                    if self._request_count > 0 and not self._camera_closing:
                        self._queue_stream_marker()

                if work.done is not None and not work.done.done():
                    if worker_cancelled:
                        work.done.cancel()
                    elif error is not None:
                        work.done.set_exception(error)
                    else:
                        work.done.set_result(result)
                self._camera_work_queue.task_done()
                if (
                    work.kind is not _CameraWorkKind.STREAM
                    and self._request_count > 0
                    and not self._camera_closing
                ):
                    self._queue_stream_marker()

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
            if self._simplyprint_api is None:
                raise RuntimeError("SimplyPrint API is not configured")
            await self._simplyprint_api.post_snapshot(
                data.id, frame, endpoint=data.endpoint
            )
            self._camera_logger.debug(f"Posted snapshot to API with id {data.id}")
            return False

        # Mark the webcam as connected if it's not already.
        if not self._printer.webcam_info.connected:
            self._printer.webcam_info.connected = True

        async with self._stream_lock:
            # Prevent racing between receiving frames and sending them.
            await self._printer.intervals.wait_for(Interval.WEBCAM)
            b64frame = base64.b64encode(frame).decode("utf-8")
            await self._send_stream(StreamMsg(b64frame))
            self._last_stream_frame_at = time.time()

            if self._request_count > 0:
                self._request_count -= 1

        return len(self._stream_lock) == 0 and self._request_count > 0
