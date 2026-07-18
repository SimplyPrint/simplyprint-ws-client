import asyncio
import datetime
import threading
import time
from typing import List, Optional, Protocol

from simplyprint_ws_client.integration.camera.base import FrameT


class CameraBackend(Protocol):
    def poll(self) -> None: ...

    def start(self) -> None: ...

    def pause(self) -> None: ...

    def stop(self) -> None: ...


class CameraHandle:
    """A client's view of one camera, independent of where it runs.

    Frames arrive via :meth:`deliver_frame` -- from the worker pool's response
    reader, or from an inline/thread :mod:`.backends` driver -- and the same
    ``receive_frame`` / ``start`` / ``pause`` / ``stop`` surface drives whichever
    backend this handle was created with.
    """

    id: int

    _waiters: List[asyncio.Future]
    _frame_time_window: List[float]
    _cached_frame: Optional[FrameT] = None
    _cached_frame_at: Optional[float] = None
    """``time.time()`` timestamp of when ``_cached_frame`` arrived (the
    producer's emit time), so cache decisions are about the *frame's* age --
    never about how recently somebody polled."""

    def __init__(
        self,
        camera_id: int,
        backend: Optional[CameraBackend] = None,
    ) -> None:
        self.id = camera_id
        self._backend: Optional[CameraBackend] = None
        if backend is not None:
            self.attach_backend(backend)
        self._frame_time_window = []
        self._waiters = []
        self._lock = threading.Lock()

    @property
    def backend(self) -> CameraBackend:
        if self._backend is None:
            raise RuntimeError("camera backend is not attached")
        return self._backend

    def attach_backend(self, backend: CameraBackend) -> None:
        if self._backend is not None:
            raise RuntimeError("camera backend is already attached")
        self._backend = backend

    def deliver_frame(self, data: Optional[FrameT], timestamp: float) -> None:
        """Deliver one frame to this handle. Called from one producer at a time
        (the PROCESS response reader, an inline loop task, or a thread courier)."""
        ready = []
        with self._lock:
            self._frame_time_window.append(timestamp)
            if len(self._frame_time_window) > 10:
                self._frame_time_window.pop(0)

            self._cached_frame = data
            self._cached_frame_at = timestamp if data is not None else None

            while self._waiters:
                fut = self._waiters.pop(0)
                if not fut.done():
                    ready.append(fut)

        for fut in ready:
            loop = fut.get_loop()
            loop.call_soon_threadsafe(self._resolve_waiter, fut, data)

    @staticmethod
    def _resolve_waiter(fut: asyncio.Future, data: Optional[FrameT]) -> None:
        # The waiter may be cancelled after the producer releases ``_lock`` but
        # before this callback runs on its event loop.
        if not fut.done():
            fut.set_result(data)

    async def receive_frame(
        self, allow_cache_age: Optional[datetime.timedelta] = None
    ) -> FrameT:
        # Always ask for a new frame (for continuous cameras this also
        # (re)starts a paused worker and refreshes its pause timer).
        now = time.time()
        self._poll()

        # Serve the cached frame when the *frame itself* is fresh enough --
        # this is the zero-latency path for live streams (the worker pushes
        # frames continuously; a demand should never block on the next one
        # when an unseen frame is already here).
        with self._lock:
            if allow_cache_age is not None and self._cached_frame is not None:
                if (
                    self._cached_frame_at is not None
                    and self._cached_frame_at + allow_cache_age.total_seconds() > now
                ):
                    return self._cached_frame

                self._cached_frame = None
                self._cached_frame_at = None

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self._lock:
            self._waiters.append(fut)
        try:
            return await fut
        except asyncio.CancelledError:
            with self._lock:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def start(self):
        self.backend.start()

    def pause(self):
        self.backend.pause()

    @property
    def fps(self) -> float:
        # Calculate the average FPS from the last 10 frames.
        with self._lock:
            if len(self._frame_time_window) < 2:
                return 0

            elapsed_time = self._frame_time_window[-1] - self._frame_time_window[0]
            num_intervals = len(self._frame_time_window) - 1

        if elapsed_time <= 0:
            return 0

        return num_intervals / elapsed_time

    def _poll(self):
        self.backend.poll()

    def stop(self) -> None:
        self.backend.stop()
