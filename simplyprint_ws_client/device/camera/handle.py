import asyncio
import datetime
import threading
from typing import TYPE_CHECKING, Any, List, Optional

from simplyprint_ws_client.device.camera.base import FrameT
from simplyprint_ws_client.device.camera.commands import (
    PollCamera,
    StartCamera,
    StopCamera,
    DeleteCamera,
)
from simplyprint_ws_client.common.utils.stoppable import StoppableInterface

if TYPE_CHECKING:
    from simplyprint_ws_client.device.camera.pool import CameraPool


class CameraHandle(StoppableInterface):
    """A client's view of one camera, independent of where it runs.

    Frames arrive via :meth:`_set_frame` -- from the PROCESS pool's response
    reader, or from an inline/thread :mod:`.backends` driver -- and the same
    ``receive_frame`` / ``start`` / ``pause`` / ``stop`` surface drives whichever
    backend this handle was created with. When ``driver`` is set the commands go
    to that backend; otherwise they go to the worker ``pool``.
    """

    pool: "CameraPool"
    id: int

    _waiters: List[asyncio.Future]
    _frame_time_window: List[float]
    _last_poll_time: Optional[datetime.datetime] = None
    _cached_frame: Optional[FrameT] = None

    def __init__(
        self, pool: "CameraPool", camera_id: int, driver: Optional[Any] = None
    ):
        self.pool = pool
        self.id = camera_id
        self._driver = driver
        self._frame_time_window = []
        self._waiters = []
        self._lock = threading.Lock()

    def _set_frame(self, data: Optional[FrameT], timestamp: float) -> None:
        """Deliver one frame to this handle. Called from one producer at a time
        (the PROCESS response reader, an inline loop task, or a thread courier)."""
        ready = []
        with self._lock:
            self._frame_time_window.append(timestamp)
            if len(self._frame_time_window) > 10:
                self._frame_time_window.pop(0)

            self._cached_frame = data

            while self._waiters:
                fut = self._waiters.pop(0)
                if not fut.done():
                    ready.append(fut)

        for fut in ready:
            loop = fut.get_loop()
            loop.call_soon_threadsafe(fut.set_result, data)

    async def receive_frame(
        self, allow_cache_age: Optional[datetime.timedelta] = None
    ) -> FrameT:
        # Always ask for a new frame.
        self._poll()

        # Although we might want to serve an old frame if it's not too old.
        now = datetime.datetime.now()
        with self._lock:
            if allow_cache_age is not None and self._cached_frame is not None:
                if (
                    self._last_poll_time is not None
                    and self._last_poll_time + allow_cache_age > now
                ):
                    return self._cached_frame

                self._cached_frame = None

            self._last_poll_time = now
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
        if self._driver is not None:
            self._driver.start()
        else:
            self.pool.submit_request(StartCamera(self.id))

    def pause(self):
        if self._driver is not None:
            self._driver.pause()
        else:
            self.pool.submit_request(StopCamera(self.id))

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
        if self._driver is not None:
            self._driver.poll()
        else:
            self.pool.submit_request(PollCamera(self.id))

    # Stoppable methods

    def is_stopped(self) -> bool:
        raise NotImplementedError()

    def stop(self) -> None:
        if self._driver is not None:
            self._driver.stop()
        else:
            self.pool.submit_request(DeleteCamera(self.id))

    def clear(self) -> None:
        raise NotImplementedError()
