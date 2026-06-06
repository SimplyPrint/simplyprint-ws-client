import asyncio
import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from .base import FrameT
from .commands import (
    PollCamera,
    StartCamera,
    StopCamera,
    DeleteCamera,
)
from ..utils.stoppable import StoppableInterface

if TYPE_CHECKING:
    from .pool import CameraPool


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

    def _set_frame(self, data: Optional[FrameT], timestamp: float) -> None:
        """Deliver one frame to this handle. Called from one producer at a time
        (the PROCESS response reader, an inline loop task, or a thread courier)."""
        # Keep track of last 10 frame times for the FPS estimate.
        self._frame_time_window.append(timestamp)
        if len(self._frame_time_window) > 10:
            self._frame_time_window.pop(0)

        self._cached_frame = data

        while self._waiters:
            fut = self._waiters.pop(0)
            if fut.done():
                continue
            loop = fut.get_loop()
            loop.call_soon_threadsafe(fut.set_result, data)

    async def receive_frame(
        self, allow_cache_age: Optional[datetime.timedelta] = None
    ) -> FrameT:
        # Always ask for a new frame.
        self._poll()

        # Although we might want to serve an old frame if it's not too old.
        if allow_cache_age is not None and self._cached_frame is not None:
            if (
                self._last_poll_time is not None
                and self._last_poll_time + allow_cache_age > datetime.datetime.now()
            ):
                return self._cached_frame

            self._cached_frame = None

        self._last_poll_time = datetime.datetime.now()
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._waiters.append(fut)
        return await fut

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
