import asyncio
import datetime
from typing import TYPE_CHECKING, List, Optional

from .base import FrameT
from .commands import (
    Response,
    PollCamera,
    StartCamera,
    StopCamera,
    DeleteCamera,
    ReceivedFrame,
)
from ..utils.stoppable import StoppableInterface

if TYPE_CHECKING:
    from .pool import CameraPool

# How long a caller waits for a frame (seconds). Without a bound, a camera that sends nothing parks every
# snapshot request for it, AI bed checks included, until something restarts it.
RECEIVE_FRAME_TIMEOUT = 15.0


class CameraHandle(StoppableInterface):
    pool: "CameraPool"
    id: int

    _waiters: List[asyncio.Future]
    _frame_time_window: List[float]
    _last_poll_time: Optional[datetime.datetime] = None
    _cached_frame: Optional[FrameT] = None

    def __init__(self, pool: "CameraPool", camera_id: int):
        self.pool = pool
        self.id = camera_id
        self._frame_time_window = []
        self._waiters = []

    def on_response(self, res: Response):
        # Called from one thread only.
        if not isinstance(res, ReceivedFrame):
            return

        # Keep track of last 10 frame times
        self._frame_time_window.append(res.time)

        if len(self._frame_time_window) > 10:
            self._frame_time_window.pop(0)

        self._cached_frame = res.data

        while self._waiters:
            fut = self._waiters.pop(0)

            if fut.done():
                continue

            loop = fut.get_loop()
            loop.call_soon_threadsafe(fut.set_result, res.data)

    async def receive_frame(
        self,
        allow_cache_age: Optional[datetime.timedelta] = None,
        timeout: Optional[float] = RECEIVE_FRAME_TIMEOUT,
    ) -> Optional[FrameT]:
        """A fresh frame, or None if the camera sends none within `timeout` seconds."""
        # Always ask for a new frame
        self.pool.submit_request(PollCamera(self.id))

        # Although we might want to serve an old frame if it's not too old
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

        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            if fut in self._waiters:
                self._waiters.remove(fut)

            return None

    def start(self):
        self.pool.submit_request(StartCamera(self.id))

    def pause(self):
        self.pool.submit_request(StopCamera(self.id))

    @property
    def fps(self) -> float:
        # Calculate the average FPS from the last 10 frames
        if len(self._frame_time_window) < 2:
            return 0

        elapsed_time = self._frame_time_window[-1] - self._frame_time_window[0]
        num_intervals = len(self._frame_time_window) - 1

        if elapsed_time <= 0:
            return 0

        return num_intervals / elapsed_time

    # Stoppable methods

    def is_stopped(self) -> bool:
        raise NotImplementedError()

    def stop(self) -> None:
        self.pool.submit_request(DeleteCamera(self.id))

    def clear(self) -> None:
        raise NotImplementedError()
