import logging
import threading
from typing import final, Optional, Callable

from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
    FrameT,
    CameraProtocolInvalidState,
    CameraProtocolConnectionError,
)
from simplyprint_ws_client.common.utils.stoppable import SyncStoppable
from simplyprint_ws_client.common.utils.synchronized import Synchronized


@final
class CameraController(SyncStoppable, Synchronized):
    """A collection of camera protocol and its state"""

    protocol: BaseCameraProtocol
    pause_timeout: Optional[int]
    logger: logging.Logger

    _frame_cb: Callable[[Optional[FrameT]], None]
    _pause_timer: Optional[threading.Timer] = None
    _main_thread: Optional[threading.Thread] = None

    def __init__(
        self,
        frame_cb: Callable[[Optional[FrameT]], None],
        protocol: BaseCameraProtocol,
        pause_timeout: Optional[int] = None,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        SyncStoppable.__init__(self)
        Synchronized.__init__(self)

        self.protocol = protocol
        self.pause_timeout = pause_timeout
        self.logger = logger
        self._frame_cb = frame_cb

    def __del__(self):
        self.stop()

    def poll(self):
        if self.protocol.polling_mode == CameraProtocolPollingMode.ON_DEMAND:
            self._exception_handler(self._read_frame)
            return

        self._refresh_timer()
        self.start()

    def start(self):
        if self.protocol.polling_mode != CameraProtocolPollingMode.CONTINUOUS:
            return

        if self._main_thread is not None and self._main_thread.is_alive():
            return

        self.clear()
        self._refresh_timer()
        self._main_thread = threading.Thread(
            target=self._exception_handler, args=(self._read_loop,), daemon=True
        )
        self._main_thread.start()

    def stop(self):
        super().stop()

        if self._main_thread:
            self._main_thread.join()
            self._main_thread = None

        if self._pause_timer:
            self._pause_timer.cancel()
            self._pause_timer = None

    def _refresh_timer(self):
        if self.protocol.polling_mode != CameraProtocolPollingMode.CONTINUOUS:
            return

        if self._pause_timer:
            self._pause_timer.cancel()
            self._pause_timer = None

        if self.pause_timeout:
            self._pause_timer = threading.Timer(self.pause_timeout, self.stop)
            self._pause_timer.daemon = True
            self._pause_timer.start()

    def _read_frame(self):
        """Read a single frame"""
        for frame in self.protocol:
            self._frame_cb(frame)
            break

    def _read_loop(self):
        """Continuous frame reading loop"""
        for frame in self.protocol:
            if self.is_stopped():
                break

            self._frame_cb(frame)

    def _exception_handler(self, f: Callable):
        try:
            f()
        except CameraProtocolConnectionError:
            self._frame_cb(None)
        except CameraProtocolInvalidState:
            self._frame_cb(None)
        except Exception as e:
            _ = e
            self.logger.debug("Failed to read from camera: %s", e)
            self._frame_cb(None)
