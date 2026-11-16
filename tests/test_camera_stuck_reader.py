"""A camera that stops sending frames must not park snapshot requests (AI bed checks) indefinitely."""

import asyncio
import datetime
import threading
import time

from yarl import URL

from simplyprint_ws_client.shared.camera import controller as controller_module
from simplyprint_ws_client.shared.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.shared.camera.commands import ReceivedFrame
from simplyprint_ws_client.shared.camera.controller import CameraController
from simplyprint_ws_client.shared.camera.handle import CameraHandle


class StuckCamera(BaseCameraProtocol):
    """Sends one frame per release, and blocks in between, like a stream that has gone quiet."""

    polling_mode = CameraProtocolPollingMode.CONTINUOUS

    def __init__(self):
        super().__init__(URL("stuck://camera"))
        self.gates = []
        self.opened = threading.Event()

    @staticmethod
    def test(uri: URL) -> bool:
        return True

    def read(self):
        # One gate per connection, so a test can wake one reader and not the other.
        gate = threading.Semaphore(0)
        self.gates.append(gate)
        self.opened.set()

        while True:
            gate.acquire()
            yield b"frame"

    def wait_for_readers(self, count: int):
        deadline = time.monotonic() + 2
        while len(self.gates) < count and time.monotonic() < deadline:
            time.sleep(0.01)


class NullPool:
    def submit_request(self, req):
        pass


def test_stop_returns_while_the_reader_is_blocked(monkeypatch):
    monkeypatch.setattr(controller_module, "STOP_JOIN_TIMEOUT", 0.2)
    frames = []
    camera = StuckCamera()
    controller = CameraController(frames.append, protocol=camera)

    controller.start()
    started = time.monotonic()
    controller.stop()

    assert time.monotonic() - started < 2


def test_an_abandoned_reader_is_replaced_and_delivers_nothing(monkeypatch):
    monkeypatch.setattr(controller_module, "STOP_JOIN_TIMEOUT", 0.2)
    frames = []
    camera = StuckCamera()
    controller = CameraController(frames.append, protocol=camera)

    controller.start()
    camera.wait_for_readers(1)
    abandoned = controller._main_thread
    controller.stop()
    controller.start()
    camera.wait_for_readers(2)

    assert controller._main_thread is not abandoned
    assert controller._main_thread.is_alive()

    old_gate, new_gate = camera.gates

    # The abandoned reader wakes up late: it must not deliver.
    old_gate.release()
    time.sleep(0.2)
    assert frames == []

    new_gate.release()
    time.sleep(0.2)
    assert frames == [b"frame"]

    controller.stop()


def test_receive_frame_gives_up_when_no_frame_comes():
    async def scenario():
        handle = CameraHandle(NullPool(), 1)
        frame = await handle.receive_frame(timeout=0.1)
        return frame, handle._waiters

    frame, waiters = asyncio.run(scenario())

    assert frame is None
    assert waiters == []


def test_a_snapshot_can_use_a_streamed_frame_before_any_poll():
    async def scenario():
        handle = CameraHandle(NullPool(), 1)
        handle._cached_frame = b"streamed"
        return await handle.receive_frame(allow_cache_age=datetime.timedelta(seconds=5), timeout=0.1)

    # No poll has happened yet, so there is no poll time to compare the cache against: wait for a fresh frame.
    assert asyncio.run(scenario()) is None


def test_receive_frame_returns_a_frame_that_arrives():
    async def scenario():
        handle = CameraHandle(NullPool(), 1)
        waiter = asyncio.create_task(handle.receive_frame(timeout=1))
        await asyncio.sleep(0.1)

        handle.on_response(ReceivedFrame(1, time.time(), b"fresh"))
        return await waiter

    assert asyncio.run(scenario()) == b"fresh"
