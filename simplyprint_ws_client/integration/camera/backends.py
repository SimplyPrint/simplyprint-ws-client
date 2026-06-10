"""Async camera execution -- the non-process backends a camera handle drives.

A camera that doesn't need CPU isolation shouldn't pay for a subprocess or pickle
a frame across it. Two backends cover that:

* :class:`InlineCameraBackend` -- an async protocol run as a task **directly on
  the consumer loop**. No process, no thread, no IPC: a frame is handed to the
  handle in-process. This is the "async camera" case.
* :class:`ThreadCameraBackend` -- a protocol (async or sync) run in its **own
  thread** (its own loop, if async); frames ride a :class:`Courier` back onto the
  consumer loop. This is the "async-but-in-another-thread" case, and also lets a
  light sync camera stream off the main loop.

Both expose the same backend surface the handle calls -- ``poll`` / ``start`` /
``pause`` / ``stop`` -- and feed frames via ``handle._set_frame(data, ts)``. The
PROCESS path keeps its own machinery in :mod:`.pool`; these are its peers.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolConnectionError,
    CameraProtocolInvalidState,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.camera.handle import CameraHandle


async def _resolve_aiter(protocol: BaseCameraProtocol):
    """Get the async iterator for ``protocol``: ``read()`` may return it directly
    (an async generator) or a coroutine that yields it."""
    result = protocol.read()
    if inspect.iscoroutine(result):
        result = await result
    return result


class InlineCameraBackend:
    """An async camera driven as a task on the consumer loop (no process/thread)."""

    def __init__(
        self,
        protocol: BaseCameraProtocol,
        handle: "CameraHandle",
        provider: EventLoopProvider,
        pause_timeout: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._protocol = protocol
        self._handle = handle
        self._provider = provider
        self._pause_timeout = pause_timeout
        self._logger = logger or logging.getLogger("camera.inline")
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._pause_timer: Optional[asyncio.TimerHandle] = None

    @property
    def _continuous(self) -> bool:
        return self._protocol.polling_mode == CameraProtocolPollingMode.CONTINUOUS

    def poll(self) -> None:
        self._provider.event_loop.call_soon_threadsafe(self._poll_on_loop)

    def start(self) -> None:
        self._provider.event_loop.call_soon_threadsafe(self._start_on_loop)

    def pause(self) -> None:
        self._provider.event_loop.call_soon_threadsafe(self._pause_on_loop)

    def stop(self) -> None:
        self.pause()

    def _poll_on_loop(self) -> None:
        if self._continuous:
            self._start_on_loop()
            self._refresh_timer()
        else:
            self._provider.event_loop.create_task(self._run(once=True))

    def _start_on_loop(self) -> None:
        if not self._continuous:
            return
        if self._task is not None and not self._task.done():
            return
        self._stopped = False
        self._refresh_timer()
        self._task = self._provider.event_loop.create_task(self._run(once=False))

    def _pause_on_loop(self) -> None:
        self._stopped = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._cancel_timer()

    def _refresh_timer(self) -> None:
        if not self._pause_timeout or not self._continuous:
            return
        self._cancel_timer()
        self._pause_timer = self._provider.event_loop.call_later(
            self._pause_timeout, self._pause_on_loop
        )

    def _cancel_timer(self) -> None:
        if self._pause_timer is not None:
            self._pause_timer.cancel()
            self._pause_timer = None

    async def _run(self, once: bool) -> None:
        try:
            aiter = await _resolve_aiter(self._protocol)
            async for frame in aiter:
                if not once and self._stopped:
                    break
                self._handle._set_frame(frame, time.time())
                if once:
                    break
        except asyncio.CancelledError:
            raise
        except (CameraProtocolConnectionError, CameraProtocolInvalidState):
            self._handle._set_frame(None, time.time())
        except Exception as e:  # noqa: BLE001
            self._logger.debug("inline camera read failed: %s", e)
            self._handle._set_frame(None, time.time())


class _PauseTimer:
    """One rescheduling pause timer instead of a new ``threading.Timer`` per poll.

    ``touch()`` pushes the deadline; the single timer thread re-checks at the
    deadline and only fires ``on_expire`` when no touch arrived in between.
    """

    def __init__(self, timeout: float, on_expire) -> None:
        self._timeout = timeout
        self._on_expire = on_expire
        self._deadline = 0.0
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def touch(self) -> None:
        with self._lock:
            self._deadline = time.monotonic() + self._timeout
            if self._timer is None:
                self._schedule(self._timeout)

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _schedule(self, delay: float) -> None:
        self._timer = threading.Timer(delay, self._check)
        self._timer.daemon = True
        self._timer.start()

    def _check(self) -> None:
        with self._lock:
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                self._schedule(remaining)
                return
            self._timer = None
        self._on_expire()


class ThreadCameraBackend:
    """A camera driven in its own thread; frames couriered onto the consumer loop."""

    def __init__(
        self,
        protocol: BaseCameraProtocol,
        handle: "CameraHandle",
        provider: EventLoopProvider,
        pause_timeout: Optional[int] = None,
        overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._protocol = protocol
        self._handle = handle
        self._pause_timeout = pause_timeout
        self._logger = logger or logging.getLogger("camera.thread")
        self._courier: Courier = Courier(
            sink=self._deliver, provider=provider, policy=overflow, maxsize=4
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pause_timer: Optional[_PauseTimer] = (
            _PauseTimer(pause_timeout, self.pause) if pause_timeout else None
        )

    @property
    def _continuous(self) -> bool:
        return self._protocol.polling_mode == CameraProtocolPollingMode.CONTINUOUS

    def _deliver(self, item) -> None:
        frame, timestamp = item
        self._handle._set_frame(frame, timestamp)

    def poll(self) -> None:
        if self._continuous:
            self.start()
            self._refresh_timer()
            return
        # ON_DEMAND: one-shot read; overlapping requests coalesce into the
        # read already in flight instead of stacking a thread per request.
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, args=(True,), daemon=True)
        self._thread.start()

    def start(self) -> None:
        if not self._continuous:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._refresh_timer()
        self._thread = threading.Thread(target=self._run, args=(False,), daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._stop.set()
        self._cancel_timer()

    def stop(self) -> None:
        self.pause()
        self._courier.close()

    def _refresh_timer(self) -> None:
        if self._pause_timer is not None and self._continuous:
            self._pause_timer.touch()

    def _cancel_timer(self) -> None:
        if self._pause_timer is not None:
            self._pause_timer.cancel()

    def _emit(self, frame) -> None:
        self._courier.post((frame, time.time()))

    def _run(self, once: bool) -> None:
        try:
            if self._protocol.is_async:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._adrive(once))
                finally:
                    loop.close()
            else:
                self._sdrive(once)
        except (CameraProtocolConnectionError, CameraProtocolInvalidState):
            self._emit(None)
        except Exception as e:  # noqa: BLE001
            self._logger.debug("threaded camera read failed: %s", e)
            self._emit(None)

    def _sdrive(self, once: bool) -> None:
        for frame in self._protocol:
            self._emit(frame)
            if once or self._stop.is_set():
                break

    async def _adrive(self, once: bool) -> None:
        aiter = await _resolve_aiter(self._protocol)
        async for frame in aiter:
            self._emit(frame)
            if once or self._stop.is_set():
                break
