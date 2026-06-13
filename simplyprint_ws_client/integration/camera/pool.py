from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Type, final

from yarl import URL

from simplyprint_ws_client.integration.camera.backends import _PauseTimer
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolConnectionError,
    CameraProtocolInvalidState,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.commands import (
    DeleteCamera,
    PollCamera,
    Request,
    StartCamera,
    StopCamera,
)
from simplyprint_ws_client.integration.camera.handle import CameraHandle
from simplyprint_ws_client.common.asyncio.coalescing_task import CoalescingTask
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.stoppable import ProcessStoppable
from simplyprint_ws_client.common.utils.synchronized import Synchronized
from simplyprint_ws_client.common.worker.context import ExecutionContext
from simplyprint_ws_client.common.worker.pool import WorkerHandle, WorkerPool


async def _resolve_aiter(protocol: BaseCameraProtocol):
    result = protocol.read()
    if inspect.iscoroutine(result):
        result = await result
    return result


def _sync_camera_producer(
    emit, is_stopped, protocol: BaseCameraProtocol, continuous: bool
) -> None:
    try:
        for frame in protocol:
            if is_stopped():
                break
            emit(frame, time.time())
            if not continuous:
                break
    except (CameraProtocolConnectionError, CameraProtocolInvalidState):
        emit(None, time.time())
    except Exception as e:  # noqa: BLE001
        logging.getLogger("camera.pool").debug("camera read failed: %s", e)
        emit(None, time.time())


async def _async_camera_producer(
    emit, is_stopped, protocol: BaseCameraProtocol, continuous: bool
) -> None:
    try:
        aiter = await _resolve_aiter(protocol)
        async for frame in aiter:
            if is_stopped():
                break
            emit(frame, time.time())
            if not continuous:
                break
    except asyncio.CancelledError:
        raise
    except (CameraProtocolConnectionError, CameraProtocolInvalidState):
        emit(None, time.time())
    except Exception as e:  # noqa: BLE001
        logging.getLogger("camera.pool").debug("camera read failed: %s", e)
        emit(None, time.time())


class _Desired(Enum):
    """The state a camera's worker should converge to. Commands set it; one
    coalesced reconcile applies it -- so a storm of start/pause/poll collapses to
    a single transition instead of a worker allocate/stop per command."""

    RUNNING = auto()  #: a continuous worker should be streaming
    PAUSED = auto()  #: no worker (idle / stream off)
    STOPPED = auto()  #: released; no worker, never again


class CameraWorkerBackend:
    """Drives one camera's worker toward a desired state.

    ``poll``/``start``/``pause``/``stop`` are cheap and may be called from any
    thread (device handlers, the pause timer, the loop): they record the desired
    state and ``trigger`` a single :class:`CoalescingTask`. The reconcile retires
    the current worker (non-blocking, via the pool reaper) and allocates at most
    one new worker, so rapid toggles never pile up workers. A monotonic
    ``_generation`` guards delivery: a retired worker's late frame is dropped.
    """

    def __init__(
        self,
        worker_pool: WorkerPool,
        context: ExecutionContext,
        protocol: BaseCameraProtocol,
        handle: CameraHandle,
        release: Callable[[int], None],
        provider: EventLoopProvider,
        pause_timeout: Optional[int] = None,
    ) -> None:
        self._worker_pool = worker_pool
        self._context = context
        self._protocol = protocol
        self._handle = handle
        self._release = release
        self._pause_timeout = pause_timeout
        self._worker: Optional[WorkerHandle] = None
        self._pause_timer: Optional[_PauseTimer] = (
            _PauseTimer(pause_timeout, self.pause) if pause_timeout else None
        )
        self._lock = threading.Lock()
        self._desired = _Desired.PAUSED
        self._oneshot = False  # a one-shot (ON_DEMAND) poll is pending
        self._generation = 0
        self._reconcile = CoalescingTask(self._reconcile_once, provider=provider)

    @property
    def _continuous(self) -> bool:
        return self._protocol.polling_mode == CameraProtocolPollingMode.CONTINUOUS

    def poll(self) -> None:
        if self._continuous:
            self.start()
            return
        with self._lock:
            self._oneshot = True
        self._reconcile.trigger()

    def start(self) -> None:
        if not self._continuous:
            return
        with self._lock:
            self._desired = _Desired.RUNNING
        self._reconcile.trigger()
        self._refresh_timer()

    def pause(self) -> None:
        self._cancel_timer()
        with self._lock:
            self._desired = _Desired.PAUSED
        self._reconcile.trigger()

    def stop(self) -> None:
        self._cancel_timer()
        with self._lock:
            self._desired = _Desired.STOPPED
        # Release the pool slot now, preserving today's timing (the handle is
        # gone from the pool map immediately); the worker is retired by the
        # reconcile on a live loop, or by WorkerPool.stop at teardown.
        self._release(self._handle.id)
        self._reconcile.trigger()

    async def _reconcile_once(self) -> None:
        with self._lock:
            desired = self._desired
            oneshot = self._oneshot
            self._oneshot = False
            old = self._worker

            # Already in the wanted steady state: nothing to do (no churn).
            if desired is _Desired.RUNNING and old is not None and not oneshot:
                return

            # Otherwise retire whatever exists; a new worker is started below if
            # wanted. Bumping the generation invalidates the old worker's frames.
            self._worker = None
            self._generation += 1
            generation = self._generation

            start_continuous = desired is _Desired.RUNNING
            start_oneshot = oneshot and desired is not _Desired.STOPPED
            want_new = start_continuous or start_oneshot

        if old is not None:
            old.stop()  # non-blocking: signalled now, joined on the reaper

        if not want_new:
            return

        worker = self._allocate(continuous=start_continuous, generation=generation)
        with self._lock:
            if self._generation == generation:
                self._worker = worker
                worker = None  # installed
        if worker is not None:
            # A newer reconcile superseded us mid-allocate; retire the orphan.
            worker.stop()

    def _allocate(self, *, continuous: bool, generation: int) -> WorkerHandle:
        producer = (
            _async_camera_producer if self._protocol.is_async else _sync_camera_producer
        )

        def on_item(payload, timestamp, _generation: int = generation) -> None:
            # Drop frames from a worker that has since been retired.
            if _generation == self._generation:
                self._handle._set_frame(payload, timestamp)

        return self._worker_pool.allocate(
            self._context,
            producer,
            on_item,
            args=(self._protocol, continuous),
            is_async=self._protocol.is_async,
        )

    def _refresh_timer(self) -> None:
        if self._pause_timer is not None and self._continuous:
            self._pause_timer.touch()

    def _cancel_timer(self) -> None:
        if self._pause_timer is not None:
            self._pause_timer.cancel()


@final
class CameraPool(ProcessStoppable, Synchronized):
    protocols: List[Type[BaseCameraProtocol]]
    allocations: Dict[int, CameraHandle]

    def __init__(self, *, event_loop_provider=None, **kwargs):
        ProcessStoppable.__init__(self, **kwargs)
        Synchronized.__init__(self)

        self.protocols = []
        self.allocations = {}
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._workers = WorkerPool(event_loop_provider=self._provider)
        self._id_counter = 0

    def submit_request(self, req: Request):
        handle = self.allocations.get(req.id)
        if handle is None or handle._driver is None:
            return
        if isinstance(req, PollCamera):
            handle._driver.poll()
        elif isinstance(req, StartCamera):
            handle._driver.start()
        elif isinstance(req, StopCamera):
            handle._driver.pause()
        elif isinstance(req, DeleteCamera):
            handle._driver.stop()

    def _release(self, camera_id: int) -> None:
        with self:
            self.allocations.pop(camera_id, None)

    def _new_id(self) -> int:
        with self:
            self._id_counter += 1
            return self._id_counter

    @staticmethod
    def _matches(protocol_cls: Type[BaseCameraProtocol], uri: URL) -> bool:
        result = protocol_cls.test(uri)
        if inspect.iscoroutine(result):
            result.close()
            return False
        return bool(result)

    def _select_protocol(self, uri: URL) -> Optional[BaseCameraProtocol]:
        for protocol_cls in self.protocols:
            if self._matches(protocol_cls, uri):
                return protocol_cls(uri)
        return None

    @staticmethod
    def _route(protocol_cls: Type[BaseCameraProtocol]) -> ExecutionContext:
        if protocol_cls.execution_context is not None:
            return protocol_cls.execution_context
        if protocol_cls.is_async:
            return ExecutionContext.INLINE
        return ExecutionContext.PROCESS

    def create(self, uri: URL, *, pause_timeout: Optional[int] = None) -> CameraHandle:
        protocol = self._select_protocol(uri)
        if protocol is None:
            raise ValueError("No protocol found for URI")

        context = self._route(type(protocol))
        camera_id = self._new_id()
        handle = CameraHandle(self, camera_id)
        handle._driver = CameraWorkerBackend(
            self._workers,
            context,
            protocol,
            handle,
            self._release,
            self._provider,
            pause_timeout,
        )
        with self:
            self.allocations[camera_id] = handle
        return handle

    def stop(self):
        super().stop()
        with self:
            handles = list(self.allocations.values())
            self.allocations.clear()
        for handle in handles:
            try:
                if handle._driver is not None:
                    handle._driver.stop()
            except Exception:  # noqa: BLE001
                pass
        self._workers.stop()
