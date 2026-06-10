from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import threading
import time
from typing import Callable, Dict, List, Optional, Type, final

from yarl import URL

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


class CameraWorkerBackend:
    def __init__(
        self,
        worker_pool: WorkerPool,
        context: ExecutionContext,
        protocol: BaseCameraProtocol,
        handle: CameraHandle,
        release: Callable[[int], None],
        pause_timeout: Optional[int] = None,
    ) -> None:
        self._worker_pool = worker_pool
        self._context = context
        self._protocol = protocol
        self._handle = handle
        self._release = release
        self._pause_timeout = pause_timeout
        self._worker: Optional[WorkerHandle] = None
        self._pause_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    @property
    def _continuous(self) -> bool:
        return self._protocol.polling_mode == CameraProtocolPollingMode.CONTINUOUS

    def poll(self) -> None:
        if self._continuous:
            self.start()
            self._refresh_timer()
            return
        self._replace_worker(continuous=False)

    def start(self) -> None:
        if not self._continuous:
            return
        with self._lock:
            if self._worker is not None:
                return
        self._replace_worker(continuous=True)
        self._refresh_timer()

    def pause(self) -> None:
        self._cancel_timer()
        with self._lock:
            worker = self._worker
            self._worker = None
        if worker is not None:
            worker.stop()

    def stop(self) -> None:
        self.pause()
        self._release(self._handle.id)

    def _replace_worker(self, *, continuous: bool) -> None:
        with self._lock:
            old_worker = self._worker
            self._worker = None
        if old_worker is not None:
            old_worker.stop()

        producer = (
            _async_camera_producer if self._protocol.is_async else _sync_camera_producer
        )
        worker = self._worker_pool.allocate(
            self._context,
            producer,
            self._handle._set_frame,
            args=(self._protocol, continuous),
            is_async=self._protocol.is_async,
        )
        with self._lock:
            self._worker = worker

    def _refresh_timer(self) -> None:
        if not self._pause_timeout or not self._continuous:
            return
        self._cancel_timer()
        self._pause_timer = threading.Timer(self._pause_timeout, self.pause)
        self._pause_timer.daemon = True
        self._pause_timer.start()

    def _cancel_timer(self) -> None:
        if self._pause_timer is not None:
            self._pause_timer.cancel()
            self._pause_timer = None


@final
class CameraPool(ProcessStoppable, Synchronized):
    protocols: List[Type[BaseCameraProtocol]]
    allocations: Dict[int, CameraHandle]

    def __init__(self, pool_size=0, *, event_loop_provider=None, **kwargs):
        ProcessStoppable.__init__(self, **kwargs)
        Synchronized.__init__(self)

        self._pool_size = pool_size or multiprocessing.cpu_count()
        self.protocols = []
        self.allocations = {}
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._workers = WorkerPool(event_loop_provider=self._provider)
        self._id_counter = 0

    @property
    def pool_size(self):
        return self._pool_size

    @pool_size.setter
    def pool_size(self, value):
        self._pool_size = value

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
