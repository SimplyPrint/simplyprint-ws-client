"""The camera pool: one :class:`CameraHandle` per configured camera source.

Where a camera runs follows its protocol (see :meth:`CameraPool._route`): an async
protocol is a task on the consumer loop, a sync one runs in the shared worker
processes of :class:`WorkerPool` -- a thread each, spread over at most one
process per core, with no limit on the number of cameras.

Protocols come from two places: the plain ``protocols`` list (fixed for the
pool's life) and :meth:`CameraPool.add_protocols`, which contributes a group
under a :class:`~simplyprint_ws_client.core.scope.Scope` and removes exactly
that group, by token, when the scope closes. Cameras already created keep
their protocol instance; removal only affects which protocol new cameras get.

A continuous camera keeps running for ``pause_timeout`` after the last request,
whether or not anyone still waits. A ``demand_driven`` protocol is told when a
frame is wanted: every poll (one per ``CameraHandle.receive_frame``) bumps a
request count that the backend sends through the worker's signal, and the
producer turns each new count into ``FrameDemand.request()`` on the protocol.
So such a camera can stay connected while it makes no frames for nobody.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    final,
)

from yarl import URL

from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolConnectionError,
    CameraProtocolInvalidState,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.handle import CameraHandle
from simplyprint_ws_client.common.asyncio.coalescing_task import CoalescingTask
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.stoppable import ProcessStoppable
from simplyprint_ws_client.common.utils.synchronized import Synchronized
from simplyprint_ws_client.common.worker.context import ExecutionContext
from simplyprint_ws_client.common.worker.pool import (
    WorkerHandle,
    WorkerPool,
    WorkerSignal,
)
from simplyprint_sdk.camera import FrameDemand

if TYPE_CHECKING:
    from simplyprint_ws_client.core.scope import Scope


#: Frames queued for the loop per camera. Latest wins: a newer frame replaces
#: one the loop has not taken yet.
_FRAME_QUEUE = 2


async def _resolve_aiter(protocol: BaseCameraProtocol):
    result = protocol.read()
    if inspect.iscoroutine(result):
        result = await result
    return result


def _attach_demand(
    protocol: BaseCameraProtocol, signal: Optional[WorkerSignal]
) -> None:
    """Give a demand-driven protocol its :class:`FrameDemand`, fed by the
    owner's signal: the count of frames requested so far. A count above the
    last one seen is a new request (several requests between two looks are one
    request: one frame serves every waiter)."""
    if signal is None:
        return
    demand = FrameDemand()
    protocol.demand = demand
    seen = [0]

    def on_count(count: int) -> None:
        # Called one at a time (WorkerSignal holds its lock), so no race on seen.
        if count > seen[0]:
            seen[0] = count
            demand.request()

    signal.listen(on_count)


def _sync_camera_producer(
    emit,
    is_stopped,
    protocol: BaseCameraProtocol,
    continuous: bool,
    signal: Optional[WorkerSignal] = None,
) -> None:
    _attach_demand(protocol, signal)
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
    emit,
    is_stopped,
    protocol: BaseCameraProtocol,
    continuous: bool,
    signal: Optional[WorkerSignal] = None,
) -> None:
    _attach_demand(protocol, signal)
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
        #: Frames requested so far (demand-driven protocols): the signal value.
        self._requested = 0
        self._generation = 0
        self._reconcile = CoalescingTask(self._reconcile_once, provider=provider)

    @property
    def _continuous(self) -> bool:
        return self._protocol.polling_mode == CameraProtocolPollingMode.CONTINUOUS

    @property
    def _demand_driven(self) -> bool:
        return self._continuous and self._protocol.demand_driven

    def poll(self) -> None:
        """A client waits for a frame (``CameraHandle.receive_frame``): every
        snapshot, live-stream frame, timelapse shot or detection pass."""
        if self._continuous:
            self.start()
            if self._demand_driven:
                self._request_frame()
            return
        with self._lock:
            self._oneshot = True
        self._reconcile.trigger()

    def _request_frame(self) -> None:
        """Tell the running producer one more frame is wanted. Without a worker
        yet (or while one is replaced) the reconcile sends the count to the
        new worker instead."""
        with self._lock:
            self._requested += 1
            count = self._requested
            worker = self._worker
        if worker is not None:
            worker.signal(count)

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
            if (
                desired is _Desired.RUNNING
                and old is not None
                and not old.stopped
                and not oneshot
            ):
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

        try:
            worker = self._allocate(continuous=start_continuous, generation=generation)
        except Exception:  # noqa: BLE001 -- e.g. an unpicklable protocol
            logging.getLogger("camera.pool").warning(
                "could not start camera %s", self._handle.id, exc_info=True
            )
            self._handle.deliver_frame(None, time.time())  # fail waiters now
            return
        installed: Optional[WorkerHandle] = None
        with self._lock:
            if self._generation == generation:
                self._worker = installed = worker
                worker = None  # installed
            count = self._requested
        if worker is not None:
            # A newer reconcile superseded us mid-allocate; retire the orphan.
            worker.stop()
        elif installed is not None and count and self._demand_driven:
            # A request may have gone to the retired worker, or come while no
            # worker ran: the new one makes a frame for it. At worst that is one
            # frame nobody waits for, once per worker start, never a lost one.
            installed.signal(count)

    def _allocate(self, *, continuous: bool, generation: int) -> WorkerHandle:
        producer = (
            _async_camera_producer if self._protocol.is_async else _sync_camera_producer
        )

        def on_item(payload, timestamp, _generation: int = generation) -> None:
            # Drop frames from a worker that has since been retired.
            if _generation == self._generation:
                self._handle.deliver_frame(payload, timestamp)

        return self._worker_pool.allocate(
            self._context,
            producer,
            on_item,
            args=(self._protocol, continuous),
            is_async=self._protocol.is_async,
            maxsize=_FRAME_QUEUE,
            # A continuous worker of a demand-driven protocol hears each
            # request; a one-shot read makes its one frame regardless.
            signal=continuous and self._demand_driven,
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

    def __init__(
        self,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        process_workers: int = 0,
    ) -> None:
        """``process_workers`` caps the worker processes sync cameras share
        (``0``: one per core). It never limits how many cameras run."""
        ProcessStoppable.__init__(self)
        Synchronized.__init__(self)

        self.protocols = []
        # (token, protocols) groups contributed under a scope; copy-on-write.
        self._scoped_protocols: Tuple[
            Tuple[object, Tuple[Type[BaseCameraProtocol], ...]], ...
        ] = ()
        self.allocations = {}
        self._provider = event_loop_provider or EventLoopProvider.default()
        if process_workers < 0:
            raise ValueError("process_workers must not be negative")
        debug = "SIMPLYPRINT_DEBUG_CAMERA" in os.environ
        if debug:
            # Worker read failures log at DEBUG on this logger.
            logging.getLogger("camera.pool").setLevel(logging.DEBUG)
        self._workers = WorkerPool(
            event_loop_provider=self._provider,
            max_processes=process_workers or None,
            log_level=logging.DEBUG if debug else None,
        )
        self.process_workers = self._workers.max_processes
        self._id_counter = 0

    def add_protocols(
        self, protocols: Iterable[Type[BaseCameraProtocol]], scope: "Scope"
    ) -> None:
        """Offer ``protocols`` to new cameras until ``scope`` closes.

        They are tried after the ones already present, in the given order. The
        undo removes this group by token, so the same class contributed by two
        scopes (or equal to one already listed) stays until its own scope goes.
        """
        group = tuple(protocols)
        if not group:
            return
        token = object()
        with self:
            self._scoped_protocols = self._scoped_protocols + ((token, group),)
        scope.on_close(lambda: self._remove_protocols(token), name="camera-protocols")

    def _remove_protocols(self, token: object) -> None:
        with self:
            self._scoped_protocols = tuple(
                entry for entry in self._scoped_protocols if entry[0] is not token
            )

    def all_protocols(self) -> Tuple[Type[BaseCameraProtocol], ...]:
        """Every protocol a new camera may get, in the order they are tried."""
        scoped = self._scoped_protocols
        return tuple(self.protocols) + tuple(
            protocol for _token, group in scoped for protocol in group
        )

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
        for protocol_cls in self.all_protocols():
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
        handle = CameraHandle(camera_id)
        handle.attach_backend(
            CameraWorkerBackend(
                self._workers,
                context,
                protocol,
                handle,
                self._release,
                self._provider,
                pause_timeout,
            )
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
                handle.backend.stop()
            except Exception:  # noqa: BLE001
                pass
        self._workers.stop()
