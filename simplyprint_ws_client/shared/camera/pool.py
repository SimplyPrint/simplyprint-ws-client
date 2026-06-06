import functools
import inspect
import logging
import multiprocessing
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from typing import final, List, Dict, Tuple, Optional, Type

from yarl import URL

from .backends import InlineCameraBackend, ThreadCameraBackend
from .base import BaseCameraProtocol, FrameT
from .commands import (
    Request,
    CreateCamera,
    PollCamera,
    StartCamera,
    StopCamera,
    DeleteCamera,
)
from .controller import CameraController
from .handle import CameraHandle
from ..asyncio.event_loop_provider import EventLoopProvider
from ..utils.stoppable import ProcessStoppable, StoppableProcess
from ..utils.synchronized import Synchronized
from ..worker.channel import SharedSlabChannel, SlabLease
from ..worker.context import ExecutionContext
from ...const import APP_DIRS


@final
class CameraWorkerProcess(StoppableProcess, Synchronized):
    """Entrypoint of camera worker process"""

    # Shared
    count: multiprocessing.Value
    command_queue: multiprocessing.Queue

    # External (pool): the parent owns this zero-copy frame channel; the child
    # attaches a non-owner view in `_run`. Set by the pool before `start()`.
    channel: Optional[SharedSlabChannel] = None
    channel_args: Optional[tuple] = None
    thread: Optional[threading.Thread] = None

    # Local (process)
    instances: Dict[int, CameraController]
    _tx: Optional[SharedSlabChannel] = None

    def __init__(self, **kwargs):
        StoppableProcess.__init__(self, **kwargs)
        self.count = multiprocessing.Value("i", 0)
        self.command_queue = multiprocessing.Queue()

    def on_request(self, req: Request):
        try:
            # Create a new camera instance.
            if isinstance(req, CreateCamera):
                with self:
                    self.instances[req.id] = CameraController(
                        functools.partial(self._send_frame, req.id),
                        protocol=req.protocol,
                        pause_timeout=req.pause_timeout,
                    )
                    self.count.value += 1

                    return

            # Execute command on a camera instance.
            with self:
                instance = self.instances.get(req.id)

            if not instance:
                logging.debug("Instance not found %s", req.id)
                return

            with instance:
                if isinstance(req, PollCamera):
                    instance.poll()
                elif isinstance(req, StartCamera):
                    instance.start()
                elif isinstance(req, StopCamera):
                    instance.stop()
                elif isinstance(req, DeleteCamera):
                    with self:
                        self.count.value -= 1
                        _ = self.instances.pop(req.id, None)
                else:
                    logging.debug("Unknown command %s", req)
        except Exception as e:
            logging.debug("Error", exc_info=e)

    def _send_frame(self, camera_id: int, frame: Optional[FrameT]):
        # Zero-copy: write the JPEG into a shared slab, send only tiny metadata.
        # `camera_id` is the channel's producer id, so one process can host many
        # cameras over one channel. Thread-safe (see SharedSlabChannel.send).
        if self._tx is not None:
            self._tx.send(camera_id, frame, time.time())

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            logging.info("Exiting on keyboard interrupt")
        except Exception as e:
            logging.error("Error", exc_info=e)

    def _run(self):
        Synchronized.__init__(self)
        self.instances = {}

        # Drop the inherited owner channel (forked from the parent) so we never
        # unlink it, and attach a non-owner view to send frames through.
        self.channel = None
        self._tx = SharedSlabChannel.attach(self.channel_args)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(process)d] %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                RotatingFileHandler(
                    filename=APP_DIRS.user_log_path / "camera_worker.log",
                    mode="a",
                    maxBytes=10 * 1024 * 1024,
                    backupCount=3,
                )
            ],
            force=True,
        )

        logging.debug("Camera worker started")

        with ThreadPoolExecutor(thread_name_prefix="CameraWorkerProcess") as tp:
            while not self.is_stopped():
                msg: Optional[Request] = self.command_queue.get()

                if msg is None:
                    break

                logging.debug(f"Received command {msg}")

                tp.submit(self.on_request, msg)

        if self._tx is not None:
            self._tx.close()  # non-owner: detaches, does not unlink

        logging.info("Exiting")

    def stop(self):
        super().stop()
        self.command_queue.put(None)  # break the command loop; the parent reader
        # closes (and unlinks) the owner channel after its loop exits.


# How many instances do we allow per process
_INSTANCES_PER_PROCESS = 10


@final
class CameraPool(ProcessStoppable, Synchronized):
    processes: List[CameraWorkerProcess]
    protocols: List[Type[BaseCameraProtocol]]
    allocations: Dict[int, Tuple[int, CameraHandle]]

    __cur_idx: int = 0

    def __init__(self, pool_size=0, *, event_loop_provider=None, **kwargs):
        ProcessStoppable.__init__(self, **kwargs)
        Synchronized.__init__(self)

        self.processes = []
        self.protocols = []
        self.allocations = {}
        # INLINE/THREAD camera handles (not process-backed); stopped on pool stop.
        self._driver_handles: List[CameraHandle] = []
        # Where INLINE/THREAD cameras deliver frames. PROCESS cameras don't need
        # it (their reader thread already targets the awaiting future's loop).
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._id_counter = 0

        self.pool_size = pool_size or multiprocessing.cpu_count()

    def _create_worker_process(self):
        return CameraWorkerProcess(daemon=True, parent_stoppable=self)

    def _consume_responses(self, process: CameraWorkerProcess):
        channel = process.channel
        try:
            while not self.is_stopped() and not process.is_stopped():
                lease = channel.recv(timeout=0.5)
                if lease is None:  # timeout, or the worker/pipe is gone
                    continue
                self._deliver_lease(lease)
        finally:
            channel.close()  # owner: close + unlink the segment

    def _pool_size(self):
        return len(self.processes)

    @property
    def pool_size(self):
        with self:
            return self._pool_size()

    @pool_size.setter
    def pool_size(self, value):
        with self:
            prev = self._pool_size()

            # No change
            if prev == value:
                return

            # Increase pool size (spawn happens deferred)
            if prev < value:
                self.processes.extend(
                    [self._create_worker_process() for _ in range(value - prev)]
                )
                return

            # Reduce pool size (terminate processes)
            self.processes, excess = self.processes[:value], self.processes[value:]

            for i, process in enumerate(excess):
                i = value + i

                # Remove allocation
                for uuid, (idx, _) in list(self.allocations.items()):
                    if idx != i:
                        continue

                    self.allocations.pop(uuid, None)

                process.stop()
                # Let the reader thread observe the stop, drain, and close+unlink
                # the channel so no shared-memory segment is left behind.
                if process.thread is not None:
                    process.thread.join(timeout=1.0)

    def _start_process(self, process: CameraWorkerProcess):
        with self:
            if process.is_alive() or process.is_stopped():
                return

            # Allocate the zero-copy frame channel lazily, only for a process that
            # actually starts (so an idle pool reserves no shared memory). The
            # child inherits channel_args across the fork in start().
            process.channel = SharedSlabChannel.create()
            process.channel_args = process.channel.child_args()

            process.start()

            process.thread = threading.Thread(
                target=self._consume_responses,
                args=(process,),
                daemon=True,
            )
            process.thread.start()

    def _next_process_idx(self):
        # Requires ownership of self to call this function.

        pool_size = self._pool_size()

        idx = self.__cur_idx

        if idx >= pool_size:
            idx = 0

        cur_process = self.processes[idx]

        # Keep allocating to the same process until it reaches the limit
        if cur_process.count.value < _INSTANCES_PER_PROCESS:
            return idx

        self.__cur_idx = (self.__cur_idx + 1) % pool_size
        return idx

    def submit_request(self, req: Request):
        if req.id not in self.allocations:
            return

        process_idx, _ = self.allocations[req.id]

        if process_idx is None:
            return

        process = self.processes[process_idx]

        if process is None:
            return

        process.command_queue.put(req)

        self._start_process(process)

    def _deliver_lease(self, lease: SlabLease):
        allocation = self.allocations.get(lease.producer_id)
        if allocation is None:
            lease.release()
            return
        _, handle = allocation
        if handle is None:
            lease.release()
            return
        try:
            # Copy out of the slab (one memcpy) so it can recycle immediately;
            # _set_frame caches the bytes and resolves the awaiting future on the
            # right loop.
            handle._set_frame(lease.to_bytes(), lease.timestamp)
        finally:
            lease.release()

    def _new_id(self) -> int:
        with self:
            self._id_counter += 1
            return self._id_counter

    @staticmethod
    def _matches(protocol_cls: Type[BaseCameraProtocol], uri: URL) -> bool:
        result = protocol_cls.test(uri)
        if inspect.iscoroutine(result):
            # An async test() can't be resolved from this synchronous path -- a
            # protocol must offer a sync test() (a URI-scheme check). Don't leak
            # the coroutine.
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
        """Where a protocol runs. Explicit override wins; otherwise an async
        protocol runs INLINE (on the loop, no process/pickle) and a sync protocol
        runs in a worker PROCESS (the proven CPU-isolated path)."""
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

        if context is ExecutionContext.PROCESS:
            return self._create_process_camera(camera_id, protocol, pause_timeout)

        handle = CameraHandle(self, camera_id)
        if context is ExecutionContext.INLINE:
            if not protocol.is_async:
                raise ValueError("INLINE execution requires an async camera protocol")
            handle._driver = InlineCameraBackend(
                protocol, handle, self._provider, pause_timeout
            )
        else:  # THREAD
            handle._driver = ThreadCameraBackend(
                protocol, handle, self._provider, pause_timeout
            )
        with self:
            self._driver_handles.append(handle)
        return handle

    def _create_process_camera(
        self, camera_id: int, protocol: BaseCameraProtocol, pause_timeout: Optional[int]
    ) -> CameraHandle:
        handle = CameraHandle(self, camera_id)
        with self:
            self.allocations[camera_id] = self._next_process_idx(), handle
        self.submit_request(
            CreateCamera(camera_id, protocol, pause_timeout=pause_timeout)
        )
        return handle

    def stop(self):
        super().stop()
        with self:
            driver_handles = list(self._driver_handles)
            self._driver_handles.clear()
        for handle in driver_handles:
            try:
                handle.stop()
            except Exception:  # noqa: BLE001
                pass
        self.pool_size = 0
