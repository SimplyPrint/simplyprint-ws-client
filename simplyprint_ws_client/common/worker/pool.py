"""The generic worker pool: run a producer in a chosen world, deliver home.

A *producer* is a callable ``producer(emit, is_stopped, *args)`` that loops,
calling ``emit(payload, timestamp)`` for each item it makes and returning when
``is_stopped()`` goes true. It may be synchronous (the only option for PROCESS)
or a coroutine function (for INLINE / THREAD).

:meth:`WorkerPool.allocate` runs one producer in a chosen :class:`ExecutionContext`
and routes every item to an ``on_item(payload, timestamp)`` sink that ALWAYS fires
on the consumer loop:

* **INLINE** -- the producer is a task on the consumer loop; ``emit`` calls the
  sink directly. No thread, no process, no copy.
* **THREAD** -- the producer runs in its own thread (its own loop, if async);
  items ride the courier back onto the consumer loop.
* **PROCESS** -- the producer runs in a subprocess; payloads ride a zero-copy
  :class:`SharedSlabChannel`; a reader thread couriers each :class:`SlabLease`
  onto the loop, where the sink reads it and the slab is recycled.

This is the engine the camera pool is rebuilt on; cameras are just one producer.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.common.worker.context import ExecutionContext
from simplyprint_ws_client.common.worker.channel import SlabLease, SharedSlabChannel

__all__ = ["WorkerPool", "WorkerHandle"]

_logger = logging.getLogger("worker.pool")

#: ``producer(emit, is_stopped, *args)`` where ``emit(payload, timestamp)``.
Producer = Callable[..., Any]
OnItem = Callable[[Any, float], None]

_DEFAULT_MAXSIZE = 8

#: How long stop() waits for a backend thread/process before giving up.
JOIN_TIMEOUT = 2.0


def _join_or_warn(target, name: str, logger: logging.Logger) -> None:
    """Join a thread/process; warn instead of silently leaking when it hangs
    (daemon backends would otherwise hide a producer that ignores its stop)."""
    target.join(timeout=JOIN_TIMEOUT)
    if target.is_alive():
        logger.warning("%s did not stop within %.1fs", name, JOIN_TIMEOUT)


class _Backend:
    def start(self) -> None: ...
    def stop(self) -> None: ...


class _InlineBackend(_Backend):
    """An async producer run as a task directly on the consumer loop."""

    def __init__(
        self,
        provider: EventLoopProvider,
        producer: Producer,
        on_item: OnItem,
        args: Tuple,
    ) -> None:
        self._provider = provider
        self._producer = producer
        self._on_item = on_item
        self._args = args
        self._task: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None

    def start(self) -> None:
        self._provider.event_loop.call_soon_threadsafe(self._launch)

    def _launch(self) -> None:
        self._stop = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        def emit(payload: Any, timestamp: float) -> None:
            self._on_item(payload, timestamp)  # already on the loop

        try:
            await self._producer(emit, self._stop.is_set, *self._args)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            _logger.exception("inline producer failed")

    def stop(self) -> None:
        def _kill() -> None:
            if self._stop is not None:
                self._stop.set()
            if self._task is not None and not self._task.done():
                self._task.cancel()

        try:
            self._provider.event_loop.call_soon_threadsafe(_kill)
        except RuntimeError:
            pass


class _ThreadBackend(_Backend):
    """A producer (sync or async) run in its own thread; items couriered home."""

    def __init__(
        self,
        provider: EventLoopProvider,
        producer: Producer,
        on_item: OnItem,
        args: Tuple,
        *,
        is_async: bool,
        policy: OverflowPolicy,
        maxsize: int,
    ) -> None:
        self._producer = producer
        self._args = args
        self._is_async = is_async
        self._stop = threading.Event()
        self._courier: Courier = Courier(
            sink=lambda item: on_item(item[0], item[1]),
            provider=provider,
            policy=policy,
            maxsize=maxsize,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        def emit(payload: Any, timestamp: float) -> None:
            self._courier.post((payload, timestamp))

        try:
            if self._is_async:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(
                        self._producer(emit, self._stop.is_set, *self._args)
                    )
                finally:
                    loop.close()
            else:
                self._producer(emit, self._stop.is_set, *self._args)
        except Exception:  # noqa: BLE001
            _logger.exception("threaded producer failed")

    def stop(self) -> None:
        self._stop.set()
        _join_or_warn(self._thread, "worker thread", _logger)
        self._courier.close()


def _process_main(
    producer: Producer, args: Tuple, child_args: tuple, stop_event: Any
) -> None:
    """Subprocess entrypoint: drive the producer, push frames through the channel."""
    channel = SharedSlabChannel.attach(child_args)

    def emit(payload: Any, timestamp: float) -> None:
        channel.send(0, payload, timestamp)

    try:
        producer(emit, stop_event.is_set, *args)
    except Exception:  # noqa: BLE001
        logging.getLogger("worker.process").exception("process producer failed")
    finally:
        channel.close()


class _ProcessBackend(_Backend):
    """A sync producer run in a subprocess; payloads ride zero-copy shared memory."""

    def __init__(
        self,
        provider: EventLoopProvider,
        producer: Producer,
        args: Tuple,
        on_item: OnItem,
        *,
        policy: OverflowPolicy,
        maxsize: int,
        n_slabs: int,
        slab_size: int,
    ) -> None:
        self._on_item = on_item
        self._channel = SharedSlabChannel.create(n_slabs=n_slabs, slab_size=slab_size)
        self._stop = mp.Event()
        self._proc = mp.Process(
            target=_process_main,
            args=(producer, args, self._channel.child_args(), self._stop),
            daemon=True,
        )
        self._courier: Courier = Courier(
            sink=self._deliver,
            provider=provider,
            policy=policy,
            maxsize=maxsize,
            on_drop=self._recycle,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)

    def _deliver(self, lease: SlabLease) -> None:
        try:
            self._on_item(lease.to_bytes(), lease.timestamp)
        finally:
            lease.release()

    @staticmethod
    def _recycle(lease: SlabLease) -> None:
        lease.release()  # a superseded/dropped frame still owns its slab

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            lease = self._channel.recv(timeout=0.5)
            if lease is None:
                continue
            self._courier.post(lease)

    def start(self) -> None:
        self._proc.start()
        self._reader.start()

    def stop(self) -> None:
        self._stop.set()
        self._proc.join(timeout=JOIN_TIMEOUT)
        if self._proc.is_alive():
            self._proc.terminate()
            _join_or_warn(self._proc, "worker process", _logger)
        _join_or_warn(self._reader, "worker reader thread", _logger)
        # drain=False: recycle any queued leases SYNCHRONOUSLY here (via _recycle,
        # which releases their memoryviews and slabs) before the channel is torn
        # down. The default drain=True would instead schedule delivery onto the
        # loop -- which would read slabs after _channel.close() (use-after-free)
        # and leave memoryviews alive across it.
        self._courier.close(drain=False)
        self._channel.close()


class WorkerHandle:
    """The parent-side handle for one allocated producer."""

    def __init__(
        self, worker_id: int, backend: _Backend, release: Callable[[int], None]
    ) -> None:
        self.id = worker_id
        self._backend = backend
        self._release = release
        self._stopped = False

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._backend.stop()
        finally:
            self._release(self.id)


class WorkerPool:
    """Allocate producers across execution contexts; deliver their items home."""

    def __init__(
        self,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        n_slabs: int = 24,
        slab_size: int = 512 * 1024,
        overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        maxsize: int = _DEFAULT_MAXSIZE,
    ) -> None:
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._n_slabs = n_slabs
        self._slab_size = slab_size
        self._overflow = overflow
        self._maxsize = maxsize
        self._handles: Dict[int, WorkerHandle] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def allocate(
        self,
        context: ExecutionContext,
        producer: Producer,
        on_item: OnItem,
        *,
        args: Tuple = (),
        is_async: Optional[bool] = None,
        overflow: Optional[OverflowPolicy] = None,
    ) -> WorkerHandle:
        """Run ``producer`` in ``context``, routing each item to ``on_item``.

        ``on_item(payload, timestamp)`` always fires on the consumer loop.
        ``is_async`` is inferred from ``producer`` when omitted.
        """
        policy = overflow or self._overflow
        if is_async is None:
            is_async = asyncio.iscoroutinefunction(producer)

        if context is ExecutionContext.INLINE:
            if not is_async:
                raise ValueError("INLINE requires an async producer")
            backend: _Backend = _InlineBackend(self._provider, producer, on_item, args)
        elif context is ExecutionContext.THREAD:
            backend = _ThreadBackend(
                self._provider,
                producer,
                on_item,
                args,
                is_async=is_async,
                policy=policy,
                maxsize=self._maxsize,
            )
        elif context is ExecutionContext.PROCESS:
            if is_async:
                raise ValueError("PROCESS requires a synchronous producer")
            backend = _ProcessBackend(
                self._provider,
                producer,
                args,
                on_item,
                policy=policy,
                maxsize=self._maxsize,
                n_slabs=self._n_slabs,
                slab_size=self._slab_size,
            )
        else:  # pragma: no cover -- exhaustive
            raise ValueError(f"unknown execution context {context!r}")

        with self._lock:
            worker_id = self._next_id
            self._next_id += 1
            handle = WorkerHandle(worker_id, backend, self._release)
            self._handles[worker_id] = handle

        backend.start()
        return handle

    def _release(self, worker_id: int) -> None:
        with self._lock:
            self._handles.pop(worker_id, None)

    def stop(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            handle.stop()
