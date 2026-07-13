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
import queue
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

    def request_stop(self) -> None:
        """Signal the producer to stop. MUST be non-blocking -- it runs on the
        consumer loop. The blocking join happens later in :meth:`finalize`."""
        ...

    def finalize(self) -> None:
        """Join the backend's thread/process and release its resources. Blocks;
        only ever called from the pool's reaper thread (or, at full teardown,
        from :meth:`WorkerPool.stop`)."""
        ...


class _InlineBackend(_Backend):
    """An async producer run as a task directly on the consumer loop."""

    def __init__(
        self,
        provider: EventLoopProvider,
        producer: Producer,
        on_item: OnItem,
        args: Tuple,
        on_done: Callable[[], None],
    ) -> None:
        self._provider = provider
        self._producer = producer
        self._on_item = on_item
        self._args = args
        self._on_done = on_done
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
        finally:
            self._on_done()

    def request_stop(self) -> None:
        def _kill() -> None:
            if self._stop is not None:
                self._stop.set()
            if self._task is not None and not self._task.done():
                self._task.cancel()

        try:
            self._provider.event_loop.call_soon_threadsafe(_kill)
        except RuntimeError:
            pass

    def finalize(self) -> None:
        # The producer is a task on the consumer loop, cancelled by request_stop;
        # there is no thread/process to join.
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
        on_done: Callable[[], None],
    ) -> None:
        self._producer = producer
        self._args = args
        self._is_async = is_async
        self._stop = threading.Event()
        self._on_done = on_done
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
        finally:
            self._on_done()

    def request_stop(self) -> None:
        self._stop.set()

    def finalize(self) -> None:
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
        on_done: Callable[[], None],
    ) -> None:
        self._on_item = on_item
        self._channel = SharedSlabChannel.create(n_slabs=n_slabs, slab_size=slab_size)
        self._stop = mp.Event()
        self._on_done = on_done
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
        try:
            while not self._stop.is_set():
                lease = self._channel.recv(timeout=0.5)
                if lease is not None:
                    self._courier.post(lease)
                    continue
                # A one-shot producer exits after its first frame. Retire it
                # immediately instead of leaving its reader thread and process
                # handle alive until the next camera demand.
                if not self._proc.is_alive():
                    return
        finally:
            self._on_done()

    def start(self) -> None:
        self._proc.start()
        self._reader.start()

    def request_stop(self) -> None:
        self._stop.set()

    def finalize(self) -> None:
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
        self,
        worker_id: int,
        backend: _Backend,
        release: Callable[[int], None],
        reap: Callable[[_Backend, Optional[Callable[[], None]]], None],
        release_capacity: Optional[Callable[[], None]] = None,
    ) -> None:
        self.id = worker_id
        self._backend = backend
        self._release = release
        self._reap = reap
        self._release_capacity = release_capacity
        self._stopped = False
        self._lock = threading.Lock()

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def _backend_done(self) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop the worker WITHOUT blocking: signal the producer, hand the
        backend to the pool's reaper for its join, and release the slot. Safe to
        call on the consumer loop -- the up-to-``JOIN_TIMEOUT`` join never runs
        here."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        try:
            self._backend.request_stop()
            self._reap(self._backend, self._release_capacity)
        finally:
            self._release(self.id)


class WorkerPool:
    """Allocate producers across execution contexts; deliver their items home."""

    def __init__(
        self,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        n_slabs: int = 24,
        slab_size: int = 768 * 1024,
        overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        maxsize: int = _DEFAULT_MAXSIZE,
        max_process_workers: Optional[int] = None,
    ) -> None:
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._n_slabs = n_slabs
        self._slab_size = slab_size
        self._overflow = overflow
        self._maxsize = maxsize
        if max_process_workers is not None and max_process_workers < 1:
            raise ValueError("max_process_workers must be at least 1")
        self._process_slots = (
            asyncio.Semaphore(max_process_workers)
            if max_process_workers is not None
            else None
        )
        self._handles: Dict[int, WorkerHandle] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        # The single owner of every blocking thread/process join. Lazily started
        # on the first stop so a pool that never stops a worker spawns no thread.
        self._reap_queue: "queue.Queue[Optional[tuple[_Backend, Optional[Callable[[], None]]]]]" = queue.Queue()
        self._reaper: Optional[threading.Thread] = None
        self._reaper_lock = threading.Lock()
        self._stopped = False
        self._reaper_done = False

    def allocate(
        self,
        context: ExecutionContext,
        producer: Producer,
        on_item: OnItem,
        *,
        args: Tuple = (),
        is_async: Optional[bool] = None,
        overflow: Optional[OverflowPolicy] = None,
        _release_capacity: Optional[Callable[[], None]] = None,
    ) -> WorkerHandle:
        """Run ``producer`` in ``context``, routing each item to ``on_item``.

        ``on_item(payload, timestamp)`` always fires on the consumer loop.
        ``is_async`` is inferred from ``producer`` when omitted.
        """
        if self._stopped:
            raise RuntimeError("worker pool is stopped")
        if (
            context is ExecutionContext.PROCESS
            and self._process_slots is not None
            and _release_capacity is None
        ):
            raise RuntimeError(
                "bounded PROCESS workers require await allocate_async(...)"
            )
        policy = overflow or self._overflow
        if is_async is None:
            is_async = asyncio.iscoroutinefunction(producer)

        handle_ref: list[WorkerHandle] = []

        def on_done() -> None:
            if handle_ref:
                handle_ref[0]._backend_done()

        if context is ExecutionContext.INLINE:
            if not is_async:
                raise ValueError("INLINE requires an async producer")
            backend: _Backend = _InlineBackend(
                self._provider, producer, on_item, args, on_done
            )
        elif context is ExecutionContext.THREAD:
            backend = _ThreadBackend(
                self._provider,
                producer,
                on_item,
                args,
                is_async=is_async,
                policy=policy,
                maxsize=self._maxsize,
                on_done=on_done,
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
                on_done=on_done,
            )
        else:  # pragma: no cover -- exhaustive
            raise ValueError(f"unknown execution context {context!r}")

        with self._lock:
            worker_id = self._next_id
            self._next_id += 1
            handle = WorkerHandle(
                worker_id,
                backend,
                self._release,
                self._reap,
                _release_capacity,
            )
            handle_ref.append(handle)
            self._handles[worker_id] = handle

        backend.start()
        return handle

    async def allocate_async(
        self,
        context: ExecutionContext,
        producer: Producer,
        on_item: OnItem,
        *,
        args: Tuple = (),
        is_async: Optional[bool] = None,
        overflow: Optional[OverflowPolicy] = None,
    ) -> WorkerHandle:
        """Allocate with async admission to the bounded PROCESS lane."""
        slots = self._process_slots if context is ExecutionContext.PROCESS else None
        if slots is None:
            return self.allocate(
                context,
                producer,
                on_item,
                args=args,
                is_async=is_async,
                overflow=overflow,
            )

        await slots.acquire()
        if self._stopped:
            slots.release()
            raise RuntimeError("worker pool is stopped")
        released = False

        def release_capacity() -> None:
            nonlocal released
            if released:
                return
            released = True
            try:
                self._provider.event_loop.call_soon_threadsafe(slots.release)
            except RuntimeError:
                pass

        try:
            return self.allocate(
                context,
                producer,
                on_item,
                args=args,
                is_async=is_async,
                overflow=overflow,
                _release_capacity=release_capacity,
            )
        except BaseException:
            release_capacity()
            raise

    def _release(self, worker_id: int) -> None:
        with self._lock:
            self._handles.pop(worker_id, None)

    def _reap(
        self,
        backend: _Backend,
        release_capacity: Optional[Callable[[], None]] = None,
    ) -> None:
        """Hand a stopped backend to the reaper for its blocking join. Never
        blocks the caller. After the pool is fully torn down (reaper joined), a
        late stop finalizes inline so nothing leaks."""
        with self._reaper_lock:
            if self._reaper_done:
                finalize_inline = True
            else:
                self._ensure_reaper_locked()
                finalize_inline = False
        if finalize_inline:
            self._finalize_one(backend, release_capacity)
        else:
            self._reap_queue.put((backend, release_capacity))

    def _ensure_reaper_locked(self) -> None:
        # Caller holds ``self._reaper_lock``.
        if self._reaper is None:
            self._reaper = threading.Thread(
                target=self._reaper_loop, name="sp-worker-reaper", daemon=True
            )
            self._reaper.start()

    def _reaper_loop(self) -> None:
        while True:
            item = self._reap_queue.get()
            if item is None:  # sentinel: the pool is stopping
                return
            self._finalize_one(*item)

    @staticmethod
    def _finalize_one(
        backend: _Backend,
        release_capacity: Optional[Callable[[], None]] = None,
    ) -> None:
        try:
            backend.finalize()
        except Exception:  # noqa: BLE001 -- one bad finalize must not wedge the reaper
            _logger.exception("worker finalize failed")
        finally:
            if release_capacity is not None:
                release_capacity()

    def stop(self) -> None:
        """Stop every worker and drain the reaper. Idempotent. This is the ONE
        sanctioned place a blocking join runs (each backend's join is internally
        bounded by ``JOIN_TIMEOUT`` + terminate), so teardown leaks no thread,
        process, or shared-memory segment."""
        with self._reaper_lock:
            if self._stopped:
                return
            self._stopped = True
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            handle.stop()  # request_stop + enqueue on the reaper + release
        with self._reaper_lock:
            reaper = self._reaper
        if reaper is not None:
            self._reap_queue.put(None)  # sentinel
            reaper.join()
            with self._reaper_lock:
                self._reaper_done = True
