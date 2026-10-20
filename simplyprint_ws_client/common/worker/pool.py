"""The generic worker pool: run a producer in a chosen world, deliver home.

A *producer* is a callable ``producer(emit, is_stopped, *args)`` that loops,
calling ``emit(payload, timestamp)`` for each item it makes and returning when
``is_stopped()`` goes true. It may be synchronous or a coroutine function.

:meth:`WorkerPool.allocate` runs one producer in a chosen :class:`ExecutionContext`
and routes every item to an ``on_item(payload, timestamp)`` sink that ALWAYS fires
on the consumer loop:

* **INLINE** -- an async producer as a task on the consumer loop; ``emit`` calls
  the sink directly. No thread, no process, no copy.
* **THREAD** -- the producer runs in its own thread (its own loop, if async);
  items ride the courier back onto the consumer loop.
* **PROCESS** -- the producer runs inside one of the pool's worker processes:
  a sync producer on its own thread there, an async one as a task on that
  process's event loop. The pool keeps at most ``max_processes`` workers (one
  per core by default) and starts them on demand: a worker takes up to
  ``producers_per_process`` producers before the next one starts, so memory
  grows with load rather than with the core count; past the process cap each
  producer joins the least-loaded worker. A worker that has been empty for
  ``idle_timeout`` shuts down. There is no limit on the number of producers.
  Payloads ride
  the worker's zero-copy :class:`SharedSlabChannel`, tagged with the producer's
  id, into a bounded per-producer courier on the consumer loop.

A worker that dies takes its producers with it: each one gets a ``None`` item
(a failed read) and then completes, so its owner can start it again. Log
records made in a worker are forwarded to the parent's loggers.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import pickle
import queue
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.common.worker.context import ExecutionContext
from simplyprint_ws_client.common.worker.channel import (
    KIND_DONE,
    KIND_LOG,
    SlabLease,
    SharedSlabChannel,
)

__all__ = ["WorkerPool", "WorkerHandle"]

_logger = logging.getLogger("worker.pool")

#: ``producer(emit, is_stopped, *args)`` where ``emit(payload, timestamp)``.
Producer = Callable[..., Any]
OnItem = Callable[[Any, float], None]

_DEFAULT_MAXSIZE = 8

#: How long stop() waits for a backend thread/process before giving up.
JOIN_TIMEOUT = 2.0

#: How long an empty worker process is kept for the next producer.
DEFAULT_IDLE_TIMEOUT = 30.0

#: Producers a worker takes before another worker starts. Each spawned worker
#: is a Python interpreter (~60 MB), so filling a few before starting the next
#: keeps a many-camera client from starting one per core up front.
DEFAULT_PRODUCERS_PER_PROCESS = 4

# Ordered through a producer's courier after its final item, so completion is
# a delivery event and never overtakes a frame that is still queued.
_PROCESS_DONE = object()

# Commands sent to a worker process.
_CMD_START = "start"
_CMD_STOP = "stop"
_CMD_SHUTDOWN = "shutdown"


def _join_or_warn(target, name: str, logger: logging.Logger) -> None:
    """Join a thread/process; warn instead of silently leaking when it hangs
    (daemon backends would otherwise hide a producer that ignores its stop)."""
    target.join(timeout=JOIN_TIMEOUT)
    if target.is_alive():
        logger.warning("%s did not stop within %.1fs", name, JOIN_TIMEOUT)


def _call_on_loop(provider: EventLoopProvider, callback: Callable[[], None]) -> None:
    """Run ``callback`` on the consumer loop, or right here if there is none."""
    try:
        loop = provider.event_loop
        if loop.is_running() and not loop.is_closed():
            loop.call_soon_threadsafe(callback)
            return
    except RuntimeError:
        pass
    callback()


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


# --------------------------------------------------------------------------- #
# Worker process side
# --------------------------------------------------------------------------- #


def _portable_record(record: logging.LogRecord) -> bytes:
    """Pickle a log record the way ``QueueHandler.prepare`` flattens one:
    message formatted, args and traceback objects dropped."""
    state = dict(record.__dict__)
    state["msg"] = record.getMessage()
    state["args"] = None
    if record.exc_info and not record.exc_text:
        state["exc_text"] = logging.Formatter().formatException(record.exc_info)
    state["exc_info"] = None
    try:
        return pickle.dumps(state)
    except Exception:  # noqa: BLE001 -- an unpicklable ``extra`` value
        keep = ("name", "levelno", "levelname", "msg", "created", "exc_text")
        return pickle.dumps({key: state.get(key) for key in keep})


class _ChannelLogHandler(logging.Handler):
    """Forwards a worker's log records to the parent over its channel."""

    def __init__(self, channel: SharedSlabChannel) -> None:
        super().__init__()
        self._channel = channel

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._channel.send_log(_portable_record(record), record.created)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def _install_log_forwarding(channel: SharedSlabChannel, level: int) -> None:
    # A forked worker inherits the parent's root handlers, whose queue listener
    # thread does not exist here (records would pile up forever); a spawned one
    # has none (records would be lost). Either way, forward to the parent.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_ChannelLogHandler(channel))
    root.setLevel(level)


class _WorkerProducer:
    __slots__ = ("stop", "task", "thread")

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.task: Optional[asyncio.Task] = None


class _WorkerRuntime:
    """Runs the producers placed on one worker process.

    Sync producers get a thread each; async producers share one event loop
    thread, started with the first of them.
    """

    def __init__(self, channel: SharedSlabChannel) -> None:
        self._channel = channel
        self._logger = logging.getLogger("worker.process")
        self._lock = threading.Lock()
        self._producers: Dict[int, _WorkerProducer] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def start(self, producer_id: int, blob: bytes) -> None:
        try:
            producer, args, is_async = pickle.loads(blob)
        except Exception:  # noqa: BLE001 -- report it as a failed producer
            self._logger.exception("could not load producer %d", producer_id)
            self._channel.send(producer_id, None, time.time())
            self._channel.send_done(producer_id, time.time())
            return

        entry = _WorkerProducer()
        with self._lock:
            self._producers[producer_id] = entry

        def emit(payload: Any, timestamp: float) -> None:
            self._channel.send(producer_id, payload, timestamp)

        if is_async:
            self._ensure_loop().call_soon_threadsafe(
                self._spawn_task, producer_id, entry, producer, args, emit
            )
            return

        entry.thread = threading.Thread(
            target=self._run_sync,
            args=(producer_id, entry, producer, args, emit),
            name=f"sp-worker-{producer_id}",
            daemon=True,
        )
        entry.thread.start()

    def stop(self, producer_id: int) -> None:
        with self._lock:
            entry = self._producers.get(producer_id)
        if entry is None:
            return
        entry.stop.set()
        if self._loop is not None and entry.thread is None:
            self._loop.call_soon_threadsafe(self._cancel_task, entry)

    def shutdown(self) -> None:
        with self._lock:
            entries = list(self._producers.values())
        for entry in entries:
            entry.stop.set()
            if self._loop is not None and entry.thread is None:
                self._loop.call_soon_threadsafe(self._cancel_task, entry)
        deadline = time.monotonic() + JOIN_TIMEOUT
        for entry in entries:
            if entry.thread is not None:
                entry.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, name="sp-worker-loop", daemon=True
            )
            self._loop_thread.start()
        return self._loop

    def _run_sync(self, producer_id, entry, producer, args, emit) -> None:
        try:
            producer(emit, entry.stop.is_set, *args)
        except Exception:  # noqa: BLE001
            self._logger.exception("process producer %d failed", producer_id)
        finally:
            self._finish(producer_id)

    def _spawn_task(self, producer_id, entry, producer, args, emit) -> None:
        async def run() -> None:
            try:
                await producer(emit, entry.stop.is_set, *args)
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                self._logger.exception("process producer %d failed", producer_id)

        entry.task = asyncio.get_running_loop().create_task(run())
        # A done-callback, not ``finally``: a task cancelled before its first
        # step never enters the coroutine body.
        entry.task.add_done_callback(lambda _task: self._finish(producer_id))
        if entry.stop.is_set():
            entry.task.cancel()

    @staticmethod
    def _cancel_task(entry: _WorkerProducer) -> None:
        if entry.task is not None:
            entry.task.cancel()

    def _finish(self, producer_id: int) -> None:
        with self._lock:
            if self._producers.pop(producer_id, None) is None:
                return
        self._channel.send_done(producer_id, time.time())


def _worker_main(commands, child_args: tuple, log_level: int) -> None:
    """Worker process entrypoint: run producers until told to shut down."""
    # Ctrl-C reaches the whole process group; the parent owns shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    channel = SharedSlabChannel.attach(child_args)
    _install_log_forwarding(channel, log_level)
    runtime = _WorkerRuntime(channel)
    try:
        while True:
            try:
                command = commands.recv()
            except (EOFError, OSError):
                break  # the parent is gone
            if command[0] == _CMD_START:
                runtime.start(command[1], command[2])
            elif command[0] == _CMD_STOP:
                runtime.stop(command[1])
            else:
                break
    finally:
        runtime.shutdown()
        channel.close()


# --------------------------------------------------------------------------- #
# Parent side of the PROCESS context
# --------------------------------------------------------------------------- #


def _reemit_log(payload: Optional[bytes]) -> None:
    if payload is None:
        return
    try:
        record = logging.makeLogRecord(pickle.loads(payload))
    except Exception:  # noqa: BLE001
        _logger.debug("dropped an undecodable worker log record", exc_info=True)
        return
    logger = logging.getLogger(record.name)
    if logger.isEnabledFor(record.levelno):
        logger.handle(record)


def _recycle(item: object) -> None:
    if isinstance(item, SlabLease):
        item.release()  # a superseded/dropped frame still owns its slab


class _ProcessBackend(_Backend):
    """One producer placed on a worker process (the parent's view of it)."""

    def __init__(
        self,
        pool: WorkerPool,
        producer: Producer,
        args: Tuple,
        on_item: OnItem,
        *,
        is_async: bool,
        provider: EventLoopProvider,
        policy: OverflowPolicy,
        maxsize: int,
        on_done: Callable[[], None],
    ) -> None:
        self.id = -1  # assigned by the pool before start()
        # Pickled here so an unpicklable producer fails the allocate call.
        self._blob = pickle.dumps((producer, args, is_async))
        self._pool = pool
        self._provider = provider
        self._on_item = on_item
        self._on_done = on_done
        self._group: Optional[_WorkerProcess] = None
        self._courier: Courier = Courier(
            sink=self._deliver,
            provider=provider,
            policy=policy,
            maxsize=maxsize,
            lossless=lambda item: item is _PROCESS_DONE,
            on_drop=_recycle,
        )

    def start(self) -> None:
        self._group = self._pool._place(self)
        self._group.send((_CMD_START, self.id, self._blob))

    def receive(self, lease: SlabLease) -> None:
        """Called by the worker's reader, on the loop or the reader thread."""
        if lease.kind == KIND_DONE:
            self._courier.post(_PROCESS_DONE)
        elif not self._courier.post(lease):
            lease.release()  # closed courier: nobody else will

    def lost(self) -> None:
        """The worker process died under this producer."""
        self._courier.close(drain=False)  # recycles queued leases right away

        def notify() -> None:
            try:
                self._on_item(None, time.time())
            finally:
                self._on_done()

        _call_on_loop(self._provider, notify)

    def _deliver(self, item: object) -> None:
        if item is _PROCESS_DONE:
            self._on_done()
            return
        if not isinstance(item, SlabLease):  # pragma: no cover -- internal invariant
            raise TypeError(f"unexpected process courier item: {type(item)!r}")
        try:
            self._on_item(item.to_bytes(), item.timestamp)
        finally:
            item.release()

    def request_stop(self) -> None:
        group = self._group
        if group is not None:
            group.detach(self.id)
            group.send((_CMD_STOP, self.id))
        # Nothing is delivered after a stop; queued frames go back to their slabs.
        self._courier.close(drain=False)

    def finalize(self) -> None:
        # The producer's thread lives in a shared worker process; there is
        # nothing to join per producer.
        pass


class _WorkerProcess(_Backend):
    """One worker process and the producers placed on it (the parent side).

    Bookkeeping (``producers``, ``retiring``, ``exited``) is guarded by the
    pool's process lock.
    """

    def __init__(
        self,
        pool: WorkerPool,
        provider: EventLoopProvider,
        *,
        n_slabs: int,
        slab_size: int,
        log_level: int,
    ) -> None:
        self._pool = pool
        self._provider = provider
        self._loop = provider.event_loop
        self._channel = SharedSlabChannel.create(n_slabs=n_slabs, slab_size=slab_size)
        self._commands_in, self._commands_out = mp.Pipe(duplex=False)
        self._send_lock = threading.Lock()
        self._proc = mp.Process(
            target=_worker_main,
            args=(self._commands_in, self._channel.child_args(), log_level),
            name="sp-worker",
            daemon=True,
        )
        self.producers: Dict[int, _ProcessBackend] = {}
        self.retiring = False
        self.exited = False
        self._closing = threading.Event()
        self._reader = threading.Thread(
            target=self._read_loop, name="sp-worker-reader", daemon=True
        )
        self._reader_started = False
        self._loop_readers = False
        self._idle_timer: Optional[asyncio.TimerHandle] = None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    @property
    def load(self) -> int:
        return len(self.producers)

    def start(self) -> None:
        self._proc.start()
        self._commands_in.close()  # the worker's end; EOF reaches it when we go
        if not self._install_loop_readers():
            self._reader_started = True
            self._reader.start()

    def send(self, command: tuple) -> bool:
        with self._send_lock:
            if self.exited or self._commands_out.closed:
                return False
            try:
                self._commands_out.send(command)
            except (OSError, EOFError, ValueError):
                return False
        return True

    def detach(self, producer_id: int) -> None:
        with self._pool._process_lock:
            if self.producers.pop(producer_id, None) is None:
                return
            idle = not self.producers and not self.retiring and not self.exited
        if idle:
            _call_on_loop(self._provider, self._arm_idle_timer)

    # -- reading ------------------------------------------------------------ #

    def _dispatch(self, lease: SlabLease) -> None:
        if lease.kind == KIND_LOG:
            _reemit_log(lease.data)
            return
        with self._pool._process_lock:
            backend = self.producers.get(lease.producer_id)
        if backend is None:
            lease.release()  # a stopped producer's late frame
            return
        backend.receive(lease)

    def _read_loop(self) -> None:
        while not self._closing.is_set():
            lease = self._channel.recv(timeout=0.5)
            if lease is not None:
                self._dispatch(lease)
            elif not self._proc.is_alive():
                self._process_exited()
                return

    def _queue_one_readable(self) -> None:
        lease = self._channel.recv(timeout=0)
        if lease is not None:
            self._dispatch(lease)

    def _process_exited(self) -> None:
        # Exit is ordered after the worker's pipe writes: drain first so its
        # last frames and completions arrive before the loss is reported.
        while True:
            lease = self._channel.recv(timeout=0)
            if lease is None:
                break
            self._dispatch(lease)
        self._remove_loop_readers()
        with self._pool._process_lock:
            self.exited = True
            lost = list(self.producers.values())
            self.producers.clear()
            retiring = self.retiring
            self._pool._forget(self)
        if not retiring:
            _logger.warning(
                "worker process %s exited unexpectedly (exit code %s); "
                "%d producer(s) lost",
                self.pid,
                self._proc.exitcode,
                len(lost),
            )
        for backend in lost:
            backend.lost()
        if not retiring:
            self._pool._reap(self)

    def _install_loop_readers(self) -> bool:
        """Use native readiness notifications when the target loop supports it.

        ``add_reader`` avoids an extra thread hop and is the reliable Unix path.
        Proactor and non-local loops fall back to the reader thread.
        """
        try:
            if asyncio.get_running_loop() is not self._loop:
                return False
        except RuntimeError:
            return False

        channel_fd = self._channel.reader_fileno()
        process_fd = self._proc.sentinel
        try:
            self._loop.add_reader(channel_fd, self._queue_one_readable)
        except (AttributeError, NotImplementedError, OSError, ValueError):
            return False
        try:
            self._loop.add_reader(process_fd, self._process_exited)
        except (AttributeError, NotImplementedError, OSError, ValueError):
            self._loop.remove_reader(channel_fd)
            return False
        self._loop_readers = True
        return True

    def _remove_loop_readers(self) -> None:
        if not self._loop_readers:
            return
        self._loop_readers = False
        for descriptor in (self._channel.reader_fileno(), self._proc.sentinel):
            try:
                self._loop.remove_reader(descriptor)
            except (
                AttributeError,
                NotImplementedError,
                OSError,
                RuntimeError,
                ValueError,
            ):
                pass

    # -- idle retirement ---------------------------------------------------- #

    def _arm_idle_timer(self) -> None:
        timeout = self._pool._idle_timeout
        if timeout is None or self.retiring or self.exited:
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        try:
            self._idle_timer = asyncio.get_running_loop().call_later(
                timeout, self._idle_expired
            )
        except RuntimeError:
            self._idle_timer = None  # no loop: WorkerPool.stop cleans up

    def _idle_expired(self) -> None:
        self._idle_timer = None
        with self._pool._process_lock:
            if self.producers or self.retiring or self.exited:
                return
            self._pool._forget(self)
        self.request_stop()
        self._pool._reap(self)

    # -- lifecycle ---------------------------------------------------------- #

    def request_stop(self) -> None:
        """Ask the worker to shut down. Call on the consumer loop (it removes
        the loop readers); :meth:`finalize` then joins it on the reaper."""
        with self._pool._process_lock:
            self.retiring = True
        if self._idle_timer is not None:
            try:
                self._idle_timer.cancel()
            except RuntimeError:
                pass
            self._idle_timer = None
        self._remove_loop_readers()
        self._shutdown()

    def _shutdown(self) -> None:
        self._closing.set()
        self.send((_CMD_SHUTDOWN,))

    def finalize(self) -> None:
        self._shutdown()  # a crashed worker is reaped without request_stop
        self._proc.join(timeout=JOIN_TIMEOUT)
        if self._proc.is_alive():
            self._proc.terminate()
            _join_or_warn(self._proc, "worker process", _logger)
        if self._reader_started:
            _join_or_warn(self._reader, "worker reader thread", _logger)
        with self._send_lock:
            self.exited = True
            self._commands_out.close()
        with self._pool._process_lock:
            lost = list(self.producers.values())
            self.producers.clear()
        for backend in lost:  # stopped with the pool while still running
            backend.lost()
        self._channel.close()


class WorkerHandle:
    """The parent-side handle for one allocated producer."""

    def __init__(
        self,
        worker_id: int,
        backend: _Backend,
        release: Callable[[int], None],
        reap: Callable[[_Backend], None],
    ) -> None:
        self.id = worker_id
        self._backend = backend
        self._release = release
        self._reap = reap
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
            self._reap(self._backend)
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
        max_processes: Optional[int] = None,
        producers_per_process: int = DEFAULT_PRODUCERS_PER_PROCESS,
        idle_timeout: Optional[float] = DEFAULT_IDLE_TIMEOUT,
        log_level: Optional[int] = None,
    ) -> None:
        """``max_processes`` caps the PROCESS workers (default: one per core);
        it never limits how many producers run. ``producers_per_process`` is how
        many producers a worker takes before another starts. ``idle_timeout``
        is how long an empty worker lingers (``None`` keeps it until
        :meth:`stop`). ``log_level`` is the workers' root log level (default:
        the parent's)."""
        if max_processes is not None and max_processes < 1:
            raise ValueError("max_processes must be at least 1")
        if producers_per_process < 1:
            raise ValueError("producers_per_process must be at least 1")
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._n_slabs = n_slabs
        self._slab_size = slab_size
        self._overflow = overflow
        self._maxsize = maxsize
        self._max_processes = max_processes or os.cpu_count() or 1
        self._producers_per_process = producers_per_process
        self._idle_timeout = idle_timeout
        self._log_level = log_level
        self._handles: Dict[int, WorkerHandle] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        #: Guards the worker processes and the producers placed on them.
        self._process_lock = threading.Lock()
        self._processes: List[_WorkerProcess] = []
        # The single owner of every blocking thread/process join. Lazily started
        # on the first stop so a pool that never stops a worker spawns no thread.
        self._reap_queue: "queue.Queue[Optional[_Backend]]" = queue.Queue()
        self._reaper: Optional[threading.Thread] = None
        self._reaper_lock = threading.Lock()
        self._stopped = False
        self._reaper_done = False

    @property
    def max_processes(self) -> int:
        return self._max_processes

    def process_loads(self) -> List[int]:
        """How many producers each live worker process is running."""
        with self._process_lock:
            return [process.load for process in self._processes]

    def allocate(
        self,
        context: ExecutionContext,
        producer: Producer,
        on_item: OnItem,
        *,
        args: Tuple = (),
        is_async: Optional[bool] = None,
        overflow: Optional[OverflowPolicy] = None,
        maxsize: Optional[int] = None,
    ) -> WorkerHandle:
        """Run ``producer`` in ``context``, routing each item to ``on_item``.

        ``on_item(payload, timestamp)`` always fires on the consumer loop.
        ``is_async`` is inferred from ``producer`` when omitted. ``overflow`` and
        ``maxsize`` bound the items queued for the loop (THREAD and PROCESS).
        """
        if self._stopped:
            raise RuntimeError("worker pool is stopped")
        policy = overflow or self._overflow
        bound = maxsize or self._maxsize
        if is_async is None:
            is_async = asyncio.iscoroutinefunction(producer)

        handle_ref: List[WorkerHandle] = []

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
                maxsize=bound,
                on_done=on_done,
            )
        elif context is ExecutionContext.PROCESS:
            backend = _ProcessBackend(
                self,
                producer,
                args,
                on_item,
                is_async=is_async,
                provider=self._provider,
                policy=policy,
                maxsize=bound,
                on_done=on_done,
            )
        else:  # pragma: no cover -- exhaustive
            raise ValueError(f"unknown execution context {context!r}")

        with self._lock:
            worker_id = self._next_id
            self._next_id += 1
            handle = WorkerHandle(worker_id, backend, self._release, self._reap)
            handle_ref.append(handle)
            self._handles[worker_id] = handle

        if isinstance(backend, _ProcessBackend):
            backend.id = worker_id
        try:
            backend.start()
        except BaseException:
            self._release(worker_id)
            raise
        return handle

    def _place(self, backend: _ProcessBackend) -> _WorkerProcess:
        """Put ``backend`` on the least-loaded worker, starting a new one when
        even that one is full and the pool is below ``max_processes``."""
        with self._process_lock:
            if self._stopped:
                raise RuntimeError("worker pool is stopped")
            process = min(self._processes, key=lambda p: p.load, default=None)
            if process is None or (
                process.load >= self._producers_per_process
                and len(self._processes) < self._max_processes
            ):
                process = _WorkerProcess(
                    self,
                    self._provider,
                    n_slabs=self._n_slabs,
                    slab_size=self._slab_size,
                    log_level=self._log_level
                    if self._log_level is not None
                    else logging.getLogger().getEffectiveLevel(),
                )
                process.start()
                self._processes.append(process)
            process.producers[backend.id] = backend
            return process

    def _forget(self, process: _WorkerProcess) -> None:
        # Caller holds ``self._process_lock``.
        try:
            self._processes.remove(process)
        except ValueError:
            pass

    def _release(self, worker_id: int) -> None:
        with self._lock:
            self._handles.pop(worker_id, None)

    def _reap(self, backend: _Backend) -> None:
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
            self._finalize_one(backend)
        else:
            self._reap_queue.put(backend)

    def _ensure_reaper_locked(self) -> None:
        # Caller holds ``self._reaper_lock``.
        if self._reaper is None:
            self._reaper = threading.Thread(
                target=self._reaper_loop, name="sp-worker-reaper", daemon=True
            )
            self._reaper.start()

    def _reaper_loop(self) -> None:
        while True:
            backend = self._reap_queue.get()
            if backend is None:  # sentinel: the pool is stopping
                return
            self._finalize_one(backend)

    @staticmethod
    def _finalize_one(backend: _Backend) -> None:
        try:
            backend.finalize()
        except Exception:  # noqa: BLE001 -- one bad finalize must not wedge the reaper
            _logger.exception("worker finalize failed")

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
        with self._process_lock:
            processes = list(self._processes)
            self._processes.clear()
        for process in processes:
            process.request_stop()
            self._reap(process)
        with self._reaper_lock:
            reaper = self._reaper
        if reaper is not None:
            self._reap_queue.put(None)  # sentinel
            reaper.join()
            with self._reaper_lock:
                self._reaper_done = True
