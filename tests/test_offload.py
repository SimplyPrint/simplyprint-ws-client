"""Tests for :class:`Offload` -- the two bounded blocking-work lanes.

Pins the properties the runtime model relies on: the result/exception of a
blocking call comes back on the loop; ``io`` and ``transfer`` are *separate*
named pools (never the shared default executor, so a stack sampled mid-stall
names the lane); shutdown joins every lane thread so teardown leaks nothing.
"""

import asyncio
import contextvars
import threading

import pytest

from simplyprint_ws_client.common.asyncio.offload import (
    Offload,
    install_default_executor,
)


def _raise() -> None:
    raise ValueError("boom")


def _raise_timeout() -> None:
    raise asyncio.TimeoutError("work timed out")


@pytest.mark.asyncio
async def test_run_io_and_run_transfer_return_values():
    off = Offload()
    try:
        assert await off.run_io(lambda x: x + 1, 1) == 2
        assert await off.run_transfer(lambda a, b: a * b, 3, 4) == 12
    finally:
        off.shutdown()


@pytest.mark.asyncio
async def test_lanes_are_separate_named_pools():
    off = Offload()
    try:
        io_name = await off.run_io(lambda: threading.current_thread().name)
        tr_name = await off.run_transfer(lambda: threading.current_thread().name)
        assert io_name.startswith("sp-io")
        assert tr_name.startswith("sp-transfer")
        assert io_name != tr_name
    finally:
        off.shutdown()


@pytest.mark.asyncio
async def test_exceptions_propagate_to_the_loop():
    off = Offload()
    try:
        with pytest.raises(ValueError, match="boom"):
            await off.run_io(_raise)
        with pytest.raises(ValueError, match="boom"):
            await off.run_transfer(_raise)
        with pytest.raises(asyncio.TimeoutError, match="work timed out"):
            await off.run_io(_raise_timeout)
    finally:
        off.shutdown()


@pytest.mark.asyncio
async def test_lanes_preserve_the_calling_context():
    operation = contextvars.ContextVar("operation", default=None)
    token = operation.set("active")
    off = Offload()
    try:
        assert await off.run_io(operation.get) == "active"
        assert await off.run_transfer(operation.get) == "active"
    finally:
        off.shutdown()
        operation.reset(token)


@pytest.mark.asyncio
async def test_executor_completion_uses_loop_owned_wakeups(monkeypatch):
    loop = asyncio.get_running_loop()
    off = Offload()
    try:
        # Reproduce the supported-runtime failure: the completion handle reaches
        # ``loop._ready`` but its cross-thread self-pipe byte is lost. The
        # helper's loop-owned watchdog timer must still process the result.
        monkeypatch.setattr(loop, "_write_to_self", lambda: None)
        assert await off.run_io(lambda: 42) == 42
    finally:
        off.shutdown()


def test_shutdown_joins_lane_threads():
    async def go() -> Offload:
        off = Offload()
        await off.run_io(lambda: 1)
        await off.run_transfer(lambda: 1)
        return off

    off = asyncio.run(go())
    off.shutdown(wait=True)

    names = [t.name for t in threading.enumerate()]
    assert not any(n.startswith("sp-io") or n.startswith("sp-transfer") for n in names)


def test_double_shutdown_is_a_noop():
    off = Offload()
    off.shutdown()
    off.shutdown()  # must not raise


def test_owned_loop_default_executor_is_explicitly_bounded():
    loop = asyncio.new_event_loop()
    executor = install_default_executor(
        loop,
        workers=2,
        thread_name_prefix="sp-test-loop",
    )
    try:
        name = loop.run_until_complete(
            asyncio.to_thread(lambda: threading.current_thread().name)
        )
        assert executor._max_workers == 2
        assert name.startswith("sp-test-loop")
    finally:
        # Direct shutdown keeps this test compatible with sandboxes where
        # shutdown_default_executor's extra helper thread cannot signal home.
        executor.shutdown(wait=True)
        loop.close()

    assert not any(
        thread.name.startswith("sp-test-loop") for thread in threading.enumerate()
    )


# -- the resizable transfer lane ------------------------------------------------

import time  # noqa: E402

from simplyprint_ws_client.common.asyncio.offload import (  # noqa: E402
    DEFAULT_IO_WORKERS,
    DEFAULT_TRANSFER_WORKERS,
    ResizableThreadPool,
)


def _until(condition, timeout: float = 5.0) -> None:
    """Poll ``condition`` (pool threads move on their own schedule)."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.005)


def _alive(prefix: str) -> int:
    return sum(t.name.startswith(prefix) for t in threading.enumerate())


class _Blocked:
    """Work items that record their start and block until released."""

    def __init__(self) -> None:
        self.started: list = []
        self.gates: dict = {}
        self._lock = threading.Lock()

    def item(self, n: int):
        self.gates[n] = threading.Event()

        def run() -> int:
            with self._lock:
                self.started.append(n)
            assert self.gates[n].wait(5)
            return n

        return run


def test_growing_while_busy_starts_queued_work_at_once():
    pool = ResizableThreadPool(1, thread_name_prefix="sp-test-grow")
    work = _Blocked()
    try:
        futures = [pool.submit(work.item(n)) for n in range(3)]
        _until(lambda: work.started == [0])
        time.sleep(0.05)
        assert work.started == [0]  # width 1: the rest wait

        pool.set_workers(3)
        _until(lambda: sorted(work.started) == [0, 1, 2])
        assert pool.threads == 3

        for gate in work.gates.values():
            gate.set()
        assert [f.result(5) for f in futures] == [0, 1, 2]
    finally:
        pool.shutdown(wait=True)
    assert _alive("sp-test-grow") == 0


def test_shrinking_while_busy_lets_running_work_finish_then_gives_threads_back():
    pool = ResizableThreadPool(3, thread_name_prefix="sp-test-shrink")
    work = _Blocked()
    try:
        running = [pool.submit(work.item(n)) for n in range(3)]
        _until(lambda: sorted(work.started) == [0, 1, 2])

        pool.set_workers(1)
        assert pool.workers == 1
        queued = pool.submit(work.item(3))
        time.sleep(0.05)
        # Nothing running was stopped, and nothing new starts above the width.
        assert not any(f.done() for f in running)
        assert 3 not in work.started and pool.threads == 3

        work.gates[0].set()
        work.gates[1].set()
        assert running[0].result(5) == 0 and running[1].result(5) == 1
        _until(lambda: pool.threads == 1)  # the two surplus threads left
        assert 3 not in work.started  # one still runs: at the width

        work.gates[2].set()
        _until(lambda: 3 in work.started)
        work.gates[3].set()
        assert queued.result(5) == 3
        assert pool.threads == 1
    finally:
        pool.shutdown(wait=True)
    assert _alive("sp-test-shrink") == 0


def test_shrinking_an_idle_pool_stops_its_surplus_threads_at_once():
    pool = ResizableThreadPool(4, thread_name_prefix="sp-test-idle")
    try:
        gate = threading.Event()
        futures = [pool.submit(gate.wait, 5) for _ in range(4)]
        _until(lambda: pool.threads == 4)
        gate.set()
        for future in futures:
            future.result(5)

        pool.set_workers(1)
        _until(lambda: pool.threads == 1 and _alive("sp-test-idle") == 1)
    finally:
        pool.shutdown(wait=True)


def test_repeated_resizing_under_load_leaks_no_threads():
    pool = ResizableThreadPool(2, thread_name_prefix="sp-test-churn")
    try:
        futures = []
        for width in (8, 1, 16, 3, 1, 32, 2):
            pool.set_workers(width)
            futures += [pool.submit(time.sleep, 0.001) for _ in range(20)]
            assert pool.threads <= 32  # never above the widest width set
        for future in futures:
            future.result(10)
        # Back to the last width (2) once the surplus is idle; exited threads
        # are really gone, not parked.
        _until(lambda: pool.threads <= 2 and _alive("sp-test-churn") <= 2)
    finally:
        pool.shutdown(wait=True)
    assert _alive("sp-test-churn") == 0


def test_threads_start_on_demand_and_work_starts_in_order():
    pool = ResizableThreadPool(8, thread_name_prefix="sp-test-order")
    try:
        assert pool.threads == 0  # nothing started before work arrives
        assert pool.submit(lambda: 1).result(5) == 1
        assert pool.threads == 1

        pool.set_workers(1)
        order: list = []
        futures = [pool.submit(order.append, n) for n in range(10)]
        for future in futures:
            future.result(5)
        assert order == list(range(10))
    finally:
        pool.shutdown(wait=True)


def test_pool_bounds_and_errors():
    with pytest.raises(ValueError):
        ResizableThreadPool(0, thread_name_prefix="sp-test-bounds")
    pool = ResizableThreadPool(1, thread_name_prefix="sp-test-bounds")
    try:
        with pytest.raises(ValueError):
            pool.set_workers(0)
        assert pool.workers == 1
        with pytest.raises(ValueError, match="boom"):
            pool.submit(_raise).result(5)
    finally:
        pool.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        pool.submit(lambda: 1)
    pool.set_workers(4)  # after shutdown: a no-op, not an error
    assert _alive("sp-test-bounds") == 0


def test_shutdown_cancels_queued_work_and_lets_running_work_finish():
    pool = ResizableThreadPool(1, thread_name_prefix="sp-test-stop")
    work = _Blocked()
    running = pool.submit(work.item(0))
    queued = pool.submit(work.item(1))
    _until(lambda: work.started == [0])

    stopper = threading.Thread(
        target=pool.shutdown, kwargs={"wait": True, "cancel_futures": True}
    )
    stopper.start()
    _until(lambda: queued.cancelled())
    work.gates[0].set()
    stopper.join(5)
    assert not stopper.is_alive()
    assert running.result(5) == 0
    assert _alive("sp-test-stop") == 0


@pytest.mark.asyncio
async def test_offload_resizes_the_transfer_lane_only():
    off = Offload()
    try:
        assert off.transfer_workers == DEFAULT_TRANSFER_WORKERS
        release = threading.Event()
        started = threading.Semaphore(0)

        def hold() -> None:
            started.release()
            assert release.wait(5)

        off.set_transfer_workers(1)
        tasks = [asyncio.ensure_future(off.run_transfer(hold)) for _ in range(3)]
        await asyncio.to_thread(started.acquire, True, 5)
        assert not started.acquire(timeout=0.05)  # width 1: one runs

        off.set_transfer_workers(3)  # grown while busy
        for _ in range(2):
            assert await asyncio.to_thread(started.acquire, True, 5)

        # The short-call lane is untouched and still answers.
        assert off._io._max_workers == DEFAULT_IO_WORKERS
        assert await off.run_io(lambda: "io") == "io"

        off.set_transfer_workers(1)  # shrunk while busy: nothing is stopped
        release.set()
        await asyncio.gather(*tasks)
        assert off.transfer_workers == 1
        with pytest.raises(ValueError):
            off.set_transfer_workers(0)
    finally:
        off.shutdown(wait=True)
    assert _alive("sp-transfer") == 0
