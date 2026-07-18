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
