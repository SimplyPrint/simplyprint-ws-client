"""Tests for :class:`Offload` -- the two bounded blocking-work lanes.

Pins the properties the runtime model relies on: the result/exception of a
blocking call comes back on the loop; ``io`` and ``transfer`` are *separate*
named pools (never the shared default executor, so a stack sampled mid-stall
names the lane); shutdown joins every lane thread so teardown leaks nothing.
"""

import asyncio
import threading

import pytest

from simplyprint_ws_client.common.asyncio.offload import Offload


def _raise() -> None:
    raise ValueError("boom")


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
