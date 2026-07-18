"""Tests for :class:`ConfigFlusher` -- coalesced, offloaded config persistence.

Pins: a storm of change events collapses to one or two atomic writes (not one per
event); a slow flush runs off the loop so the heartbeat keeps beating; ``aclose``
drains a pending write once; and the dead-loop fallback writes any pending change
synchronously. Uses the real ``JsonConfigManager`` (real json + fsync + replace).
"""

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from simplyprint_ws_client.common.asyncio.offload import Offload
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.config.flusher import ConfigFlusher
from simplyprint_ws_client.core.config.json import JsonConfigManager

from tests._loop_heartbeat import LoopHeartbeat


def _manager(tmp_path: Path) -> JsonConfigManager:
    manager = JsonConfigManager(name="flushbench", base_directory=str(tmp_path))
    config = PrinterConfig.get_new()
    config.id = 1
    config.in_setup = False
    manager.persist(config)
    return manager


def _count_replace(monkeypatch) -> dict:
    calls = {"n": 0}
    real = os.replace

    def counting(src, dst):
        calls["n"] += 1
        return real(src, dst)

    monkeypatch.setattr(os, "replace", counting)
    return calls


@pytest.mark.asyncio
async def test_trigger_storm_coalesces_to_one_or_two_writes(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    offload = Offload()
    replaces = _count_replace(monkeypatch)
    flusher = ConfigFlusher(manager, offload, loop=asyncio.get_running_loop())

    try:

        def hammer():
            for _ in range(50):
                flusher.trigger()

        thread = threading.Thread(target=hammer)  # half the storm off-thread
        thread.start()
        hammer()
        thread.join()

        await asyncio.sleep(0.4)  # past the debounce window
        await flusher.aclose()

        assert 1 <= replaces["n"] <= 2  # not one write per event
        on_disk = json.loads((tmp_path / "flushbench.json").read_text())
        assert isinstance(on_disk, list)  # valid final state
    finally:
        offload.shutdown()


@pytest.mark.asyncio
async def test_slow_flush_does_not_stall_the_loop(tmp_path):
    manager = _manager(tmp_path)
    real_flush = manager.flush

    def slow_flush(config=None):
        time.sleep(2.0)
        return real_flush(config)

    manager.flush = slow_flush
    offload = Offload()
    flusher = ConfigFlusher(
        manager, offload, loop=asyncio.get_running_loop(), delay=0.0
    )

    try:
        async with LoopHeartbeat(interval=0.01) as hb:
            flusher.trigger()
            await asyncio.sleep(0.3)  # the 2s flush is running on the io lane
        assert hb.max_gap_ms < 200  # loop kept beating during the slow flush
    finally:
        await flusher.aclose()
        offload.shutdown()


@pytest.mark.asyncio
async def test_aclose_drains_a_pending_write_once(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    offload = Offload()
    replaces = _count_replace(monkeypatch)
    flusher = ConfigFlusher(manager, offload, loop=asyncio.get_running_loop())

    flusher.trigger()
    await flusher.aclose()  # debounce skipped; one final write
    offload.shutdown()

    assert replaces["n"] == 1


def test_flush_now_if_dirty_writes_on_a_dead_loop(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    offload = Offload()
    replaces = _count_replace(monkeypatch)

    loop = asyncio.new_event_loop()
    loop.close()  # dead loop: trigger cannot schedule, stays dirty
    flusher = ConfigFlusher(manager, offload, loop=loop)

    flusher.trigger()
    assert flusher.dirty is True
    flusher.flush_now_if_dirty()  # sync fallback writes the pending change
    assert replaces["n"] == 1

    offload.shutdown()


def test_flush_now_if_dirty_is_a_noop_when_clean(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    offload = Offload()
    replaces = _count_replace(monkeypatch)

    loop = asyncio.new_event_loop()
    loop.close()
    flusher = ConfigFlusher(manager, offload, loop=loop)

    flusher.flush_now_if_dirty()  # nothing pending -> no write
    assert replaces["n"] == 0

    offload.shutdown()
