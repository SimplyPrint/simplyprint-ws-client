"""The decoupled scheduling contract: a client's own lifecycle (init once at
scheduler entry, tick at the tick rate) runs whether or not the client is
allocated to SimplyPrint, while allocation remains keyed on ``active``.

This is the regression net for the driver-starvation deadlock: device drivers
are started by init and re-armed by tick, and they are what *produce* device
liveness -- so a client that enters scheduling inactive (or goes inactive)
must keep running init/tick, or it can never recover.
"""

import logging
from datetime import timedelta

import pytest

from simplyprint_ws_client.core.client import Client
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.manager import ClientList
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.core.settings import ClientSettings


class _RecordingClient(Client[PrinterConfig]):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.init_calls = 0
        self.tick_calls = 0
        self.halt_calls = 0

    async def init(self):
        self.init_calls += 1

    async def tick(self, delta: timedelta):
        self.tick_calls += 1

    async def halt(self):
        self.halt_calls += 1


class _FakeManager:
    """The three calls _schedule_client makes, with no network behind them."""

    def __init__(self):
        self.allocated = set()
        self.allocate_calls = 0
        self.deallocate_calls = 0

    def is_allocated(self, client):
        return client.unique_id in self.allocated

    async def allocate(self, client):
        self.allocate_calls += 1
        self.allocated.add(client.unique_id)

    async def deallocate(self, client):
        self.deallocate_calls += 1
        self.allocated.discard(client.unique_id)


@pytest.fixture
def scheduler():
    # SINGLE mode keeps ensure_added/ensure_removed free of protocol sends, so
    # _schedule_client can be driven directly with no connection behind it.
    settings = ClientSettings(Client, PrinterConfig, camera_workers=None)
    scheduler = Scheduler(client_list=ClientList(), settings=settings)
    scheduler.manager = _FakeManager()
    return scheduler


@pytest.fixture
def client():
    config = PrinterConfig.get_new()
    config.id = 1
    return _RecordingClient(config)


@pytest.mark.asyncio
async def test_inactive_client_still_inits_and_ticks(scheduler, client):
    # The Bambu deadlock: a client entering scheduling with active=False used
    # to never get init() (so its device drivers never started) and never
    # tick. Now its own lifecycle runs; only allocation is withheld.
    client.active = False

    await scheduler._schedule_client(client)

    assert client.initialized is True
    assert client.init_calls == 1
    assert client.tick_calls == 1
    assert scheduler.manager.allocate_calls == 0


@pytest.mark.asyncio
async def test_init_runs_exactly_once_per_lifetime(scheduler, client):
    client.active = False
    await scheduler._schedule_client(client)
    # Force the tick window open again; init must not repeat.
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)
    assert client.init_calls == 1
    assert client.tick_calls == 2


@pytest.mark.asyncio
async def test_active_edge_allocates_without_reinit(scheduler, client):
    client.active = False
    await scheduler._schedule_client(client)
    assert scheduler.manager.allocate_calls == 0

    client.active = True
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)
    assert scheduler.manager.allocate_calls == 1
    assert client.init_calls == 1  # allocation is not (re)initialization


@pytest.mark.asyncio
async def test_deactivated_client_deallocates_halts_and_keeps_ticking(
    scheduler, client
):
    # The post-disconnect deadlock: deallocation used to stop the tick sweep
    # (and with it the drivers' ensure_started retries) for good. Now halt is
    # SimplyPrint-side parking only; the device side keeps ticking.
    client.active = True
    await scheduler._schedule_client(client)
    assert scheduler.manager.is_allocated(client)

    client.active = False
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)
    assert not scheduler.manager.is_allocated(client)
    assert scheduler.manager.deallocate_calls == 1
    assert client.halt_calls == 1

    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)
    assert client.tick_calls == 3
    assert client.halt_calls == 1  # halt fired once on the edge, not per pass


@pytest.mark.asyncio
async def test_failing_tick_does_not_stall_the_allocation_machine(scheduler, client):
    # tick runs ahead of the allocation state machine; a tick that raises (or
    # times out) every pass must not starve allocation -- or deactivation --
    # forever. (Found by review: the pre-decoupling order ran tick after
    # ensure_added, so a stuck tick could never block allocation.)
    async def bad_tick(delta):
        raise RuntimeError("boom")

    client.tick = bad_tick
    client.active = True
    await scheduler._schedule_client(client)
    assert scheduler.manager.is_allocated(client)

    client.active = False
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)
    assert not scheduler.manager.is_allocated(client)
    assert client.halt_calls == 1


@pytest.mark.asyncio
async def test_tick_timeout_is_logged_once_without_traceback(scheduler, client, caplog):
    async def timeout_tick(delta):
        raise TimeoutError("send ping timed out")

    client.tick = timeout_tick
    caplog.set_level(logging.DEBUG, logger=client.logger.name)

    await scheduler._schedule_client(client)
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)

    records = [record for record in caplog.records if record.name == client.logger.name]
    assert not [record for record in records if record.levelno >= logging.ERROR]
    assert len([record for record in records if record.levelno == logging.WARNING]) == 1
    assert "Error while ticking client" not in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_quiet_tick_failure_resets_after_success(scheduler, client, caplog):
    calls = 0

    async def flaky_tick(delta):
        nonlocal calls
        calls += 1
        if calls != 2:
            raise TimeoutError("send ping timed out")

    client.tick = flaky_tick
    caplog.set_level(logging.WARNING, logger=client.logger.name)

    await scheduler._schedule_client(client)
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)
    scheduler._last_ticked.clear()
    await scheduler._schedule_client(client)

    warnings = [
        record
        for record in caplog.records
        if record.name == client.logger.name and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 2
