"""R1 client-side hub policy (docs/r1-agreement.md, "Client-side hub policy").

- a printer's hub is a consistent hash of its ``unique_id``: the same in every
  process, so a restart moves nobody;
- hub count is ``clamp(ceil(printers / 32), 1, 32)``;
- a hub-count change moves only the printers whose slot changed (the fewest
  adds: the server's 500-add burst never refills);
- hub opens and adds are jittered.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta

import pytest
from yarl import URL

import simplyprint_ws_client.core.client as client_module
import simplyprint_ws_client.core.manager as manager_module
from simplyprint_ws_client.core.client import ClientState
from simplyprint_ws_client.core.manager import (
    ClientConnectionManager,
    ClientList,
    hub_count,
    jump_hash,
    stable_key,
)
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.events import EventBus

from tests._wire_harness import new_client

_WS = URL("wss://ws.example")
IDS = [f"printer-{i:04d}" for i in range(1000)]


class _Connection:
    """Stands in for SimplyPrintConnection: records connects, never opens."""

    def __init__(self, *args, **kwargs):
        self.event_bus = EventBus()
        self.connected = False
        self.v = 0
        self.connects = 0
        self.loop_task = None
        self.url = "ws://fake"

    async def connect(self, hint=None):
        self.connects += 1

    async def disconnect(self):
        pass

    def stop(self):
        pass


def _manager(monkeypatch, ids, **kwargs):
    monkeypatch.setattr(manager_module, "SimplyPrintConnection", _Connection)
    client_list = ClientList()
    clients = [new_client(uid) for uid in ids]
    for client in clients:
        client_list.add(client)
    manager = ClientConnectionManager(ConnectionMode.MULTI, client_list, _WS, **kwargs)
    return manager, clients


# -- hub count -----------------------------------------------------------------


@pytest.mark.parametrize(
    "printers, hubs",
    [
        (0, 1),
        (1, 1),
        (5, 1),
        (32, 1),
        (33, 2),
        (64, 2),
        (65, 3),
        (500, 16),
        (32 * 32, 32),
        (32 * 32 + 1, 32),
        (100_000, 32),
    ],
)
def test_hub_count_is_clamped_ceil_of_printers_over_32(printers, hubs):
    assert hub_count(printers) == hubs


# -- stable assignment -----------------------------------------------------------


def test_the_key_is_a_fixed_digest_not_the_salted_builtin_hash():
    # hash() of a str changes per process (PYTHONHASHSEED); blake2b does not.
    expected = int.from_bytes(
        hashlib.blake2b(b"printer-0001", digest_size=8).digest(), "big"
    )
    assert stable_key("printer-0001") == expected
    assert stable_key(17) == stable_key("17")


def test_jump_hash_stays_in_range_and_spreads():
    for buckets in (1, 2, 7, 32):
        slots = [jump_hash(stable_key(uid), buckets) for uid in IDS]
        assert set(slots) == set(range(buckets))
        # Roughly even: no slot holds more than twice its share.
        assert max(slots.count(b) for b in range(buckets)) < 2 * len(IDS) / buckets


def test_assignment_is_the_same_across_restarts(monkeypatch):
    # Two "processes": fresh managers, printers registered in different orders.
    first, _ = _manager(monkeypatch, IDS[:100])
    second, _ = _manager(monkeypatch, list(reversed(IDS[:100])))

    assert first.hub_count() == second.hub_count() == 4
    assert {uid: first.hub_index(uid) for uid in IDS[:100]} == {
        uid: second.hub_index(uid) for uid in IDS[:100]
    }


@pytest.mark.parametrize("hubs", range(1, 32))
def test_a_count_change_moves_only_printers_to_the_new_hub(hubs):
    before = {uid: jump_hash(stable_key(uid), hubs) for uid in IDS}
    after = {uid: jump_hash(stable_key(uid), hubs + 1) for uid in IDS}
    moved = [uid for uid in IDS if before[uid] != after[uid]]

    # Every move goes to the one new hub; nobody shuffles between old hubs.
    assert all(after[uid] == hubs for uid in moved)
    # About 1/(hubs+1) of the printers -- the minimum any scheme can do.
    assert len(moved) < 2 * len(IDS) / (hubs + 1)


# -- the manager -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_manager_places_each_printer_on_its_hash_slot(monkeypatch):
    manager, clients = _manager(monkeypatch, IDS[:70])  # 70 printers: 3 hubs
    for client in clients:
        await manager.allocate(client)

    assert manager.hub_count() == 3
    assert len(manager.views) == 3
    for client in clients:
        view = manager.client_views[client.unique_id]
        assert view.index == manager.hub_index(client.unique_id)
        assert not manager.is_misplaced(client)


@pytest.mark.asyncio
async def test_a_small_install_keeps_one_hub(monkeypatch):
    manager, clients = _manager(monkeypatch, IDS[:5])
    for client in clients:
        await manager.allocate(client)

    assert len(manager.views) == 1
    (view,) = manager.views
    assert len(view) == 5


@pytest.mark.asyncio
async def test_growing_past_32_misplaces_only_the_new_hubs_share(monkeypatch):
    manager, clients = _manager(monkeypatch, IDS[:32])
    for client in clients:
        await manager.allocate(client)
    (hub0,) = manager.views
    assert not any(manager.is_misplaced(c) for c in clients)

    # The 33rd printer: two hubs now.
    newcomer = new_client(IDS[32])
    manager.client_list.add(newcomer)
    misplaced = [c for c in clients if manager.is_misplaced(c)]

    assert misplaced  # with 32 printers some do land in slot 1
    assert {c.unique_id for c in misplaced} == {
        uid for uid in IDS[:32] if jump_hash(stable_key(uid), 2) == 1
    }

    # Moving them (what the scheduler does) leaves everyone else in place.
    for client in misplaced:
        await manager.deallocate(client)
        await manager.allocate(client)
    await manager.allocate(newcomer)

    for client in clients:
        view = manager.client_views[client.unique_id]
        assert view.index == manager.hub_index(client.unique_id)
        if client not in misplaced:
            assert view is hub0


@pytest.mark.asyncio
async def test_a_hub_is_reopened_in_its_own_slot(monkeypatch):
    manager, clients = _manager(monkeypatch, IDS[:40])
    for client in clients:
        await manager.allocate(client)
    slot = manager.hub_index(clients[0].unique_id)
    peers = [c for c in clients if manager.hub_index(c.unique_id) == slot]

    for client in peers:  # the hub empties and is closed
        await manager.deallocate(client)
    assert slot not in {view.index for view in manager.views}

    await manager.allocate(peers[0])
    assert manager.client_views[peers[0].unique_id].index == slot


# -- jitter -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_several_hubs_a_new_hub_opens_after_a_jitter(monkeypatch):
    manager, clients = _manager(monkeypatch, IDS[:40])  # 2 hubs
    monkeypatch.setattr(manager_module, "_jitter", lambda upper: upper / 2)

    await manager.allocate(clients[0])
    view = manager.client_views[clients[0].unique_id]

    assert view.opening is not None and not view.opening.done()
    assert view.connection.connects == 0  # not yet
    # A second printer on the same hub does not open it early.
    same = next(
        c
        for c in clients[1:]
        if manager.hub_index(c.unique_id) == manager.hub_index(clients[0].unique_id)
    )
    await manager.allocate(same)
    assert view.connection.connects == 0

    manager.stop()  # cancels the pending open
    with pytest.raises(asyncio.CancelledError):
        await view.opening
    assert view.connection.connects == 0


@pytest.mark.asyncio
async def test_a_single_hub_opens_at_once(monkeypatch):
    manager, clients = _manager(monkeypatch, IDS[:5])
    monkeypatch.setattr(manager_module, "_jitter", lambda upper: upper / 2)

    await manager.allocate(clients[0])
    view = manager.client_views[clients[0].unique_id]

    assert view.opening is None
    assert view.connection.connects == 1


@pytest.mark.asyncio
async def test_a_hub_open_spreads_its_adds(monkeypatch):
    class Clock:
        min = datetime.min
        current = datetime(2026, 1, 1)

        @classmethod
        def now(cls):
            return cls.current

    monkeypatch.setattr(client_module, "datetime", Clock)
    monkeypatch.setattr(client_module, "_jitter", lambda upper: upper / 2)
    client = new_client("printer-a")
    client.state = ClientState.NOT_CONNECTED
    sent = []

    async def capture(message, *_args, **_kwargs):
        sent.append(message)

    monkeypatch.setattr(client, "send", capture)

    client.spread_add(32)  # a full hub: up to 1.6 s, here 0.8 s

    assert not await client.ensure_added(ConnectionMode.MULTI)
    assert sent == []
    Clock.current += timedelta(seconds=0.79)
    assert not await client.ensure_added(ConnectionMode.MULTI)
    assert sent == []
    Clock.current += timedelta(seconds=0.02)
    assert not await client.ensure_added(ConnectionMode.MULTI)
    assert len(sent) == 1


def test_the_add_jitter_never_shortens_a_server_retry_after(monkeypatch):
    monkeypatch.setattr(client_module, "_jitter", lambda upper: upper / 2)
    client = new_client("printer-b")
    client._apply_server_retry_after("Rate limited. Retry-After: 249s")

    client.spread_add(32)

    assert client._pending_action_delay == timedelta(seconds=249)


# -- a stable count: grow at once, shrink after a grace, not mid-replay ----------


@pytest.fixture
def clock(monkeypatch):
    """The manager's monotonic clock, in seconds, advanced by hand."""
    now = {"s": 1000.0}
    monkeypatch.setattr(manager_module, "_monotonic", lambda: now["s"])
    return now


async def _allocated(monkeypatch, ids):
    manager, clients = _manager(monkeypatch, ids)
    for client in clients:
        await manager.allocate(client)
    return manager, clients


@pytest.mark.asyncio
async def test_the_hub_count_shrinks_only_after_staying_lower_for_the_grace(
    monkeypatch, clock
):
    manager, clients = await _allocated(monkeypatch, IDS[:33])
    assert manager.hub_count() == 2

    manager.client_list.remove(clients[-1])  # 32: one hub would do
    assert manager.hub_count() == 2  # the scheduler's next pass sees it
    clock["s"] += manager_module.HUB_SHRINK_GRACE_S - 1
    assert manager.hub_count() == 2
    assert not any(manager.is_misplaced(c) for c in clients[:32])

    clock["s"] += 1
    assert manager.hub_count() == 1


@pytest.mark.asyncio
async def test_a_printer_leaving_and_coming_back_moves_nobody(monkeypatch, clock):
    # 32 <-> 33 used to flip 1 <-> 2 hubs, moving half the printers each time
    # (an add from a burst that never refills, and a full-state resend).
    manager, clients = await _allocated(monkeypatch, IDS[:33])
    flapping = clients[-1]
    for _ in range(3):
        manager.client_list.remove(flapping)
        assert manager.hub_count() == 2
        clock["s"] += manager_module.HUB_SHRINK_GRACE_S - 1
        manager.client_list.add(flapping)
        assert manager.hub_count() == 2  # back before the grace ran out
    clock["s"] += manager_module.HUB_SHRINK_GRACE_S
    assert manager.hub_count() == 2
    assert not any(manager.is_misplaced(c) for c in clients)


@pytest.mark.asyncio
async def test_growth_applies_at_once(monkeypatch, clock):
    manager, clients = await _allocated(monkeypatch, IDS[:32])
    manager.client_list.add(new_client(IDS[32]))
    assert manager.hub_count() == 2
    assert any(manager.is_misplaced(c) for c in clients)


@pytest.mark.asyncio
async def test_a_replay_is_counted_once_it_is_done(monkeypatch, clock):
    manager, clients = await _allocated(monkeypatch, IDS[:20])
    with manager.replaying():
        for uid in IDS[20:70]:
            manager.client_list.add(new_client(uid))
            # Not re-hashed on a partial list (2 hubs at 33, 3 at 65...).
            assert manager.hub_count() == 1
            assert not any(manager.is_misplaced(c) for c in clients)
    assert manager.hub_count() == 3  # 70 printers, in one step


@pytest.mark.asyncio
async def test_the_scheduler_places_nobody_while_a_replay_runs(monkeypatch):
    from simplyprint_ws_client.core.scheduler import Scheduler
    from simplyprint_ws_client.core.settings import ClientSettings

    monkeypatch.setattr(manager_module, "SimplyPrintConnection", _Connection)
    scheduler = Scheduler(ClientList(), ClientSettings(mode=ConnectionMode.MULTI))
    scheduler.use_running_loop()
    client = new_client(IDS[0])
    scheduler.client_list.add(client)

    with scheduler.replaying():
        await scheduler._schedule_client(client)
        assert not scheduler.manager.is_allocated(client)

    await scheduler._schedule_client(client)
    assert scheduler.manager.is_allocated(client)
