"""LeaseDriver restart/start-failure contracts.

restart() must actually bounce: the old lease is closed and awaited (handlers
detached, refcount released) BEFORE the new lease is built, a same-URL restart
additionally trips the shared Reconnecting wire (siblings may keep it leased),
concurrent restarts coalesce, and a stop() racing the sequence wins. Persistent
start() failures must surface after a bounded interval: the printer is reported
offline once, with the failure as the reason.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.integration.drivers import (
    DeviceReachability,
    LeaseDriver,
)
from simplyprint_ws_client.wire.errors import AuthenticationError
from simplyprint_ws_client.wire.events import ActivityTimeout, Disconnected, WireEvent
from simplyprint_ws_client.wire.keepalive import KeepaliveTimeout
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import NotConnected, Transport


class FakeClient:
    """The minimal client surface a lease driver touches."""

    def __init__(self, loop) -> None:
        self._loop = loop
        self.logger = logging.getLogger("test.lease.driver")
        self.connected_edges: List[object] = []
        self.disconnected_edges: List[object] = []

    @property
    def event_loop(self):
        return self._loop

    def submit_to_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def on_device_connected(self, driver) -> None:
        self.connected_edges.append(driver)

    async def on_device_disconnected(self, driver, reason=None) -> None:
        self.disconnected_edges.append(reason)

    def clear_camera_uri(self) -> None:  # pragma: no cover - unused
        pass


class TrippableTransport(Reconnecting):
    """A Reconnecting whose trips a test records (hooks never run)."""

    def __init__(self) -> None:
        super().__init__(yarl.URL("ws://fake"))
        self.trips: List[int] = []

    async def open(self) -> None:  # pragma: no cover - never started
        pass

    async def recv(self) -> Optional[object]:  # pragma: no cover - never started
        await asyncio.Event().wait()
        return None

    async def write(self, message: object) -> None:  # pragma: no cover
        pass

    async def aclose(self) -> None:  # pragma: no cover - never started
        pass

    def trip(self, generation: int, reason: Exception) -> None:
        self.trips.append(generation)
        super().trip(generation, reason)


class FakeLease:
    """The lease surface the driver touches, with a journaled close."""

    def __init__(self, url: yarl.URL, transport, journal: List[str]) -> None:
        self.url = url
        self.transport = transport
        self.event_bus = EventBus()
        self.closed = False
        self.connected = False
        self.last_activity = asyncio.get_event_loop().time()
        self._journal = journal
        self.close_gate: Optional[asyncio.Event] = None

    @property
    def generation(self) -> int:
        return self.transport.generation

    async def close(self) -> None:
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed = True
        self.event_bus.clear_all()  # mirrors the real Lease.close
        self._journal.append("close")

    def close_soon(self) -> None:
        self.closed = True
        self._journal.append("close_soon")

    def create_task(self, coro):  # pragma: no cover - unused (never connected)
        return asyncio.get_event_loop().create_task(coro)


class FakeLeaseDriver(LeaseDriver):
    default_name = "fake"

    def __init__(self, client, url, **kwargs) -> None:
        super().__init__(client, url, **kwargs)
        self.journal: List[str] = []
        self.built: List[FakeLease] = []
        self.transport_for_next = None

    def acquire_lease(self, url, options) -> FakeLease:
        transport = self.transport_for_next or TrippableTransport()
        lease = FakeLease(yarl.URL(str(url)), transport, self.journal)
        self.built.append(lease)
        self.journal.append("start")
        return lease


class PooledTransport(Transport):
    def __init__(self, url: yarl.URL) -> None:
        self.url = url
        self.events: EventBus[WireEvent] = EventBus()
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.starts = 0
        self.stops = 0

    @property
    def connected(self) -> bool:
        return False

    def supervising(self) -> bool:
        return self.starts > self.stops

    def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1

    async def send(self, message: object) -> None:
        raise NotConnected("test transport is down")

    def trip(self, generation: int, reason: Exception) -> None:
        return None


class PooledLeaseDriver(LeaseDriver):
    def __init__(self, client, url, pool) -> None:
        super().__init__(client, url)
        self.pool = pool

    def acquire_lease(self, url, options):
        return self.pool.connect(url)


async def _settle(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


def _driver(url: str = "ws://host/a") -> FakeLeaseDriver:
    client = FakeClient(asyncio.get_event_loop())
    return FakeLeaseDriver(client, lambda: url)


@pytest.mark.asyncio
async def test_restart_closes_the_old_lease_before_building_the_new():
    driver = _driver()
    driver.start()
    (old,) = driver.built

    driver.restart()
    await _settle(lambda: len(driver.built) == 2)

    assert old.closed
    assert driver.journal == ["start", "close", "start"]  # fully sequenced
    assert driver.lease is driver.built[1]


@pytest.mark.asyncio
async def test_old_lease_events_never_reach_handlers_after_restart():
    driver = _driver()
    driver.start()
    (old,) = driver.built

    driver.restart()
    await _settle(lambda: len(driver.built) == 2)

    # The old lease's bus was cleared by its close -- a stale Disconnected
    # cannot clobber the fresh link.
    before = list(driver.client.disconnected_edges)
    await old.event_bus.emit(Disconnected(1))
    assert driver.client.disconnected_edges == before


@pytest.mark.asyncio
async def test_authentication_rejection_requests_credential_refresh():
    driver = _driver()
    driver.start()
    lease = driver.lease
    driver.request_credential_refresh = MagicMock()

    await lease.event_bus.emit(
        Disconnected(lease.generation, code=AuthenticationError("auth"))
    )

    assert driver.session.reachability is DeviceReachability.DOWN
    driver.request_credential_refresh.assert_called_once_with()


@pytest.mark.asyncio
async def test_route_activity_timeout_marks_only_its_current_device_source_down():
    driver = _driver()
    driver.start()
    lease = driver.lease
    await driver.set_reachability(DeviceReachability.UP)

    await lease.event_bus.emit(
        ActivityTimeout(
            lease.generation,
            code=KeepaliveTimeout("silent"),
            last_activity=lease.last_activity,
        )
    )

    assert driver.session.reachability is DeviceReachability.DOWN
    assert isinstance(driver.client.disconnected_edges[-1], KeepaliveTimeout)


@pytest.mark.asyncio
async def test_activity_after_timeout_decision_discards_the_stale_offline_edge():
    driver = _driver()
    driver.start()
    lease = driver.lease
    await driver.set_reachability(DeviceReachability.UP)
    stale_activity = lease.last_activity
    lease.last_activity += 1

    await lease.event_bus.emit(
        ActivityTimeout(
            lease.generation,
            code=KeepaliveTimeout("silent"),
            last_activity=stale_activity,
        )
    )

    assert driver.session.reachability is DeviceReachability.UP


@pytest.mark.asyncio
async def test_concurrent_restarts_coalesce_into_one_rerun():
    driver = _driver()
    driver.start()

    driver.restart()
    driver.restart()  # lands while the first is in flight -> one rerun
    driver.restart()  # also coalesces into the same rerun
    await _settle(lambda: len(driver.built) == 3 and not driver._restarting)

    # initial start + first sequence + exactly one coalesced rerun.
    assert driver.journal == ["start", "close", "start", "close", "start"]


@pytest.mark.asyncio
async def test_same_url_restart_trips_the_shared_reconnecting_wire():
    driver = _driver()
    driver.start()
    (old,) = driver.built

    driver.restart()
    await _settle(lambda: len(driver.built) == 2)

    assert old.transport.trips == [old.transport.generation]


@pytest.mark.asyncio
async def test_moved_url_restart_does_not_kick_the_old_wire():
    url = ["ws://host/a"]
    client = FakeClient(asyncio.get_event_loop())
    driver = FakeLeaseDriver(client, lambda: url[0])
    driver.start()
    (old,) = driver.built

    url[0] = "ws://host/b"  # credential/host rotation: endpoint moves
    driver.restart()
    await _settle(lambda: len(driver.built) == 2)

    assert old.transport.trips == []  # siblings handle their own refresh
    assert driver.built[1].url == yarl.URL("ws://host/b")


@pytest.mark.asyncio
async def test_moved_endpoint_releases_old_transport_before_acquiring_new():
    urls = ["ws://host/a"]
    transports: List[PooledTransport] = []

    def build(url: yarl.URL, _params: object) -> PooledTransport:
        transport = PooledTransport(url)
        transports.append(transport)
        return transport

    pool = Pool(build=build, key=lambda url, _params: str(url))
    driver = PooledLeaseDriver(
        FakeClient(asyncio.get_running_loop()),
        lambda: urls[0],
        pool,
    )
    driver.start()
    old_lease = driver.lease
    old_transport = transports[0]

    urls[0] = "ws://host/b"
    driver.ensure_current()
    await _settle(lambda: driver.lease is not None and driver.lease.url.host == "host")
    await _settle(lambda: driver.lease is not old_lease)

    assert old_lease.closed
    assert old_transport.stops == 1
    assert set(pool.endpoints) == {"ws://host/b"}
    assert len(transports) == 2

    await driver.close()


@pytest.mark.asyncio
async def test_stop_racing_an_in_flight_restart_wins():
    driver = _driver()
    driver.start()
    (old,) = driver.built
    old.close_gate = asyncio.Event()  # park the sequence inside close()

    driver.restart()
    await asyncio.sleep(0.02)  # the sequence is awaiting the gated close
    driver.stop()
    old.close_gate.set()
    await _settle(lambda: not driver._restarting)

    assert driver.lease is None  # teardown was not resurrected
    assert len(driver.built) == 1  # no new lease after stop


@pytest.mark.asyncio
async def test_persistent_start_failures_flip_the_down_edge_once():
    client = FakeClient(asyncio.get_event_loop())
    boom = ["host not set"]

    def url() -> str:
        if boom:
            raise ValueError(boom[0])
        return "ws://host/a"

    driver = FakeLeaseDriver(client, url)
    driver.START_FAILURE_DOWN_AFTER = 0.01

    driver.start()
    await asyncio.sleep(0.02)
    for _ in range(4):
        driver.start()
    await _settle(lambda: client.disconnected_edges)

    assert len(client.disconnected_edges) == 1  # edge fired exactly once
    assert "host not set" in str(client.disconnected_edges[0])
    assert driver.session.reachability is DeviceReachability.DOWN
    assert "host not set" in str(driver.session.reason)

    # A real recovery starts a new reachable generation and re-arms the edge.
    boom.clear()
    driver.start()
    assert driver.lease is not None
    await driver.set_reachability(DeviceReachability.UP)
    assert driver.session.reachability is DeviceReachability.UP
    assert driver.session.generation == 1

    # A later outage fires a fresh edge.
    driver.stop()
    boom.append("gone again")
    driver.start()
    await asyncio.sleep(0.02)
    driver.start()
    await _settle(lambda: len(client.disconnected_edges) == 2)
    assert "gone again" in str(client.disconnected_edges[1])
    assert driver.session.reachability is DeviceReachability.DOWN
