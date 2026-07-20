import logging
from typing import List, Optional

import pytest

from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.state import PrinterStatus
from simplyprint_ws_client.integration.client import PrinterClient
from simplyprint_ws_client.integration.drivers import (
    DeviceDriver,
    DeviceReachability,
    DeviceSession,
    DeviceSource,
)


class SessionDriver(DeviceDriver):
    def __init__(self, client: "SessionClient") -> None:
        super().__init__(client, name="session")
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1


class SessionClient(PrinterClient[PrinterConfig]):
    def __init__(self) -> None:
        self.connected_edges: List[DeviceSession] = []
        self.disconnected_edges: List[Optional[object]] = []
        self.camera_clears = 0
        super().__init__(PrinterConfig.get_new(), context=ClientContext())
        self.driver = self.attach_driver(SessionDriver(self))

    async def on_device_connected(self, driver: DeviceDriver) -> None:
        await super().on_device_connected(driver)
        self.connected_edges.append(driver.session)

    async def on_device_disconnected(
        self, driver: DeviceDriver, reason: Optional[object] = None
    ) -> None:
        self.disconnected_edges.append(reason)
        await super().on_device_disconnected(driver, reason)

    def clear_camera_uri(self) -> None:
        self.camera_clears += 1


def test_driver_is_attached_during_construction():
    client = SessionClient()

    assert client.drivers == (client.driver,)
    assert not client.active


@pytest.mark.asyncio
async def test_duplicate_edges_are_idempotent_and_keep_first_down_observation():
    client = SessionClient()
    client.printer.status = PrinterStatus.PRINTING
    source = DeviceSource(lease_id=1, wire_generation=4)

    assert await client.driver.set_reachability(DeviceReachability.UP, source=source)
    up = client.driver.session
    assert client.active
    assert up.generation == 1
    assert not await client.driver.set_reachability(
        DeviceReachability.UP, source=source
    )
    assert client.driver.session == up
    assert len(client.connected_edges) == 1

    assert await client.driver.set_reachability(
        DeviceReachability.DOWN, reason="link lost", source=source
    )
    down = client.driver.session
    assert down.generation == up.generation
    assert down.reason == "link lost"
    assert not await client.driver.set_reachability(
        DeviceReachability.DOWN, reason="duplicate", source=source
    )
    assert client.driver.session == down
    assert client.driver.session.observed_at == down.observed_at
    assert len(client.disconnected_edges) == 1
    assert not client.active


@pytest.mark.asyncio
async def test_reachability_logs_identify_the_device_link(caplog):
    client = SessionClient()

    with caplog.at_level(logging.INFO):
        await client.driver.set_reachability(DeviceReachability.UP)
        await client.driver.set_reachability(
            DeviceReachability.DOWN, reason="link lost"
        )

    messages = [record.getMessage() for record in caplog.records]
    assert "Printer reachable via session device link" in messages
    assert "Printer unreachable via session device link (link lost)" in messages


@pytest.mark.asyncio
async def test_down_preserves_device_status_and_clears_camera_once():
    client = SessionClient()
    client.printer.status = PrinterStatus.PRINTING
    source = DeviceSource(lease_id=1, wire_generation=1)

    await client.driver.set_reachability(DeviceReachability.UP, source=source)
    clears_before_down = client.camera_clears
    await client.driver.set_reachability(DeviceReachability.DOWN, source=source)
    down = client.driver.session

    assert not client.active
    assert client.printer.status is PrinterStatus.PRINTING
    assert client.camera_clears == clears_before_down + 1

    assert await client.driver.set_reachability(DeviceReachability.UP, source=source)
    assert client.driver.session.generation == down.generation + 1
    assert client.active
    assert client.printer.status is PrinterStatus.PRINTING


@pytest.mark.asyncio
async def test_close_makes_session_terminal_and_rejects_late_edges():
    client = SessionClient()
    source = DeviceSource(lease_id=1, wire_generation=1)
    await client.driver.set_reachability(DeviceReachability.UP, source=source)
    edge_count = len(client.connected_edges) + len(client.disconnected_edges)

    await client.driver.close()

    assert client.driver.session.reachability is DeviceReachability.STOPPED
    assert client.driver.stops == 1
    assert not await client.driver.set_reachability(
        DeviceReachability.UP,
        source=DeviceSource(lease_id=2, wire_generation=1),
    )
    assert not await client.driver.set_reachability(
        DeviceReachability.DOWN, source=source
    )
    assert len(client.connected_edges) + len(client.disconnected_edges) == edge_count


@pytest.mark.asyncio
async def test_new_source_with_reused_wire_generation_wins_over_stale_source():
    client = SessionClient()
    client.printer.status = PrinterStatus.PRINTING
    old_source = DeviceSource(lease_id=1, wire_generation=1)
    new_source = DeviceSource(lease_id=2, wire_generation=1)

    await client.driver.set_reachability(DeviceReachability.UP, source=old_source)
    assert await client.driver.set_reachability(
        DeviceReachability.UP, source=new_source
    )
    current = client.driver.session
    assert current.generation == 2
    assert current.source == new_source

    assert not await client.driver.set_reachability(
        DeviceReachability.DOWN, reason="stale", source=old_source
    )
    assert client.driver.session == current

    assert await client.driver.set_reachability(
        DeviceReachability.DOWN, reason="current", source=new_source
    )
    assert client.driver.session.reachability is DeviceReachability.DOWN
    assert client.driver.session.reason == "current"
