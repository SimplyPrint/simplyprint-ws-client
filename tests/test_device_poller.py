"""DevicePoller edge contract: silence — including cold-start silence — flips
the disconnected edge exactly once; contact flips connected; DeviceAuthError
runs the single-flight credential refresh."""

import asyncio
import logging

import pytest

from simplyprint_ws_client.integration.driver import DeviceAuthError
from simplyprint_ws_client.integration.poller import DevicePoller


class FakeClient:
    """The minimal client surface a poller touches."""

    def __init__(self, loop, poll):
        self._loop = loop
        self.logger = logging.getLogger("test.poller")
        self.poll_device = poll
        self.connected_edges = []
        self.disconnected_edges = []
        self.refreshes = 0

    @property
    def event_loop(self):
        return self._loop

    def submit_to_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def on_device_connected(self, driver):
        self.connected_edges.append(driver)

    async def on_device_disconnected(self, driver, reason=None):
        self.disconnected_edges.append((driver, reason))

    async def refresh_device_credentials(self, driver):
        self.refreshes += 1
        return False


async def _wait_for(predicate, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_cold_start_silence_fires_the_disconnected_edge_once():
    # A device NEVER reached still reports its offline edge offline_after
    # seconds after the poller starts -- an unplugged printer must not stay
    # forever unknown.
    async def poll():
        raise ConnectionError("unreachable")

    client = FakeClient(asyncio.get_running_loop(), poll)
    poller = DevicePoller(
        client, interval=0.01, offline_after=0.05, failure_backoff=0.01
    )
    poller.start()
    try:
        await _wait_for(lambda: client.disconnected_edges)
        assert poller.is_connected is False
        # The edge fires exactly once while silence continues.
        await asyncio.sleep(0.1)
        assert len(client.disconnected_edges) == 1
    finally:
        poller.stop()


@pytest.mark.asyncio
async def test_contact_fires_connected_then_silence_disconnects():
    outcomes = [None, None, ConnectionError("gone")]

    async def poll():
        outcome = outcomes.pop(0) if outcomes else ConnectionError("gone")
        if outcome is not None:
            raise outcome

    client = FakeClient(asyncio.get_running_loop(), poll)
    poller = DevicePoller(
        client, interval=0.01, offline_after=0.05, failure_backoff=0.01
    )
    poller.start()
    try:
        await _wait_for(lambda: client.connected_edges)
        assert poller.is_connected is True
        await _wait_for(lambda: client.disconnected_edges)
        assert poller.is_connected is False
    finally:
        poller.stop()


@pytest.mark.asyncio
async def test_auth_error_runs_single_flight_credential_refresh():
    async def poll():
        raise DeviceAuthError("session expired")

    client = FakeClient(asyncio.get_running_loop(), poll)
    poller = DevicePoller(
        client, interval=0.01, offline_after=10.0, failure_backoff=0.01
    )
    poller.start()
    try:
        await _wait_for(lambda: client.refreshes >= 1)
    finally:
        poller.stop()
