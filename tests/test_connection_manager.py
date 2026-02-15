import asyncio

import pytest

from simplyprint_ws_client.core.client import Client, ClientState
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.connection_manager import (
    ClientConnectionManager,
    ClientList,
)
from simplyprint_ws_client.core.ws_protocol.connection import ConnectionMode
from simplyprint_ws_client.core.ws_protocol.events import ConnectionLostEvent


class _DummyConnection:
    def __init__(self):
        self.disconnect_calls = 0

    async def disconnect(self):
        self.disconnect_calls += 1


class _DummyView:
    def __init__(self, client: Client, connection: _DummyConnection):
        self._clients = {client.unique_id}
        self.connection = connection

    def discard(self, client: Client):
        self._clients.discard(client.unique_id)

    def __len__(self):
        return len(self._clients)


@pytest.mark.asyncio
async def test_deallocate_waits_for_connection_lost_handlers():
    client = Client(PrinterConfig.get_new())
    client.v = 3
    client.state = ClientState.CONNECTED

    manager = ClientConnectionManager(ConnectionMode.SINGLE, ClientList())
    connection = _DummyConnection()
    manager.client_views[client.unique_id] = _DummyView(client, connection)

    gate = asyncio.Event()
    handler_started = asyncio.Event()

    async def blocking_listener(_event: ConnectionLostEvent):
        handler_started.set()
        await gate.wait()

    # Run before built-in listeners to maximize race surface.
    client.event_bus.on(ConnectionLostEvent, blocking_listener, priority=100)

    task = asyncio.create_task(manager.deallocate(client))

    await asyncio.wait_for(handler_started.wait(), timeout=1.0)
    await asyncio.sleep(0.01)

    # Regression guard: with old emit_task behavior this completed immediately.
    assert not task.done()

    gate.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert connection.disconnect_calls == 1


@pytest.mark.asyncio
async def test_deallocate_does_not_leave_late_connection_lost_event():
    client = Client(PrinterConfig.get_new())
    client.v = 5
    client.state = ClientState.CONNECTED

    manager = ClientConnectionManager(ConnectionMode.SINGLE, ClientList())
    manager.client_views[client.unique_id] = _DummyView(client, _DummyConnection())

    async def slow_listener(_event: ConnectionLostEvent):
        await asyncio.sleep(0.03)

    # Delay execution of built-in _on_connection_lost in the listener chain.
    client.event_bus.on(ConnectionLostEvent, slow_listener, priority=100)

    await manager.deallocate(client)

    # Simulate immediate re-allocation transition.
    client.state = ClientState.NOT_CONNECTED
    await asyncio.sleep(0.05)

    # Regression guard: stale async loss event must not flip us back to CONNECTING.
    assert client.state == ClientState.NOT_CONNECTED
