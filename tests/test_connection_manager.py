import asyncio

import pytest

import simplyprint_ws_client.core.manager as connection_manager_module
from simplyprint_ws_client.core.client import Client, ClientState
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.manager import (
    ClientConnectionManager,
    ClientList,
    ClientView,
)
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionIncomingEvent,
    SimplyPrintConnectionLostEvent,
    SimplyPrintConnectionOutgoingEvent,
)
from simplyprint_ws_client.core.protocol.messages import PingMsg
from simplyprint_ws_client.events import EventBus


class _DummyConnection:
    def __init__(self):
        self.disconnect_calls = 0

    async def disconnect(self):
        self.disconnect_calls += 1


class _RecordingConnection:
    instances = []

    def __init__(self, *args, **kwargs):
        self.event_bus = EventBus()
        self.connected = False
        self.v = 0
        self.connect_hints = []
        self.disconnect_calls = 0
        self.stop_calls = 0
        self.loop_task = None
        self.url = "ws://fake"
        type(self).instances.append(self)

    async def connect(self, hint=None):
        self.connect_hints.append(hint)

    async def disconnect(self):
        self.disconnect_calls += 1

    def stop(self):
        self.stop_calls += 1


class _DummyView:
    def __init__(self, client: Client, connection: _DummyConnection):
        self._clients = {client.unique_id}
        self.connection = connection

    def discard(self, client: Client):
        self._clients.discard(client.unique_id)

    def __len__(self):
        return len(self._clients)


def _client_list(count: int):
    client_list = ClientList()
    clients = [Client(PrinterConfig.get_new()) for _ in range(count)]

    for client in clients:
        client_list.add(client)

    return client_list, clients


def _fake_connections(monkeypatch):
    _RecordingConnection.instances = []
    # 2.0 re-pin: SimplyPrintConnection was renamed SimplyPrintConnection; the seam
    # and every assertion are unchanged.
    monkeypatch.setattr(
        connection_manager_module, "SimplyPrintConnection", _RecordingConnection
    )
    return _RecordingConnection.instances


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

    async def blocking_listener(_event: SimplyPrintConnectionLostEvent):
        handler_started.set()
        await gate.wait()

    # Run before built-in listeners to maximize race surface.
    client.event_bus.on(SimplyPrintConnectionLostEvent, blocking_listener, priority=100)

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

    async def slow_listener(_event: SimplyPrintConnectionLostEvent):
        await asyncio.sleep(0.03)

    # Delay execution of built-in _on_connection_lost in the listener chain.
    client.event_bus.on(SimplyPrintConnectionLostEvent, slow_listener, priority=100)

    await manager.deallocate(client)

    # Simulate immediate re-allocation transition.
    client.state = ClientState.NOT_CONNECTED
    await asyncio.sleep(0.05)

    # Regression guard: stale async loss event must not flip us back to CONNECTING.
    assert client.state == ClientState.NOT_CONNECTED


def test_manager_rejects_invalid_max_clients_per_connection():
    with pytest.raises(ValueError):
        ClientConnectionManager(
            ConnectionMode.MULTI,
            ClientList(),
            max_clients_per_connection=0,
        )


@pytest.mark.asyncio
async def test_multi_mode_without_capacity_reuses_one_view(monkeypatch):
    connections = _fake_connections(monkeypatch)
    client_list, clients = _client_list(3)
    manager = ClientConnectionManager(ConnectionMode.MULTI, client_list)

    for client in clients:
        await manager.allocate(client)

    assert len(manager.views) == 1
    assert len(connections) == 1
    assert len(next(iter(manager.views))) == 3


@pytest.mark.asyncio
async def test_multi_mode_capacity_spreads_clients_across_views(monkeypatch):
    connections = _fake_connections(monkeypatch)
    client_list, clients = _client_list(5)
    manager = ClientConnectionManager(
        ConnectionMode.MULTI,
        client_list,
        max_clients_per_connection=2,
    )

    for client in clients:
        await manager.allocate(client)

    assert len(manager.views) == 3
    assert len(connections) == 3
    assert sorted(len(view) for view in manager.views) == [1, 2, 2]
    assert all(len(view) <= 2 for view in manager.views)
    assert all(
        hint.mode is ConnectionMode.MULTI
        for connection in connections
        for hint in connection.connect_hints
    )


@pytest.mark.asyncio
async def test_deallocate_removes_empty_view(monkeypatch):
    connections = _fake_connections(monkeypatch)
    client_list, clients = _client_list(2)
    manager = ClientConnectionManager(
        ConnectionMode.MULTI,
        client_list,
        max_clients_per_connection=1,
    )

    await manager.allocate(clients[0])
    first_connection = manager.get_connection_for_client(clients[0])

    await manager.deallocate(clients[0])

    assert len(manager.views) == 0
    assert first_connection.disconnect_calls == 1

    await manager.allocate(clients[1])

    assert len(manager.views) == 1
    assert len(connections) == 2
    assert manager.get_connection_for_client(clients[1]) is not first_connection


@pytest.mark.asyncio
async def test_multi_mode_reuses_partially_free_view(monkeypatch):
    connections = _fake_connections(monkeypatch)
    client_list, clients = _client_list(3)
    manager = ClientConnectionManager(
        ConnectionMode.MULTI,
        client_list,
        max_clients_per_connection=2,
    )

    await manager.allocate(clients[0])
    await manager.allocate(clients[1])
    first_connection = manager.get_connection_for_client(clients[0])

    await manager.deallocate(clients[0])
    await manager.allocate(clients[2])

    assert len(manager.views) == 1
    assert len(connections) == 1
    assert manager.get_connection_for_client(clients[2]) is first_connection
    assert sorted(len(view) for view in manager.views) == [2]


@pytest.mark.asyncio
async def test_multi_mode_outgoing_messages_use_assigned_connection(monkeypatch):
    _fake_connections(monkeypatch)
    client_list, clients = _client_list(2)
    manager = ClientConnectionManager(
        ConnectionMode.MULTI,
        client_list,
        max_clients_per_connection=1,
    )

    for client in clients:
        await manager.allocate(client)

    sent_by_connection = {}
    for connection in manager.connections:
        sent_by_connection[connection] = []
        connection.event_bus.on(
            SimplyPrintConnectionOutgoingEvent,
            lambda msg, _v, connection=connection: sent_by_connection[
                connection
            ].append(msg.for_client),
        )

    await clients[0].send(PingMsg())
    await clients[1].send(PingMsg())

    first_connection = manager.get_connection_for_client(clients[0])
    second_connection = manager.get_connection_for_client(clients[1])
    assert sent_by_connection[first_connection] == [clients[0].unique_id]
    assert sent_by_connection[second_connection] == [clients[1].unique_id]


@pytest.mark.asyncio
async def test_multi_mode_incoming_messages_route_only_inside_assigned_view(
    monkeypatch,
):
    _fake_connections(monkeypatch)
    client_list, clients = _client_list(2)
    manager = ClientConnectionManager(
        ConnectionMode.MULTI,
        client_list,
        max_clients_per_connection=1,
    )

    for client in clients:
        await manager.allocate(client)

    received = {client.unique_id: [] for client in clients}
    for client in clients:
        client.event_bus.on(
            SimplyPrintConnectionIncomingEvent,
            lambda msg, _v, client=client: received[client.unique_id].append(
                msg.for_client
            ),
            priority=100,
        )

    first_connection = manager.get_connection_for_client(clients[0])
    stray = PingMsg()
    stray.for_client = clients[1].unique_id
    await first_connection.event_bus.emit(SimplyPrintConnectionIncomingEvent, stray, 0)

    targeted = PingMsg()
    targeted.for_client = clients[0].unique_id
    await first_connection.event_bus.emit(
        SimplyPrintConnectionIncomingEvent, targeted, 0
    )

    assert received[clients[0].unique_id] == [clients[0].unique_id]
    assert received[clients[1].unique_id] == []


@pytest.mark.asyncio
async def test_emit_all_uses_snapshot_when_view_is_mutated():
    client_list = ClientList()
    c1 = Client(PrinterConfig.get_new())
    c2 = Client(PrinterConfig.get_new())
    c3 = Client(PrinterConfig.get_new())
    client_list.add(c1)
    client_list.add(c2)
    client_list.add(c3)

    view = ClientView(ConnectionMode.MULTI, object(), client_list)
    view.add(c1)
    view.add(c2)
    view.add(c3)

    received = []

    async def on_c1():
        received.append(c1.unique_id)
        # Mutate the live set mid-fanout. Snapshot iteration must still deliver to c3.
        view.discard(c3)

    async def on_c2():
        received.append(c2.unique_id)

    async def on_c3():
        received.append(c3.unique_id)

    c1.event_bus.on("fanout", on_c1)
    c2.event_bus.on("fanout", on_c2)
    c3.event_bus.on("fanout", on_c3)

    await view.emit("fanout")

    assert set(received) == {c1.unique_id, c2.unique_id, c3.unique_id}


@pytest.mark.asyncio
async def test_emit_all_ignores_stale_client_ids():
    client_list = ClientList()
    c1 = Client(PrinterConfig.get_new())
    c2 = Client(PrinterConfig.get_new())
    client_list.add(c1)
    client_list.add(c2)

    view = ClientView(ConnectionMode.MULTI, object(), client_list)
    view.add(c1)
    view.add(c2)
    view.clients.add("stale-client-id")

    received = []

    async def on_c1():
        received.append(c1.unique_id)

    async def on_c2():
        received.append(c2.unique_id)

    c1.event_bus.on("fanout", on_c1)
    c2.event_bus.on("fanout", on_c2)

    await view.emit("fanout")

    assert set(received) == {c1.unique_id, c2.unique_id}
