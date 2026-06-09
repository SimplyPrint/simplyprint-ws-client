"""Core WS protocol events are delivered through the existing EventBus."""

import asyncio

import pytest

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.core.ws_protocol.connection import Connection
from simplyprint_ws_client.core.ws_protocol.events import (
    ConnectionEstablishedEvent,
    ConnectionIncomingEvent,
    ConnectionLostEvent,
)


@pytest.mark.asyncio
async def test_transport_events_emit_protocol_events_in_order():
    conn = Connection()
    payload = '{"type":"pong"}'
    order = []

    conn.event_bus.on(ConnectionEstablishedEvent, lambda e: order.append(("est", e.v)))

    async def on_incoming(msg, v):
        await asyncio.sleep(0)
        order.append(("inc", msg.type, v))

    conn.event_bus.on(ConnectionIncomingEvent, on_incoming)
    conn.event_bus.on(ConnectionLostEvent, lambda e: order.append(("lost", e.v)))

    await conn._on_connected(Connected(1))
    await conn._on_message(MessageReceived(1, payload))
    await conn._on_disconnected(Disconnected(1))

    assert order == [("est", 0), ("inc", "pong", 0), ("lost", 0)]
    assert conn.v == 1


@pytest.mark.asyncio
async def test_protocol_event_emission_does_not_drop_bursts():
    conn = Connection()
    payload = '{"type":"pong"}'
    got = []

    async def on_incoming(_msg, v):
        got.append(v)

    conn.event_bus.on(ConnectionIncomingEvent, on_incoming)

    for _ in range(500):
        await conn._on_message(MessageReceived(1, payload))

    assert got == [0] * 500
