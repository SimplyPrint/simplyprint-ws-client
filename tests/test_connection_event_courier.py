"""Core WS protocol events are delivered through the existing EventBus."""

import asyncio

import pytest

from simplyprint_ws_client.common.wire.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.core.protocol.connection import CloudConnection
from simplyprint_ws_client.core.protocol.events import (
    CloudConnectionEstablishedEvent,
    CloudConnectionIncomingEvent,
    CloudConnectionLostEvent,
)


@pytest.mark.asyncio
async def test_transport_events_emit_protocol_events_in_order():
    conn = CloudConnection()
    payload = '{"type":"pong"}'
    order = []

    conn.event_bus.on(
        CloudConnectionEstablishedEvent, lambda e: order.append(("est", e.v))
    )

    async def on_incoming(msg, v):
        await asyncio.sleep(0)
        order.append(("inc", msg.type, v))

    conn.event_bus.on(CloudConnectionIncomingEvent, on_incoming)
    conn.event_bus.on(CloudConnectionLostEvent, lambda e: order.append(("lost", e.v)))

    await conn.protocol._on_connected(Connected(1))
    await conn.protocol._on_message(MessageReceived(1, payload))
    await conn.protocol._on_disconnected(Disconnected(1))

    assert order == [("est", 0), ("inc", "pong", 0), ("lost", 0)]
    assert conn.v == 1


@pytest.mark.asyncio
async def test_protocol_event_emission_does_not_drop_bursts():
    conn = CloudConnection()
    payload = '{"type":"pong"}'
    got = []

    async def on_incoming(_msg, v):
        got.append(v)

    conn.event_bus.on(CloudConnectionIncomingEvent, on_incoming)

    for _ in range(500):
        await conn.protocol._on_message(MessageReceived(1, payload))

    assert got == [0] * 500
