"""Core WS protocol events are delivered through the existing EventBus."""

import asyncio

import pytest

from types import SimpleNamespace

from simplyprint_ws_client.events import EventBus


from simplyprint_ws_client.wire.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.core.protocol.connection import SimplyPrintConnection
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionEstablishedEvent,
    SimplyPrintConnectionIncomingEvent,
    SimplyPrintConnectionLostEvent,
)


@pytest.mark.asyncio
async def test_transport_events_emit_protocol_events_in_order():
    conn = SimplyPrintConnection()
    payload = '{"type":"pong"}'
    order = []

    conn.event_bus.on(
        SimplyPrintConnectionEstablishedEvent, lambda e: order.append(("est", e.v))
    )

    async def on_incoming(msg, v):
        await asyncio.sleep(0)
        order.append(("inc", msg.type, v))

    conn.event_bus.on(SimplyPrintConnectionIncomingEvent, on_incoming)
    conn.event_bus.on(
        SimplyPrintConnectionLostEvent, lambda e: order.append(("lost", e.v))
    )

    await conn.protocol._on_connected(Connected(1))
    await conn.protocol._on_message(MessageReceived(1, payload))
    await conn.protocol._on_disconnected(Disconnected(1))

    assert order == [("est", 0), ("inc", "pong", 0), ("lost", 0)]
    assert conn.v == 1


@pytest.mark.asyncio
async def test_protocol_event_emission_does_not_drop_bursts():
    conn = SimplyPrintConnection()
    payload = '{"type":"pong"}'
    got = []

    async def on_incoming(_msg, v):
        got.append(v)

    conn.event_bus.on(SimplyPrintConnectionIncomingEvent, on_incoming)

    for _ in range(500):
        await conn.protocol._on_message(MessageReceived(1, payload))

    assert got == [0] * 500


# --------------------------------------------------------------------------- #
# The courier'd dispatch contract: transport-bus events are queued, ordered,
# lossless -- and a wedged client handler can no longer stall the transport's
# emit (the supervise/recv liveness property the keepalive-wedge bug broke).
# --------------------------------------------------------------------------- #


def _fake_transport() -> SimpleNamespace:
    """The only surface ``attach`` touches: a transport-shaped events bus."""
    return SimpleNamespace(events=EventBus())


async def _drain(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("courier did not drain in time")
        await asyncio.sleep(0.002)


@pytest.mark.asyncio
async def test_transport_bus_events_dispatch_in_order_without_loss():
    """Connected, a 500-message burst, and Disconnected ride one FIFO courier:
    exact order, zero drops, and every message observes the pre-bump ``v``."""
    conn = SimplyPrintConnection()
    conn.use_running_loop()
    transport = _fake_transport()
    conn.protocol.attach(transport)

    order = []
    conn.event_bus.on(
        SimplyPrintConnectionEstablishedEvent, lambda e: order.append(("est", e.v))
    )
    conn.event_bus.on(
        SimplyPrintConnectionIncomingEvent,
        lambda msg, v: order.append(("inc", v)),
    )
    conn.event_bus.on(
        SimplyPrintConnectionLostEvent, lambda e: order.append(("lost", e.v))
    )

    payload = '{"type":"pong"}'
    await transport.events.emit(Connected(1))
    for _ in range(500):
        await transport.events.emit(MessageReceived(1, payload))
    await transport.events.emit(Disconnected(1))

    await _drain(lambda: len(order) == 502)
    assert order[0] == ("est", 0)
    assert order[1:-1] == [("inc", 0)] * 500  # every message saw the old epoch
    assert order[-1] == ("lost", 0)
    assert conn.v == 1
    conn.protocol.detach()


@pytest.mark.asyncio
async def test_wedged_client_handler_does_not_block_the_transport_bus():
    """The original production bug: a handler that never returns must not stop
    the transport's emit from returning (recv/drop-detection liveness)."""
    conn = SimplyPrintConnection()
    conn.use_running_loop()
    transport = _fake_transport()
    conn.protocol.attach(transport)

    entered = asyncio.Event()
    never = asyncio.Event()

    async def wedged(_msg, _v):
        entered.set()
        await never.wait()

    conn.event_bus.on(SimplyPrintConnectionIncomingEvent, wedged)

    payload = '{"type":"pong"}'
    await transport.events.emit(MessageReceived(1, payload))
    await asyncio.wait_for(entered.wait(), 2.0)

    # The handler is wedged -- further transport emits must return at once.
    await asyncio.wait_for(
        transport.events.emit(MessageReceived(1, payload)), timeout=0.5
    )
    assert conn.protocol._courier.pending() >= 1  # queued, not dispatched

    never.set()
    conn.protocol.detach()


@pytest.mark.asyncio
async def test_stalled_dispatch_logs_the_watchdog_error(caplog):
    """Past the depth threshold the stall is surfaced once per attach epoch."""
    conn = SimplyPrintConnection()
    conn.use_running_loop()
    transport = _fake_transport()
    conn.protocol.attach(transport)

    never = asyncio.Event()

    async def wedged(_msg, _v):
        await never.wait()

    conn.event_bus.on(SimplyPrintConnectionIncomingEvent, wedged)

    payload = '{"type":"pong"}'
    threshold = conn.protocol.DISPATCH_STALL_THRESHOLD
    with caplog.at_level("ERROR"):
        for _ in range(threshold + 5):
            await transport.events.emit(MessageReceived(1, payload))

    stalls = [r for r in caplog.records if "dispatch stalled" in r.message]
    assert len(stalls) == 1  # reported once, not per event

    never.set()
    conn.protocol.detach()


@pytest.mark.asyncio
async def test_detach_drops_queued_events_and_stops_dispatch():
    conn = SimplyPrintConnection()
    conn.use_running_loop()
    transport = _fake_transport()
    conn.protocol.attach(transport)

    got = []
    conn.event_bus.on(SimplyPrintConnectionIncomingEvent, lambda m, v: got.append(v))

    never = asyncio.Event()

    async def wedged(_msg, _v):
        await never.wait()

    conn.event_bus.on(SimplyPrintConnectionIncomingEvent, wedged)

    payload = '{"type":"pong"}'
    for _ in range(5):
        await transport.events.emit(MessageReceived(1, payload))

    conn.protocol.detach()
    never.set()
    await asyncio.sleep(0.05)
    # At most the one in-flight dispatch landed; the queued rest were dropped.
    assert len(got) <= 1
    # Emits after detach are not even enqueued.
    await transport.events.emit(MessageReceived(1, payload))
    await asyncio.sleep(0.02)
    assert len(got) <= 1
