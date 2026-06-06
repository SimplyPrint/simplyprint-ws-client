"""Guard for the core WS connection's courier-based event delivery.

``Connection`` used to fan its lifetime/message events with
``event_bus.emit_task`` (a ``concurrent.futures.Future`` per event); it now posts
them to a same-loop, UNBOUNDED, async-sink :class:`Courier`. These tests pin the
properties that swap must preserve: strict FIFO order across the generation
sequence (Established -> Incoming -> Lost), delivery to *async* listeners (the
real risk -- a sync-only sink would silently drop them), and no dropped events
under a burst. The existing connection contract tests (version generation,
message routing, transport seam, message order) cover the rest, unchanged.
"""

import asyncio

import pytest

from simplyprint_ws_client.core.ws_protocol.connection import Connection
from simplyprint_ws_client.core.ws_protocol.events import (
    ConnectionEstablishedEvent,
    ConnectionIncomingEvent,
    ConnectionLostEvent,
)


async def _wait_until(predicate, attempts=400):
    for _ in range(attempts):
        await asyncio.sleep(0.005)
        if predicate():
            break


@pytest.mark.asyncio
async def test_events_deliver_in_order_to_sync_and_async_listeners():
    conn = Connection()
    order = []

    conn.event_bus.on(ConnectionEstablishedEvent, lambda e: order.append(("est", e.v)))

    async def on_incoming(msg, v):
        await asyncio.sleep(
            0
        )  # suspend mid-event; the serialized drain must hold order
        order.append(("inc", msg, v))

    conn.event_bus.on(ConnectionIncomingEvent, on_incoming)
    conn.event_bus.on(ConnectionLostEvent, lambda e: order.append(("lost", e.v)))

    # The exact shapes the loop posts: instance, then class + args, then instance.
    conn._post(ConnectionEstablishedEvent(1))
    conn._post(ConnectionIncomingEvent, "payload", 1)
    conn._post(ConnectionLostEvent(1))

    await _wait_until(lambda: len(order) == 3)

    assert order == [("est", 1), ("inc", "payload", 1), ("lost", 1)]


@pytest.mark.asyncio
async def test_events_are_never_dropped_under_a_burst():
    conn = Connection()
    got = []

    async def on_incoming(_msg, v):
        got.append(v)

    conn.event_bus.on(ConnectionIncomingEvent, on_incoming)

    for i in range(500):
        conn._post(ConnectionIncomingEvent, "m", i)

    await _wait_until(lambda: len(got) == 500)

    assert got == list(range(500))  # UNBOUNDED: ordered and not one lost
