"""Brand-free tests for the threaded WS ``WebSocketPool`` on the Courier.

A WS link is 1:1, and its wire fires callbacks on its own daemon thread; these
pin that a lease delivers on the consumer loop, ref-counts the transport, and
sends through the wire -- with a fake wire (no websocket-client, no socket).
"""

import asyncio
import threading

import pytest

from simplyprint_ws_client.contrib.connection.websocket import (
    WebSocketPool,
    WebSocketTransport,
    WsParams,
)
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider


class FakeWire:
    """A websocket-client stand-in whose callbacks tests fire from any thread."""

    def __init__(self):
        self._on_message = None
        self._on_connected = None
        self._on_disconnected = None
        self._connected = False
        self.started = 0
        self.stopped = 0
        self.sent = []

    @property
    def connected(self):
        return self._connected

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1
        self._connected = False

    def send(self, data):
        self.sent.append(data)
        return True

    # -- test drivers (simulate the wire's daemon thread) --
    def fire_connected(self):
        self._connected = True
        self._on_connected()

    def fire_message(self, message):
        self._on_message(message)

    def fire_disconnected(self, reason="dropped"):
        self._connected = False
        self._on_disconnected(reason)


def _pool(loop):
    wires = []

    def transport_factory(params):
        wire = FakeWire()
        wires.append(wire)
        return WebSocketTransport(params, wire_factory=lambda p, log: wire)

    pool = WebSocketPool(
        transport_factory=transport_factory,
        event_loop_provider=EventLoopProvider(loop=loop),
    )
    return pool, wires


def _from_thread(fn, *args):
    t = threading.Thread(target=fn, args=args)
    t.start()
    t.join(1.0)


@pytest.mark.asyncio
async def test_message_from_wire_thread_arrives_on_the_loop():
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    pool, wires = _pool(loop)

    got = []
    lease = pool.connect(WsParams("ws://printer:9999"))
    lease.on_message(lambda msg: got.append((msg, threading.get_ident())))

    wire = wires[0]
    assert wire.started == 1
    _from_thread(wire.fire_connected)
    _from_thread(wire.fire_message, "frame")

    for _ in range(200):
        await asyncio.sleep(0.005)
        if got:
            break

    assert got and got[0][0] == "frame"
    assert got[0][1] == loop_thread  # delivered on the loop, not the wire thread

    lease.close()
    pool.stop()


@pytest.mark.asyncio
async def test_connect_disconnect_reach_the_lease():
    loop = asyncio.get_running_loop()
    pool, wires = _pool(loop)

    events = []
    lease = pool.connect(WsParams("ws://printer:9999"))
    lease.on_connected(lambda: events.append("up"))
    lease.on_disconnected(lambda e: events.append(("down", e.transient)))

    _from_thread(wires[0].fire_connected)
    _from_thread(wires[0].fire_disconnected)

    for _ in range(200):
        await asyncio.sleep(0.005)
        if len(events) == 2:
            break

    assert events == ["up", ("down", True)]
    lease.close()
    pool.stop()


@pytest.mark.asyncio
async def test_send_and_refcount():
    loop = asyncio.get_running_loop()
    pool, wires = _pool(loop)
    params = WsParams("ws://printer:9999")

    a = pool.connect(params)
    b = pool.connect(params)
    assert len(wires) == 1  # one shared transport per endpoint

    assert a.send("hello") is True
    assert wires[0].sent == ["hello"]

    a.close()
    assert params in pool._refs  # b still holds it
    b.close()
    assert params not in pool._refs  # last lease left -> torn down
    pool.stop()
