import asyncio
import subprocess
import sys

import pytest

from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    AsyncTransport,
    Connected,
    Disconnected,
    MessageReceived,
    TransportEvent,
)
from simplyprint_ws_client.contrib.connection.websocket import (
    AsyncWebSocketPool,
    AsyncWebSocketTransport,
    WsParams,
)
from simplyprint_ws_client.contrib.connection.websocket.base import (
    WebSocket,
    WebSocketClosed,
)
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.utils.backoff import ConstantBackoff


def test_importing_async_ws_does_not_load_websockets():
    code = (
        "import sys\n"
        "import simplyprint_ws_client.contrib.connection.websocket.aio\n"
        "assert 'websockets' not in sys.modules, 'websockets eagerly imported'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


class _FakeWire(WebSocket):
    def __init__(self):
        self.inbound = asyncio.Queue()
        self.connects = 0
        self.closed = 0
        self._open = False

    async def connect(self, url: str, **params) -> None:
        self.connects += 1
        self.url = url
        self.params = params
        self._open = True

    async def send(self, data: str) -> None:
        self.sent = data

    async def recv(self):
        item = await self.inbound.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed += 1
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open


@pytest.mark.asyncio
async def test_async_ws_transport_emits_events_and_reconnects():
    wires = []

    def wire_factory(params, logger):
        wire = _FakeWire()
        wires.append(wire)
        return wire

    transport = AsyncWebSocketTransport(
        WsParams("ws://printer/ws"),
        wire_factory=wire_factory,
        backoff=ConstantBackoff(0),
        connect_kwargs={"open_timeout": 0.5},
    )
    seen = []
    transport.events.on(Connected, lambda e: seen.append("up"))
    transport.events.on(MessageReceived, lambda e: seen.append(("msg", e.payload)))
    transport.events.on(Disconnected, lambda e: seen.append(("down", e.transient)))

    transport.start()
    try:
        for _ in range(200):
            await asyncio.sleep(0.005)
            if wires:
                break
        assert wires[0].url == "ws://printer/ws"
        assert wires[0].params == {"open_timeout": 0.5}

        await wires[0].inbound.put("hello")
        await wires[0].inbound.put(WebSocketClosed("drop"))

        for _ in range(200):
            await asyncio.sleep(0.005)
            if len(wires) >= 2 and ("msg", "hello") in seen:
                break
    finally:
        transport.stop()
        await asyncio.sleep(0.02)

    assert "up" in seen
    assert ("msg", "hello") in seen
    assert ("down", True) in seen
    assert len(wires) >= 2


class _RecordingTransport(AsyncTransport):
    def __init__(self, params):
        self.params = params
        self.events = EventBus()
        self.state = ConnectionState.OFFLINE
        self.started = 0
        self.stopped = 0
        self.sent = []
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        self.started += 1
        self._connected = True

    def stop(self) -> None:
        self.stopped += 1
        self._connected = False

    async def send(self, payload) -> None:
        self.sent.append(payload)

    def emit(self, event: TransportEvent) -> None:
        self.events.emit_sync(type(event), event)


@pytest.mark.asyncio
async def test_async_ws_pool_shares_routes_and_refcounts():
    built = []

    def factory(params):
        transport = _RecordingTransport(params)
        built.append(transport)
        return transport

    pool = AsyncWebSocketPool(transport_factory=factory)
    params = WsParams("ws://printer/ws")

    a = await pool.connect(params)
    b = await pool.connect(params)
    got_a, got_b = [], []
    a.on_message(got_a.append)
    b.on_message(got_b.append)

    assert len(built) == 1
    assert built[0].started == 1

    built[0].emit(MessageReceived("frame"))
    await a.send("command")

    assert got_a == ["frame"]
    assert got_b == ["frame"]
    assert built[0].sent == ["command"]

    await a.close()
    assert built[0].stopped == 0
    await b.close()
    assert built[0].stopped == 1
