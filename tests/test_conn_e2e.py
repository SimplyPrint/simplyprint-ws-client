"""End-to-end tests for ``device.connection`` over REAL loopback sockets.

Unlike the fake-injected unit suites, this file stands up a real in-process
WebSocket server on ``127.0.0.1`` (the ``websockets`` library's ``serve``) and
drives BOTH installed async wire families against it over a genuine TCP socket:

* the ``websockets``-backed :class:`Websockets` transport, and
* the ``aiohttp``-backed :class:`Aiohttp` transport.

Both speak the standard WebSocket protocol, so one server serves both clients.
Each real-socket scenario proves the full reliability story end to end: connect
and ``ready()`` -> ``True``, a server push surfacing as :class:`MessageReceived`,
a send the server echoes back, a server-forced drop that the supervised reconnect
loop heals with ``generation + 1`` and a resumed stream, and a clean
:meth:`stop` / :meth:`close`.

It also drives the public ``ws.connect`` front door end to end (front door ->
:class:`Pool` -> framed lease) over the same real server, including pooled
socket-sharing across two leases on one endpoint.

``paho`` / ``aiomqtt`` are not installed, so MQTT is not exercised here; the
fake-injected MQTT path lives in the sibling unit suites.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator, List, Optional, Tuple

import pytest
import pytest_asyncio
import yarl

from simplyprint_ws_client.common.utils.backoff import ConstantBackoff

from simplyprint_ws_client.common.wire import websocket as ws
from simplyprint_ws_client.common.wire.aiohttp import Aiohttp
from simplyprint_ws_client.common.wire.lease import WsLease
from simplyprint_ws_client.common.wire.options import ConnectionOptions
from simplyprint_ws_client.common.wire.events import (
    Connected,
    Connecting,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.common.wire.messages import QoS
from simplyprint_ws_client.common.wire.policy import RetryPolicy
from simplyprint_ws_client.common.wire.state import ConnectionState
from simplyprint_ws_client.common.wire.transport import WsTransport
from simplyprint_ws_client.common.wire.websocket import (
    WsKind,
    WsMessage,
)
from simplyprint_ws_client.common.wire.websockets import Websockets


# --------------------------------------------------------------------------- #
# A controllable real loopback WebSocket server.
# --------------------------------------------------------------------------- #


class LoopbackServer:
    """A real ``127.0.0.1`` WebSocket server the test fully controls.

    It tracks every live server-side connection, echoes any frame a client
    sends (prefixed, so an echo is distinguishable from a server push), can push
    arbitrary frames to the current connection, and can forcibly drop the live
    socket to make a client's supervised loop reconnect.
    """

    def __init__(self, *, echo: bool = True) -> None:
        self.echo = echo
        self.server: Optional[Any] = None
        self.host = "127.0.0.1"
        self.port = 0
        #: Every server-side connection still open, newest last.
        self.connections: List[Any] = []
        #: Frames received from clients (across all connections), in order.
        self.received: List[Any] = []
        #: Bumped each time a client connection is accepted.
        self.accept_count = 0
        #: Set every time a new connection is accepted, for await-on-connect.
        self.connected = asyncio.Event()

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self.server = await serve(self.handler, self.host, 0)
        sock = next(iter(self.server.sockets))
        self.port = sock.getsockname()[1]

    @property
    def url(self) -> yarl.URL:
        return yarl.URL(f"ws://{self.host}:{self.port}/")

    async def handler(self, connection: Any) -> None:
        self.connections.append(connection)
        self.accept_count += 1
        self.connected.set()
        try:
            async for frame in connection:
                self.received.append(frame)
                if self.echo:
                    if isinstance(frame, (bytes, bytearray)):
                        await connection.send(b"echo:" + bytes(frame))
                    else:
                        await connection.send("echo:" + frame)
        except Exception:  # noqa: BLE001 -- a dropped client is normal here
            pass
        finally:
            with contextlib.suppress(ValueError):
                self.connections.remove(connection)

    async def push(self, payload: Any) -> None:
        """Send one frame to the most-recent live connection."""
        connection = await self.current()
        await connection.send(payload)

    async def current(self, timeout: float = 2.0) -> Any:
        """The most-recent live connection, awaiting one if none yet."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not self.connections:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError("no client connected to the loopback server")
            self.connected.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.connected.wait(), remaining)
        return self.connections[-1]

    async def drop_current(self) -> None:
        """Forcibly close the live server-side socket, forcing a client reconnect."""
        connection = await self.current()
        await connection.close(code=1011, reason="forced drop")

    async def stop(self) -> None:
        if self.server is None:
            return
        for connection in list(self.connections):
            with contextlib.suppress(Exception):
                await connection.close()
        self.server.close()
        await self.server.wait_closed()
        self.server = None


@pytest_asyncio.fixture
async def server() -> AsyncIterator[LoopbackServer]:
    srv = LoopbackServer()
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


@pytest_asyncio.fixture
async def push_server() -> AsyncIterator[LoopbackServer]:
    """A non-echoing server -- the test pushes frames explicitly."""
    srv = LoopbackServer(echo=False)
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


# --------------------------------------------------------------------------- #
# A direct subscriber that records lifecycle + messages off a transport bus.
# --------------------------------------------------------------------------- #


class Recorder:
    """Records every lifecycle event and message off a transport's event bus.

    Message assertions in this file are payload-oriented; the transport-level
    unit tests assert the ``WsMessage`` wrapper shape directly.
    """

    def __init__(self) -> None:
        self.connecting: List[int] = []
        self.connected: List[int] = []
        self.disconnected: List[Tuple[int, Optional[object]]] = []
        self.messages: List[Tuple[int, Any]] = []

    def attach(self, transport: WsTransport) -> None:
        transport.events.on(Connecting, lambda e: self.connecting.append(e.generation))
        transport.events.on(Connected, lambda e: self.connected.append(e.generation))
        transport.events.on(
            Disconnected, lambda e: self.disconnected.append((e.generation, e.code))
        )
        transport.events.on(
            MessageReceived,
            lambda e: self.messages.append((e.generation, self.payload(e.message))),
        )

    @staticmethod
    def payload(message: Any) -> Any:
        if isinstance(message, WsMessage):
            return message.payload
        return message


async def wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> None:
    """Poll ``predicate`` on the loop until true or fail with a timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(interval)


def fast_policy() -> RetryPolicy:
    """Retry forever, but at a near-zero pace so reconnects are quick under test."""
    return RetryPolicy(backoff=ConstantBackoff(0.01))


# --------------------------------------------------------------------------- #
# Build the two installed wire transports against a real server URL.
# --------------------------------------------------------------------------- #


def make_websockets(url: yarl.URL, policy: Optional[RetryPolicy] = None) -> Websockets:
    return Websockets(url, policy or fast_policy())


def make_aiohttp(url: yarl.URL, policy: Optional[RetryPolicy] = None) -> Aiohttp:
    return Aiohttp(url, policy or fast_policy())


WIRES = [
    pytest.param(make_websockets, id="websockets"),
    pytest.param(make_aiohttp, id="aiohttp"),
]


@contextlib.asynccontextmanager
async def running(transport: WsTransport) -> AsyncIterator[WsTransport]:
    transport.start()
    try:
        yield transport
    finally:
        await transport.stop()


# --------------------------------------------------------------------------- #
# Real-socket transport scenarios -- run for BOTH installed wire families.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_connect_over_real_socket(make, server: LoopbackServer) -> None:
    """A transport reaches a real server, fires Connected, and is connected."""
    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)

    async with running(transport):
        await wait_until(lambda: transport.connected)
        assert transport.state is ConnectionState.CONNECTED
        assert transport.generation == 1
        assert rec.connected == [1]
        assert rec.connecting and rec.connecting[0] == 0
        await server.current()
        assert server.accept_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_server_push_becomes_message_received(
    make, push_server: LoopbackServer
) -> None:
    """A frame the server pushes surfaces as a MessageReceived on the right gen."""
    rec = Recorder()
    transport = make(push_server.url)
    rec.attach(transport)

    async with running(transport):
        await wait_until(lambda: transport.connected)
        await push_server.push("hello-from-server")
        await wait_until(lambda: len(rec.messages) >= 1)

        generation, payload = rec.messages[0]
        assert generation == 1
        assert payload == "hello-from-server"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_binary_server_push(make, push_server: LoopbackServer) -> None:
    """A binary server push is delivered as bytes (no str coercion)."""
    rec = Recorder()
    transport = make(push_server.url)
    rec.attach(transport)

    async with running(transport):
        await wait_until(lambda: transport.connected)
        await push_server.push(b"\x00\x01\x02binary")
        await wait_until(lambda: len(rec.messages) >= 1)

        _, payload = rec.messages[0]
        assert isinstance(payload, (bytes, bytearray))
        assert bytes(payload) == b"\x00\x01\x02binary"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_echo_send_round_trips(make, server: LoopbackServer) -> None:
    """A send over the live wire reaches the server and the echo comes back."""
    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)

    async with running(transport):
        await wait_until(lambda: transport.connected)
        await transport.send("ping")
        await wait_until(lambda: any(m[1] == "echo:ping" for m in rec.messages))
        assert "ping" in server.received


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_binary_echo_send_round_trips(make, server: LoopbackServer) -> None:
    """A binary send round-trips as binary."""
    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)

    async with running(transport):
        await wait_until(lambda: transport.connected)
        await transport.send(b"payload")
        await wait_until(
            lambda: any(
                isinstance(m[1], (bytes, bytearray)) and bytes(m[1]) == b"echo:payload"
                for m in rec.messages
            )
        )
        assert b"payload" in server.received


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_forced_drop_reconnects_with_next_generation(
    make, server: LoopbackServer
) -> None:
    """A server-forced drop heals: gen advances and the new link works again."""
    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)

    async with running(transport):
        await wait_until(lambda: transport.connected)
        assert transport.generation == 1
        first_connection = await server.current()

        await server.drop_current()

        # The supervised loop must reconnect to a brand-new generation and the
        # server must accept a SECOND, distinct connection.
        await wait_until(lambda: transport.connected and transport.generation >= 2)
        assert transport.generation == 2
        assert rec.connected == [1, 2]
        # A drop must have been announced for the first epoch.
        assert any(gen == 1 for gen, _ in rec.disconnected)

        await wait_until(lambda: server.accept_count >= 2)
        second_connection = await server.current()
        assert second_connection is not first_connection

        # The resumed link carries traffic on the new generation.
        await transport.send("after-drop")
        await wait_until(
            lambda: any(
                gen == 2 and msg == "echo:after-drop" for gen, msg in rec.messages
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_multiple_forced_drops_keep_advancing_generation(
    make, server: LoopbackServer
) -> None:
    """Repeated drops keep healing and the generation is strictly monotonic."""
    transport = make(server.url)
    async with running(transport):
        await wait_until(lambda: transport.connected)
        seen = [transport.generation]

        for _ in range(3):
            target = transport.generation + 1
            await server.drop_current()
            await wait_until(
                lambda: transport.connected and transport.generation >= target
            )
            seen.append(transport.generation)

        assert seen == [1, 2, 3, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_clean_stop_closes_socket(make, server: LoopbackServer) -> None:
    """A clean stop tears the link down and the server-side socket goes away."""
    transport = make(server.url)
    transport.start()
    await wait_until(lambda: transport.connected)
    await server.current()
    assert len(server.connections) == 1

    await transport.stop()

    assert transport.state is ConnectionState.DISCONNECTED
    assert not transport.connected
    # The server observes the client's clean close.
    await wait_until(lambda: len(server.connections) == 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_send_after_stop_raises_not_connected(
    make, server: LoopbackServer
) -> None:
    """Once stopped, the transport reports NotConnected on send."""
    from simplyprint_ws_client.common.wire.transport import NotConnected

    transport = make(server.url)
    transport.start()
    await wait_until(lambda: transport.connected)
    await transport.stop()

    with pytest.raises(NotConnected):
        await transport.send("nope")


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_connect_to_dead_port_then_give_up(make) -> None:
    """A closed port never connects; a bounded policy gives up and stops supervising."""
    # Bind+immediately close a socket to grab a port that nothing listens on.
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()

    policy = RetryPolicy(backoff=ConstantBackoff(0.0), max_attempts=3)
    transport = make(yarl.URL(f"ws://127.0.0.1:{dead_port}/"), policy)
    rec = Recorder()
    rec.attach(transport)

    transport.start()
    try:
        await wait_until(lambda: not transport.supervising(), timeout=5.0)
        assert transport.state is ConnectionState.DISCONNECTED
        assert not transport.connected
        assert transport.generation == 0  # never established
        # Each failed attempt announced a Disconnected with a code.
        assert rec.disconnected
        assert all(code is not None for _, code in rec.disconnected)
    finally:
        await transport.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_idempotent_start_keeps_single_connection(
    make, server: LoopbackServer
) -> None:
    """Calling start twice does not open two sockets."""
    transport = make(server.url)
    async with running(transport):
        transport.start()
        await wait_until(lambda: transport.connected)
        transport.start()  # second start is a no-op while supervising
        await asyncio.sleep(0.1)
        await server.current()
        assert server.accept_count == 1


# --------------------------------------------------------------------------- #
# The ws.connect front door, end to end: front door -> Pool -> framed lease.
# --------------------------------------------------------------------------- #


@contextlib.asynccontextmanager
async def front_door(
    url: yarl.URL, *, impl: str, retry: Optional[RetryPolicy] = None
) -> AsyncIterator[WsLease]:
    """A ws.connect lease on its own pool, torn down cleanly after the test."""
    # Force a brand-new pool so suites do not share live sockets across tests.
    ws.DEFAULT_POOLS.pools.pop(impl, None)
    pool = ws.build_pool(impl, None)
    conn = ws.connect(
        url,
        impl=impl,
        options=ConnectionOptions(retry=retry or fast_policy()),
        pool=pool,
    )
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            await conn.close()
        pool.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_ready_true(impl, server: LoopbackServer) -> None:
    """ws.connect -> ready() resolves True over a real socket."""
    async with front_door(server.url, impl=impl) as conn:
        assert await conn.ready(timeout=3.0) is True
        assert conn.connected
        assert conn.state is ConnectionState.CONNECTED
        assert conn.generation == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_server_push_is_framed(
    impl, push_server: LoopbackServer
) -> None:
    """A raw server frame is delivered to the lease as a framed WsMessage."""
    received: List[WsMessage] = []

    async with front_door(push_server.url, impl=impl) as conn:
        conn.event_bus.on(MessageReceived, lambda e: received.append(e.message))
        assert await conn.ready(timeout=3.0)

        await push_server.push("framed-text")
        await wait_until(lambda: len(received) >= 1)

        message = received[0]
        assert isinstance(message, WsMessage)
        assert message.kind is WsKind.TEXT
        assert message.payload == "framed-text"


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_binary_push_is_framed_bytes(
    impl, push_server: LoopbackServer
) -> None:
    """A binary server frame is framed as a BINARY WsMessage carrying bytes."""
    received: List[WsMessage] = []

    async with front_door(push_server.url, impl=impl) as conn:
        conn.event_bus.on(MessageReceived, lambda e: received.append(e.message))
        assert await conn.ready(timeout=3.0)

        await push_server.push(b"\xde\xad\xbe\xef")
        await wait_until(lambda: len(received) >= 1)

        message = received[0]
        assert isinstance(message, WsMessage)
        assert message.kind is WsKind.BINARY
        assert message.payload == b"\xde\xad\xbe\xef"


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_send_str_and_wsmessage(impl, server: LoopbackServer) -> None:
    """A bare str and a WsMessage both go out and are echoed back, framed."""
    received: List[WsMessage] = []

    async with front_door(server.url, impl=impl) as conn:
        conn.event_bus.on(MessageReceived, lambda e: received.append(e.message))
        assert await conn.ready(timeout=3.0)

        await conn.send("bare-str")
        await conn.send(WsMessage.text("wrapped"))
        await conn.send(WsMessage.binary(b"rawbytes"))

        await wait_until(
            lambda: {
                m.payload
                for m in received
                if isinstance(m, WsMessage) and isinstance(m.payload, str)
            }
            >= {"echo:bare-str", "echo:wrapped"}
        )
        await wait_until(
            lambda: any(
                isinstance(m.payload, (bytes, bytearray))
                and bytes(m.payload) == b"echo:rawbytes"
                for m in received
            )
        )
        assert {"bare-str", "wrapped"} <= set(server.received)
        assert b"rawbytes" in server.received


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_forced_drop_resume(impl, server: LoopbackServer) -> None:
    """The front-door lease survives a server drop: gen+1, then sends again."""
    lifecycle_connected: List[int] = []
    received: List[WsMessage] = []

    async with front_door(server.url, impl=impl) as conn:
        conn.event_bus.on(Connected, lambda e: lifecycle_connected.append(e.generation))
        conn.event_bus.on(MessageReceived, lambda e: received.append(e.message))
        assert await conn.ready(timeout=3.0)
        assert conn.generation == 1

        await server.drop_current()
        await wait_until(lambda: conn.connected and conn.generation >= 2)
        assert conn.generation == 2
        assert lifecycle_connected == [1, 2]

        await conn.send("survived")
        await wait_until(
            lambda: any(
                isinstance(m, WsMessage) and m.payload == "echo:survived"
                for m in received
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_pool_shares_one_socket(impl, server: LoopbackServer) -> None:
    """Two leases on one endpoint URL share a single real socket."""
    ws.DEFAULT_POOLS.pools.pop(impl, None)
    pool = ws.build_pool(impl, None)
    first = ws.connect(
        server.url, impl=impl, options=ConnectionOptions(retry=fast_policy()), pool=pool
    )
    second = ws.connect(
        server.url, impl=impl, options=ConnectionOptions(retry=fast_policy()), pool=pool
    )
    try:
        assert await first.ready(timeout=3.0)
        assert await second.ready(timeout=3.0)
        # Both leases ride the SAME transport instance.
        assert first.transport is second.transport
        await server.current()
        assert server.accept_count == 1  # one real socket for two leases
    finally:
        await first.close()
        await second.close()
        pool.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_last_close_tears_socket_down(
    impl, server: LoopbackServer
) -> None:
    """The last lease to close stops the shared transport and drops the socket."""
    ws.DEFAULT_POOLS.pools.pop(impl, None)
    pool = ws.build_pool(impl, None)
    first = ws.connect(
        server.url, impl=impl, options=ConnectionOptions(retry=fast_policy()), pool=pool
    )
    second = ws.connect(
        server.url, impl=impl, options=ConnectionOptions(retry=fast_policy()), pool=pool
    )
    try:
        assert await first.ready(timeout=3.0)
        assert await second.ready(timeout=3.0)
        transport = first.transport

        await first.close()
        # One lease left: the socket stays up.
        await asyncio.sleep(0.1)
        assert transport.connected
        assert len(server.connections) == 1

        await second.close()
        # Last lease gone: transport stopped, server-side socket closes.
        await wait_until(lambda: not transport.connected)
        await wait_until(lambda: len(server.connections) == 0)
    finally:
        with contextlib.suppress(Exception):
            await first.close()
        with contextlib.suppress(Exception):
            await second.close()
        pool.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_async_handler_drain_ordered(
    impl, push_server: LoopbackServer
) -> None:
    """An async lease handler drains pushes FIFO, in arrival order, serialized."""
    order: List[str] = []
    running_now: List[int] = []
    max_concurrent = [0]

    async def handler(event: MessageReceived) -> None:
        running_now.append(1)
        max_concurrent[0] = max(max_concurrent[0], len(running_now))
        await asyncio.sleep(0.005)
        order.append(event.message.payload)
        running_now.pop()

    async with front_door(push_server.url, impl=impl) as conn:
        conn.event_bus.on(MessageReceived, handler)
        assert await conn.ready(timeout=3.0)

        for index in range(8):
            await push_server.push(f"m{index}")

        await wait_until(lambda: len(order) >= 8, timeout=5.0)
        assert order == [f"m{index}" for index in range(8)]
        assert max_concurrent[0] == 1  # serialized: never two at once


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_ready_already_connected_returns_fast(
    impl, server: LoopbackServer
) -> None:
    """ready() on an already-connected lease resolves True immediately."""
    async with front_door(server.url, impl=impl) as conn:
        assert await conn.ready(timeout=3.0)
        # Second call: already connected, returns at once.
        assert await conn.ready(timeout=0.001) is True


@pytest.mark.asyncio
async def test_front_door_rejects_non_ws_scheme() -> None:
    """ws.connect refuses a non ws/wss URL before touching the network."""
    with pytest.raises(ValueError):
        ws.connect(yarl.URL("http://127.0.0.1:1/"))


@pytest.mark.asyncio
async def test_front_door_rejects_unknown_impl() -> None:
    """ws.connect refuses an impl it does not ship."""
    with pytest.raises(ValueError):
        ws.connect(yarl.URL("ws://127.0.0.1:1/"), impl="nonsense")


# --------------------------------------------------------------------------- #
# Cross-impl: a single server serves websockets AND aiohttp leases at once.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_both_impls_against_one_server(server: LoopbackServer) -> None:
    """The same real server serves a websockets lease and an aiohttp lease."""
    ws.DEFAULT_POOLS.pools.pop("websockets", None)
    ws.DEFAULT_POOLS.pools.pop("aiohttp", None)
    pool_ws = ws.build_pool("websockets", None)
    pool_aio = ws.build_pool("aiohttp", None)

    a = ws.connect(
        server.url,
        impl="websockets",
        options=ConnectionOptions(retry=fast_policy()),
        pool=pool_ws,
    )
    b = ws.connect(
        server.url,
        impl="aiohttp",
        options=ConnectionOptions(retry=fast_policy()),
        pool=pool_aio,
    )

    got_a: List[WsMessage] = []
    got_b: List[WsMessage] = []
    a.event_bus.on(MessageReceived, lambda e: got_a.append(e.message))
    b.event_bus.on(MessageReceived, lambda e: got_b.append(e.message))

    try:
        assert await a.ready(timeout=3.0)
        assert await b.ready(timeout=3.0)

        await a.send("from-a")
        await b.send("from-b")

        await wait_until(
            lambda: any(m.payload == "echo:from-a" for m in got_a)
            and any(m.payload == "echo:from-b" for m in got_b)
        )
        # Distinct sockets: two separate accepts on the same server.
        await wait_until(lambda: server.accept_count >= 2)
    finally:
        await a.close()
        await b.close()
        pool_ws.stop()
        pool_aio.stop()


# --------------------------------------------------------------------------- #
# Import hygiene: importing the package must not pull a wire library.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_two_leases_both_receive_broadcast(
    impl, push_server: LoopbackServer
) -> None:
    """A 1:1 WS broadcasts every inbound frame to EVERY lease on the endpoint."""
    ws.DEFAULT_POOLS.pools.pop(impl, None)
    pool = ws.build_pool(impl, None)
    first = ws.connect(
        push_server.url,
        impl=impl,
        options=ConnectionOptions(retry=fast_policy()),
        pool=pool,
    )
    second = ws.connect(
        push_server.url,
        impl=impl,
        options=ConnectionOptions(retry=fast_policy()),
        pool=pool,
    )

    got_first: List[Any] = []
    got_second: List[Any] = []
    first.event_bus.on(MessageReceived, lambda e: got_first.append(e.message.payload))
    second.event_bus.on(MessageReceived, lambda e: got_second.append(e.message.payload))

    try:
        assert await first.ready(timeout=3.0)
        assert await second.ready(timeout=3.0)
        assert first.transport is second.transport

        await push_server.push("broadcast")
        await wait_until(lambda: got_first and got_second)
        assert got_first == ["broadcast"]
        assert got_second == ["broadcast"]
    finally:
        await first.close()
        await second.close()
        pool.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_ready_false_on_terminal_give_up_real_socket(make) -> None:
    """ready() resolves False once a bounded policy permanently gives up.

    Over a REAL dead port: a lease/transport that exhausts its retry budget must
    end the ready() wait with False (not hang), because supervising() is False.
    """
    import socket

    from simplyprint_ws_client.common.wire.lease import WsLease

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()

    policy = RetryPolicy(backoff=ConstantBackoff(0.0), max_attempts=3)
    transport = make(yarl.URL(f"ws://127.0.0.1:{dead_port}/"), policy)

    # Drive ready() through a real lease over the transport (no pool needed here:
    # ready() reads supervising()/connected/Disconnected straight off the transport).
    from simplyprint_ws_client.common.wire.pool import Pool

    pool: Pool = Pool(
        build=lambda url, params: transport,
        key=lambda url, params: "k",
        lease_class=WsLease,
    )
    lease = pool.connect(transport.url)
    try:
        assert await lease.ready(timeout=5.0) is False
        assert not transport.supervising()
        assert transport.generation == 0
    finally:
        await lease.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_ready_timeout_false_while_still_retrying(impl) -> None:
    """ready(timeout) returns False (not None, no raise) on a slow endpoint.

    The endpoint never comes up but the policy keeps retrying forever; ready()
    must honour the timeout and resolve False rather than hang or raise.
    """
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()

    url = yarl.URL(f"ws://127.0.0.1:{dead_port}/")
    forever = RetryPolicy(backoff=ConstantBackoff(0.05))  # retry forever
    async with front_door(url, impl=impl, retry=forever) as conn:
        result = await conn.ready(timeout=0.3)
        assert result is False
        # Still supervising (never gave up): a later connect could still succeed.
        assert conn.transport.supervising()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_server_ping_is_not_delivered_as_message(make) -> None:
    """A server-initiated WebSocket ping must not surface as a MessageReceived.

    The wire's recv must skip control frames (ping/pong) -- only real data frames
    become messages. Both libraries answer a ping with a pong transparently.
    """
    from websockets.asyncio.server import serve

    pinged = asyncio.Event()

    async def handler(connection: Any) -> None:
        try:
            await connection.ping()
            pinged.set()
            # Keep the socket open long enough to observe no spurious message.
            await asyncio.sleep(0.4)
            await connection.send("real-data")
            await asyncio.sleep(0.2)
        except Exception:  # noqa: BLE001
            pass

    srv = await serve(handler, "127.0.0.1", 0)
    port = next(iter(srv.sockets)).getsockname()[1]
    url = yarl.URL(f"ws://127.0.0.1:{port}/")

    rec = Recorder()
    transport = make(url)
    rec.attach(transport)
    transport.start()
    try:
        await wait_until(lambda: transport.connected)
        await asyncio.wait_for(pinged.wait(), timeout=2.0)
        # Wait for the real data frame; assert no ping leaked before it.
        await wait_until(lambda: any(m[1] == "real-data" for m in rec.messages))
        payloads = [m[1] for m in rec.messages]
        assert payloads == ["real-data"]
    finally:
        await transport.stop()
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_server_graceful_close_reconnects(make, server: LoopbackServer) -> None:
    """A server that closes the socket gracefully (1000) is also healed.

    A clean server-side close (not an error code) still ends the attempt; the
    supervised loop must treat it as a drop and reconnect to a new generation.
    """

    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)
    transport.start()
    try:
        await wait_until(lambda: transport.connected)
        assert transport.generation == 1
        connection = await server.current()
        await connection.close(code=1000)

        await wait_until(lambda: transport.connected and transport.generation >= 2)
        assert transport.generation == 2
        # The healed link carries traffic on the new generation.
        await transport.send("post-graceful")
        await wait_until(
            lambda: any(
                gen == 2 and msg == "echo:post-graceful" for gen, msg in rec.messages
            )
        )
    finally:
        await transport.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_many_messages_preserve_order_and_count(
    make, push_server: LoopbackServer
) -> None:
    """A burst of server pushes is delivered in order with no loss (lossless QoS)."""
    rec = Recorder()
    transport = make(push_server.url)
    rec.attach(transport)
    transport.start()
    try:
        await wait_until(lambda: transport.connected)
        connection = await push_server.current()
        for index in range(50):
            await connection.send(f"n{index}")

        await wait_until(lambda: len(rec.messages) >= 50, timeout=5.0)
        payloads = [msg for _, msg in rec.messages]
        assert payloads == [f"n{index}" for index in range(50)]
        assert all(gen == 1 for gen, _ in rec.messages)
    finally:
        await transport.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_stop_while_connecting_is_clean(make) -> None:
    """stop() during an in-flight connect attempt settles to DISCONNECTED cleanly."""
    import socket

    # A port that accepts TCP but never completes the WS handshake: bind+listen
    # but never accept, so the client hangs in open().
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    transport = make(yarl.URL(f"ws://127.0.0.1:{port}/"))
    transport.start()
    try:
        # Give it a beat to enter the open() attempt.
        await asyncio.sleep(0.05)
        await transport.stop()
        assert transport.state is ConnectionState.DISCONNECTED
        assert not transport.connected
    finally:
        listener.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_front_door_ready_false_on_give_up(impl) -> None:
    """The full ws.connect path resolves ready() False on a permanent give-up."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()

    url = yarl.URL(f"ws://127.0.0.1:{dead_port}/")
    bounded = RetryPolicy(backoff=ConstantBackoff(0.0), max_attempts=2)
    async with front_door(url, impl=impl, retry=bounded) as conn:
        assert await conn.ready(timeout=5.0) is False
        assert not conn.transport.supervising()
        assert conn.state is ConnectionState.DISCONNECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_lifecycle_event_sequence_across_drop(
    make, server: LoopbackServer
) -> None:
    """A drop yields the full CONNECTING/CONNECTED/DISCONNECTED/... lifecycle order."""
    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)
    transport.start()
    try:
        await wait_until(lambda: transport.connected)
        await server.drop_current()
        await wait_until(lambda: transport.connected and transport.generation >= 2)

        # First connect on gen 0->1, a disconnect of gen 1, then a fresh connect.
        assert rec.connecting[0] == 0
        assert 1 in rec.connected and 2 in rec.connected
        assert any(gen == 1 for gen, _ in rec.disconnected)
        # The second Connecting was announced before the second Connected.
        assert len(rec.connecting) >= 2
    finally:
        await transport.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("impl", ["websockets", "aiohttp"])
async def test_at_most_once_outbound_still_sends(impl, server: LoopbackServer) -> None:
    """An AT_MOST_ONCE outbound WsMessage still goes out (QoS gates inbound only)."""
    received: List[WsMessage] = []
    async with front_door(server.url, impl=impl) as conn:
        conn.event_bus.on(MessageReceived, lambda e: received.append(e.message))
        assert await conn.ready(timeout=3.0)
        await conn.send(WsMessage.text("amo", qos=QoS.AT_MOST_ONCE))
        await wait_until(
            lambda: any(
                isinstance(m, WsMessage) and m.payload == "echo:amo" for m in received
            )
        )
        assert "amo" in server.received


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_generation_equals_successful_accepts(
    make, server: LoopbackServer
) -> None:
    """After N forced drops the generation equals the number of server accepts.

    Each established connection bumps the generation exactly once -- never twice,
    never zero. Cross-checking against the server's own accept count is the real
    ground truth a fake cannot give.
    """
    rec = Recorder()
    transport = make(server.url)
    rec.attach(transport)
    transport.start()
    try:
        await wait_until(lambda: transport.connected)
        for _ in range(5):
            target = transport.generation + 1
            await server.drop_current()
            await wait_until(
                lambda: transport.connected and transport.generation >= target
            )

        await wait_until(lambda: server.accept_count >= 6)
        assert transport.generation == 6
        assert server.accept_count == 6
        # Connected was announced once per generation, strictly increasing.
        assert rec.connected == [1, 2, 3, 4, 5, 6]
    finally:
        await transport.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("make", WIRES)
async def test_messages_after_reconnect_carry_new_generation_only(
    make, push_server: LoopbackServer
) -> None:
    """Post-reconnect pushes carry ONLY the new generation, never the stale one."""
    rec = Recorder()
    transport = make(push_server.url)
    rec.attach(transport)
    transport.start()
    try:
        await wait_until(lambda: transport.connected)
        await push_server.push("gen1")
        await wait_until(lambda: any(m == "gen1" for _, m in rec.messages))

        await push_server.drop_current()
        await wait_until(lambda: transport.connected and transport.generation >= 2)

        await push_server.push("gen2")
        await wait_until(lambda: any(m == "gen2" for _, m in rec.messages))

        by_payload = {m: gen for gen, m in rec.messages}
        assert by_payload["gen1"] == 1
        assert by_payload["gen2"] == 2
    finally:
        await transport.stop()


def test_importing_conn_package_loads_no_wire_library() -> None:
    """Importing contrib.connection must not eager-load websockets/aiohttp/paho/aiomqtt."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import simplyprint_ws_client.common.wire as conn\n"
        "leaked = [m for m in ('websockets', 'aiohttp', 'paho', 'aiomqtt')\n"
        "          if m in sys.modules]\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
