"""Brand-free tests for the new connection library (``contrib.connection``).

Every wire is faked and injected -- no real broker, socket, or wire library is
touched. The fakes satisfy the same seams production wires do:

* a tiny :class:`FakeTransport` for pool refcounting and lease delivery;
* a :class:`Reconnecting` subclass driven by a queue for the supervised
  reconnect loop;
* a fake aiomqtt client (``.messages`` / ``.subscribe`` / ``.publish``) injected
  into :class:`AioMqtt` for MQTT multi-topic routing;
* a fake ``websockets`` socket injected into :class:`Websockets` for WS broadcast.

These pin the library contract: one shared transport per endpoint, the
connect/drop/reconnect state machine, lease delivery through EventBus, MQTT vs WS
routing, and the two front doors.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import pytest
import yarl

from simplyprint_ws_client.common.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection import (
    Connection,
    MqttConnection,
    WsConnection,
)
from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    Connecting,
    ConnectionEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.contrib.connection.keepalive import (
    Keepalive,
    KeepaliveTimeout,
)
from simplyprint_ws_client.contrib.connection.messages import QoS
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.pool import Pool
from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    MqttTransport,
    NotConnected,
    TransientError,
    Transport,
)

from simplyprint_ws_client.contrib.connection import mqtt
from simplyprint_ws_client.contrib.connection import websocket as ws
from simplyprint_ws_client.contrib.connection.aiomqtt import AioMqtt
from simplyprint_ws_client.contrib.connection.mqtt import (
    MqttBroker,
    MqttMessage,
    mqtt_message_route,
)
from simplyprint_ws_client.contrib.connection.websocket import WsMessage
from simplyprint_ws_client.contrib.connection.websockets import Websockets
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def wait_for(predicate, timeout: float = 2.0) -> None:
    """Poll ``predicate`` on the loop until true, or raise after ``timeout``."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


def current_provider() -> EventLoopProvider:
    """A provider bound to the running test loop."""
    return EventLoopProvider(loop=asyncio.get_event_loop())


# --------------------------------------------------------------------------- #
# A minimal fake Transport for pool + lease tests (no wire, no loop work).
# --------------------------------------------------------------------------- #


class FakeTransport(Transport):
    """A do-nothing transport that records start/stop and lets a test emit.

    It satisfies the whole :class:`Transport` contract with no I/O: ``start`` /
    ``stop`` only flip a flag and bump counters, and a test can drive lifecycle
    by calling :meth:`go_up` / :meth:`go_down` (which emit on the bus exactly as
    a real transport would).
    """

    def __init__(self, url: yarl.URL) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[ConnectionEvent] = EventBus()
        self.starts = 0
        self.stops = 0
        self.sent: List[object] = []
        self.live = False
        self.supervises = True

    @property
    def connected(self) -> bool:
        return self.live

    def start(self) -> None:
        self.starts += 1
        self.state = ConnectionState.CONNECTING

    async def stop(self) -> None:
        self.stops += 1
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: object) -> None:
        if not self.live:
            raise NotConnected("fake transport not connected")
        self.sent.append(message)

    def supervising(self) -> bool:
        return self.supervises

    async def go_up(self) -> None:
        self.generation += 1
        self.live = True
        self.state = ConnectionState.CONNECTED
        await self.events.emit(Connected(self.generation))

    async def go_down(self, *, terminal: bool = False, code: object = None) -> None:
        self.live = False
        self.state = ConnectionState.DISCONNECTED
        if terminal:
            self.supervises = False
        await self.events.emit(Disconnected(self.generation, code=code))

    async def deliver(self, message: object) -> None:
        await self.events.emit(MessageReceived(self.generation, message))


class FakeRoutedTransport(FakeTransport, MqttTransport):
    """A fake broker transport that routes a message by an attribute ``topic``."""

    async def subscribe(self, topic: str) -> None:  # pragma: no cover - unused here
        pass

    async def unsubscribe(self, topic: str) -> None:  # pragma: no cover - unused
        pass


def build_pool(transports: List[FakeTransport], *, routed: bool = False) -> Pool:
    """A pool keyed by URL string that records the transports it builds."""
    cls = FakeRoutedTransport if routed else FakeTransport

    def build(url: yarl.URL, params: object) -> FakeTransport:
        transport = cls(url)
        transports.append(transport)
        return transport

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        route=(lambda message: message.topic) if routed else None,
        provider=current_provider(),
    )


# --------------------------------------------------------------------------- #
# Pool: one transport per endpoint, refcounted, torn down on last close.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pool_shares_one_transport_per_endpoint():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    url = yarl.URL("ws://host/path")

    a = pool.connect(url)
    b = pool.connect(url)

    assert len(transports) == 1  # one wire built for the endpoint
    assert a.transport is b.transport  # both leases share it
    assert transports[0].starts == 1  # started once, by the first lease
    assert a is not b


@pytest.mark.asyncio
async def test_pool_distinct_endpoints_get_distinct_transports():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)

    a = pool.connect(yarl.URL("ws://host/one"))
    b = pool.connect(yarl.URL("ws://host/two"))

    assert len(transports) == 2
    assert a.transport is not b.transport


@pytest.mark.asyncio
async def test_pool_refcounts_and_tears_down_on_last_close():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    url = yarl.URL("ws://host/path")

    a = pool.connect(url)
    b = pool.connect(url)
    transport = transports[0]

    await a.close()
    assert transport.stops == 0  # b still holds the lease
    assert str(url) in pool.endpoints  # endpoint still present

    await b.close()
    assert transport.stops == 1  # last lease stopped the wire
    assert str(url) not in pool.endpoints  # endpoint dropped

    # A fresh connect after teardown builds a brand-new transport.
    c = pool.connect(url)
    assert len(transports) == 2
    assert c.transport is not transport
    await c.close()


@pytest.mark.asyncio
async def test_lease_keepalive_probes_and_times_out_when_idle():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://host/path"))
    probes: List[Connection] = []
    disconnected: List[object] = []

    lease.event_bus.on(Disconnected, lambda event: disconnected.append(event.code))
    lease.keepalive(
        Keepalive(interval=0.01, max_misses=1, probe=lambda conn: probes.append(conn))
    )

    await transports[0].go_up()

    await wait_for(lambda: probes)
    await wait_for(lambda: disconnected)
    assert probes == [lease]
    assert isinstance(disconnected[-1], KeepaliveTimeout)

    await lease.close()


# --------------------------------------------------------------------------- #
# Reconnecting engine: lifecycle events, generation, clean cancel.
# --------------------------------------------------------------------------- #


class DrivableWire(Reconnecting):
    """A :class:`Reconnecting` whose hooks are driven by a per-attempt inbox.

    ``open`` succeeds (or fails the first N times); ``recv`` blocks on an inbox a
    test feeds -- a queued ``BaseException`` simulates a wire drop. This exercises
    the supervised loop with no real socket.
    """

    def __init__(self, url: yarl.URL, *, fail_opens: int = 0, **kw) -> None:
        super().__init__(url, **kw)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.opens = 0
        self.closes = 0
        self.fail_opens = fail_opens
        self.sent: List[object] = []

    async def open(self) -> None:
        self.opens += 1
        if self.opens <= self.fail_opens:
            raise TransientError("open failed")

    async def recv(self) -> Optional[object]:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def write(self, message: object) -> None:
        self.sent.append(message)

    async def aclose(self) -> None:
        self.closes += 1


@pytest.mark.asyncio
async def test_reconnecting_emits_connecting_connected_message():
    seen: List[Any] = []
    wire = DrivableWire(
        yarl.URL("ws://x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
    )
    wire.events.on(Connecting, lambda e: seen.append(("connecting", e.generation)))
    wire.events.on(Connected, lambda e: seen.append(("connected", e.generation)))
    wire.events.on(MessageReceived, lambda e: seen.append(("msg", e.message)))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)
        wire.inbox.put_nowait("hello")
        await wait_for(lambda: ("msg", "hello") in seen)

        assert ("connecting", 0) in seen  # first reach is on generation 0
        assert ("connected", 1) in seen  # bumped exactly once on a live wire
        assert wire.generation == 1
        assert wire.state is ConnectionState.CONNECTED
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_reconnecting_drop_emits_disconnected_and_reconnects_once():
    downs: List[Any] = []
    ups: List[int] = []
    wire = DrivableWire(
        yarl.URL("ws://x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
    )
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.events.on(Disconnected, lambda e: downs.append((e.generation, e.code)))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)
        drop = RuntimeError("boom")
        wire.inbox.put_nowait(drop)  # ends attempt 1

        # The drop announces Disconnected tagged with the failure, then a fresh
        # attempt brings the wire back on the next generation.
        await wait_for(lambda: len(downs) == 1)
        assert downs[0][0] == 1  # generation of the dropped link
        assert isinstance(downs[0][1], TransientError)
        assert isinstance(downs[0][1].transport_error, RuntimeError)

        await wait_for(lambda: wire.generation == 2 and wire.connected)
        assert ups == [1, 2]  # generation bumped exactly once per attempt
        assert wire.opens == 2  # a fresh open per attempt
        assert wire.closes >= 1  # the dropped wire was torn down
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_reconnecting_send_raises_not_connected_when_down():
    wire = DrivableWire(
        yarl.URL("ws://x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
    )
    # Never started: no live wire.
    with pytest.raises(NotConnected):
        await wire.send("nope")

    wire.start()
    try:
        await wait_for(lambda: wire.connected)
        await wire.send("ok")
        assert wire.sent == ["ok"]
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_reconnecting_stop_is_clean_no_leaked_tasks():
    before = {t for t in asyncio.all_tasks() if not t.done()}
    wire = DrivableWire(
        yarl.URL("ws://x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
    )
    wire.start()
    await wait_for(lambda: wire.connected)
    opens_at_stop = wire.opens

    await wire.stop()
    # Give the loop a turn; no new attempts after stop, and the task is gone.
    await asyncio.sleep(0.03)
    assert wire.opens == opens_at_stop
    assert wire.task is None
    assert not wire.connected
    assert not wire.supervising()  # stopped is not supervising

    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


@pytest.mark.asyncio
async def test_reconnecting_gives_up_when_policy_exhausted():
    downs: List[Any] = []
    wire = DrivableWire(
        yarl.URL("ws://x"),
        fail_opens=100,  # never connects
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=3),
        provider=current_provider(),
    )
    wire.events.on(Disconnected, lambda e: downs.append(e.generation))
    wire.start()
    try:
        await wait_for(lambda: not wire.supervising())
        assert wire.gave_up is True
        assert wire.state is ConnectionState.DISCONNECTED
        assert len(downs) == 3  # one Disconnected per failed attempt, then stop
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# Connection lease: inbound delivery, ready(), send(), ready_transport().
# --------------------------------------------------------------------------- #


def lease_on(transport: FakeTransport, lease_cls=Connection) -> Connection:
    """A standalone lease over ``transport`` with a real (but ignored) pool.

    The lease only delegates ``close`` to the pool; for delivery/ready/send tests we
    emit on its EventBus directly, the same seam the pool uses after routing.
    """
    pool = Pool(
        build=lambda url, params: transport,
        key=lambda url, params: "k",
        provider=current_provider(),
    )
    return lease_cls(
        pool,
        transport,
        transport.url,
        "k",
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_lease_event_bus_delivers_inbound():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)
    got: List[object] = []
    lease.event_bus.on(MessageReceived, lambda e: got.append(e.message))

    await lease.event_bus.emit(MessageReceived(1, "frame-1"))
    await lease.event_bus.emit(MessageReceived(1, "frame-2"))

    assert got == ["frame-1", "frame-2"]


@pytest.mark.asyncio
async def test_lease_ready_true_on_connect():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)

    waiter = asyncio.ensure_future(lease.ready(timeout=1.0))
    await asyncio.sleep(0)  # let ready() subscribe
    transport.generation = 1
    transport.live = True
    await lease.event_bus.emit(Connected(1))

    assert await waiter is True


@pytest.mark.asyncio
async def test_lease_ready_false_on_timeout():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)
    # Never connects.
    assert await lease.ready(timeout=0.05) is False


@pytest.mark.asyncio
async def test_lease_ready_false_on_terminal_giveup():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)

    waiter = asyncio.ensure_future(lease.ready(timeout=1.0))
    await asyncio.sleep(0)
    transport.supervises = False
    await lease.event_bus.emit(Disconnected(0, code=None))

    assert await waiter is False


@pytest.mark.asyncio
async def test_lease_send_raises_not_connected_when_down():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)
    with pytest.raises(NotConnected):
        await lease.send("payload")

    await transport.go_up()
    await lease.send("payload")
    assert transport.sent == ["payload"]


@pytest.mark.asyncio
async def test_lease_transport_awaits_live_connection():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)

    waiter = asyncio.ensure_future(lease.ready_transport())
    await asyncio.sleep(0)
    assert not waiter.done()  # blocks while down
    transport.generation = 1
    transport.live = True
    await lease.event_bus.emit(Connected(1))

    resolved = await waiter
    assert resolved is transport


# --------------------------------------------------------------------------- #
# MQTT routing: multi-topic on one conn; per-broker scoping; via a fake client.
# --------------------------------------------------------------------------- #


class FakeAioMqttClient:
    """An aiomqtt-shaped fake: ``async with`` once, ``.messages`` is an async
    iterator a test feeds, plus ``.subscribe`` / ``.unsubscribe`` / ``.publish``."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.subscribed: List[str] = []
        self.unsubscribed: List[str] = []
        self.published: List[tuple] = []
        self.messages = self  # ``client.messages.__aiter__()``

    async def __aenter__(self) -> "FakeAioMqttClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def __aiter__(self) -> "FakeAioMqttClient":
        return self

    async def __anext__(self) -> Any:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def subscribe(self, topic: str) -> None:
        self.subscribed.append(topic)

    async def unsubscribe(self, topic: str) -> None:
        self.unsubscribed.append(topic)

    async def publish(self, topic, payload, qos=0, retain=False) -> None:
        self.published.append((topic, payload, qos, retain))


class WireMqttMessage:
    """An aiomqtt message stand-in -- ``AioMqtt`` forwards these wire-shaped and
    routes them by ``.topic``."""

    def __init__(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False):
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.retain = retain


def build_mqtt_pool(clients: List[FakeAioMqttClient]) -> Pool:
    """A pool of :class:`AioMqtt` wired to fresh fake clients, keyed by broker
    endpoint, handing out :class:`MqttConnection` leases."""

    def factory(url, logger):
        client = FakeAioMqttClient()
        clients.append(client)
        return client

    def build(url: yarl.URL, params: object):
        return AioMqtt(
            url,
            RetryPolicy(backoff=ConstantBackoff(0)),
            current_provider(),
            client_factory=factory,
        )

    def key(url: yarl.URL, params: object):
        return params if isinstance(params, MqttBroker) else MqttBroker.from_url(url)

    return Pool(
        build=build,
        key=key,
        route=mqtt_message_route,
        lease_class=MqttConnection,
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_mqtt_multi_topic_on_one_connection():
    clients: List[FakeAioMqttClient] = []
    pool = build_mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")

    lease = pool.connect(url, MqttBroker.from_url(url))
    got: List[str] = []
    lease.event_bus.on(MessageReceived, lambda e: got.append(e.message.topic))

    await lease.subscribe("a")
    await lease.subscribe("b")
    await wait_for(lambda: lease.connected)
    assert len(clients) == 1  # one shared socket
    client = clients[0]

    client.inbox.put_nowait(WireMqttMessage("a", b"1"))
    client.inbox.put_nowait(WireMqttMessage("b", b"2"))
    client.inbox.put_nowait(WireMqttMessage("c", b"3"))  # not subscribed

    await wait_for(lambda: len(got) >= 2)
    await asyncio.sleep(0.02)  # give a stray 'c' a chance to (wrongly) arrive
    assert sorted(got) == ["a", "b"]  # c is filtered out by the lease's topics

    await lease.close()


@pytest.mark.asyncio
async def test_mqtt_two_leases_each_get_only_their_topic():
    clients: List[FakeAioMqttClient] = []
    pool = build_mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    first = pool.connect(url, broker)
    second = pool.connect(url, broker)
    assert first.transport is second.transport  # one shared broker socket

    got_first: List[str] = []
    got_second: List[str] = []
    first.event_bus.on(MessageReceived, lambda e: got_first.append(e.message.topic))
    second.event_bus.on(MessageReceived, lambda e: got_second.append(e.message.topic))

    await first.subscribe("alpha")
    await second.subscribe("beta")
    await wait_for(lambda: first.connected)
    assert len(clients) == 1  # one aiomqtt client built for the shared socket

    client = clients[0]
    client.inbox.put_nowait(WireMqttMessage("alpha", b"a"))
    client.inbox.put_nowait(WireMqttMessage("beta", b"b"))

    await wait_for(lambda: got_first and got_second)
    await asyncio.sleep(0.02)
    assert got_first == ["alpha"]  # each lease sees only its own topic
    assert got_second == ["beta"]

    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_mqtt_wildcard_subscription_routes():
    clients: List[FakeAioMqttClient] = []
    pool = build_mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    lease = pool.connect(url, MqttBroker.from_url(url))
    got: List[str] = []
    lease.event_bus.on(MessageReceived, lambda e: got.append(e.message.topic))

    await lease.subscribe("device/#")
    await wait_for(lambda: lease.connected)
    client = clients[0]
    client.inbox.put_nowait(WireMqttMessage("device/temp", b"1"))
    client.inbox.put_nowait(WireMqttMessage("device", b"2"))
    client.inbox.put_nowait(WireMqttMessage("other", b"3"))

    await wait_for(lambda: len(got) >= 2)
    await asyncio.sleep(0.02)
    assert sorted(got) == ["device", "device/temp"]  # 'other' filtered out

    await lease.close()


# --------------------------------------------------------------------------- #
# WebSocket routing: 1:1 broadcast -- every lease gets every message.
# --------------------------------------------------------------------------- #


class FakeWsSocket:
    """A ``websockets``-shaped fake socket: ``recv`` blocks on an inbox a test
    feeds raw ``str``/``bytes`` frames into, ``send`` records the raw frame, and
    ``close`` is counted. The :class:`Websockets` wire holds one of these as its
    plain ``socket`` attribute and calls these three coroutines directly."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: List[object] = []
        self.closed = 0

    async def recv(self) -> object:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, frame: object) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        self.closed += 1


def build_ws_pool(sockets: List[FakeWsSocket]) -> Pool:
    async def connect_factory(url: str, **kwargs) -> FakeWsSocket:
        socket = FakeWsSocket()
        sockets.append(socket)
        return socket

    def build(url: yarl.URL, params: object) -> Websockets:
        return Websockets(
            url,
            RetryPolicy(backoff=ConstantBackoff(0)),
            current_provider(),
            connect_factory=connect_factory,
        )

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        lease_class=WsConnection,
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_ws_broadcast_every_lease_gets_every_message():
    sockets: List[FakeWsSocket] = []
    pool = build_ws_pool(sockets)
    url = yarl.URL("ws://host/path")

    one = pool.connect(url)
    two = pool.connect(url)
    assert one.transport is two.transport  # one shared socket
    assert len(sockets) == 0  # built lazily on the first open

    got_one: List[str] = []
    got_two: List[str] = []
    one.event_bus.on(MessageReceived, lambda e: got_one.append(e.message.payload))
    two.event_bus.on(MessageReceived, lambda e: got_two.append(e.message.payload))

    await wait_for(lambda: one.connected)
    assert len(sockets) == 1
    socket = sockets[0]
    socket.inbox.put_nowait("broadcast")  # a raw text frame off the wire

    await wait_for(lambda: got_one and got_two)
    assert got_one == ["broadcast"]  # both leases get the same frame
    assert got_two == ["broadcast"]

    await one.close()
    await two.close()


@pytest.mark.asyncio
async def test_ws_send_routes_to_the_wire():
    sockets: List[FakeWsSocket] = []
    pool = build_ws_pool(sockets)
    lease = pool.connect(yarl.URL("ws://host/path"))
    await wait_for(lambda: lease.connected)

    await lease.send("hi")  # bare str -> text frame
    await lease.send(b"bytes")  # bare bytes -> binary frame
    await lease.send(WsMessage.binary(b"explicit"))

    # The front-door lease reduces a WsMessage to the raw str/bytes the wire sends.
    socket = sockets[0]
    assert socket.sent[0] == "hi"  # text frame
    assert socket.sent[1] == b"bytes"  # binary frame
    assert socket.sent[2] == b"explicit"  # explicit WsMessage -> its bytes

    await lease.close()


@pytest.mark.asyncio
async def test_lease_event_bus_awaits_async_message_handlers():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)
    handled: List[bytes] = []

    async def slow(event: MessageReceived) -> None:
        await asyncio.sleep(0)
        handled.append(event.message.payload)

    lease.event_bus.on(MessageReceived, slow)

    await lease.event_bus.emit(
        MessageReceived(1, MqttMessage("t", b"payload", qos=QoS.AT_MOST_ONCE))
    )

    assert handled == [b"payload"]


@pytest.mark.asyncio
async def test_lease_event_bus_preserves_lifecycle_order():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)
    seen: List[type] = []

    async def slow(event: ConnectionEvent) -> None:
        await asyncio.sleep(0)
        seen.append(type(event))

    lease.event_bus.on(Connected, slow)
    lease.event_bus.on(Disconnected, slow)

    for _ in range(10):
        await lease.event_bus.emit(Connected(1))
        await lease.event_bus.emit(Disconnected(1, code=None))

    assert seen == [Connected, Disconnected] * 10


@pytest.mark.asyncio
async def test_lease_event_bus_does_not_drop_message_bursts():
    transport = FakeTransport(yarl.URL("ws://x"))
    lease = lease_on(transport)
    handled: List[bytes] = []

    async def slow(event: MessageReceived) -> None:
        await asyncio.sleep(0)
        handled.append(event.message.payload)

    lease.event_bus.on(MessageReceived, slow)

    for i in range(20):
        await lease.event_bus.emit(
            MessageReceived(1, MqttMessage("t", bytes([i]), qos=QoS.AT_LEAST_ONCE))
        )

    assert handled == [bytes([i]) for i in range(20)]


# --------------------------------------------------------------------------- #
# Front doors: mqtt.connect / ws.connect return the right type, apply ?topic=.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mqtt_connect_returns_lease_and_applies_url_topics():
    clients: List[FakeAioMqttClient] = []
    pool = build_mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/?topic=foo/%23")  # foo/# url-encoded

    lease = mqtt.connect(url, pool=pool)
    assert isinstance(lease, MqttConnection)
    assert isinstance(lease, MqttConnection)

    # The URL's ?topic= is recorded on the lease synchronously for routing,
    # and asserted on the shared socket once it comes up.
    assert "foo/#" in lease.topics
    await wait_for(lambda: lease.connected)
    await wait_for(lambda: "foo/#" in clients[0].subscribed)

    got: List[str] = []
    lease.event_bus.on(MessageReceived, lambda e: got.append(e.message.topic))
    clients[0].inbox.put_nowait(WireMqttMessage("foo/bar", b"x"))
    clients[0].inbox.put_nowait(WireMqttMessage("nope", b"y"))
    await wait_for(lambda: got == ["foo/bar"])  # wildcard routes, 'nope' filtered

    await lease.close()


@pytest.mark.asyncio
async def test_mqtt_connect_bare_bytes_uses_default_topic():
    clients: List[FakeAioMqttClient] = []
    pool = build_mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/?topic=cmd")

    lease = mqtt.connect(url, pool=pool)
    await wait_for(lambda: lease.connected)

    await lease.send(b"payload")  # bare bytes -> wrapped onto the URL's first topic
    await wait_for(lambda: clients[0].published)
    topic, payload, _qos, _retain = clients[0].published[0]
    assert topic == "cmd"
    assert payload == b"payload"

    await lease.close()


@pytest.mark.asyncio
async def test_mqtt_connect_rejects_url_without_host():
    with pytest.raises(ValueError):
        mqtt.connect(yarl.URL("mqtt:///?topic=x"))


@pytest.mark.asyncio
async def test_ws_connect_returns_ws_lease():
    sockets: List[FakeWsSocket] = []
    pool = build_ws_pool(sockets)
    url = yarl.URL("wss://host/socket")

    lease = ws.connect(url, pool=pool)
    assert isinstance(lease, WsConnection)
    assert not isinstance(lease, MqttConnection)

    await wait_for(lambda: lease.connected)
    got: List[str] = []
    lease.event_bus.on(MessageReceived, lambda e: got.append(e.message.payload))
    sockets[0].inbox.put_nowait("hello")  # a raw text frame off the wire
    await wait_for(lambda: got == ["hello"])

    await lease.close()


@pytest.mark.asyncio
async def test_ws_connect_rejects_non_ws_scheme():
    with pytest.raises(ValueError):
        ws.connect(yarl.URL("http://host/socket"))


@pytest.mark.asyncio
async def test_front_doors_share_one_transport_per_endpoint():
    # ws.connect twice on the same URL shares one socket (front-door pooling).
    sockets: List[FakeWsSocket] = []
    pool = build_ws_pool(sockets)
    url = yarl.URL("wss://host/socket")
    a = ws.connect(url, pool=pool)
    b = ws.connect(url, pool=pool)
    assert a.transport is b.transport
    await a.close()
    await b.close()
