"""Proof that the transport contract spans the *async* family, not just paho.

``AsyncMqttTransport`` (aiomqtt) is wrapped as an :class:`AsyncTransport` and
publishes the exact same :class:`TransportEvent` surface a synchronous paho
:class:`Transport` would; ``AsyncMqttPool`` shows pooling-as-a-capability for
it. Both are exercised with injected fakes -- no broker, and aiomqtt need not be
installed -- so this pins the abstraction, not a live integration.
"""

import asyncio
import subprocess
import sys

import pytest

from simplyprint_ws_client.contrib.connection.mqtt import (
    AsyncMqttPool,
    AsyncMqttTransport,
    MqttParams,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    AsyncTransport,
    Connected,
    Disconnected,
    MessageReceived,
    StateChanged,
)
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.utils.backoff import ConstantBackoff


def test_importing_async_mqtt_does_not_load_aiomqtt():
    # The default factory imports aiomqtt lazily, so importing the module (and
    # building a transport) must not drag the optional dependency in.
    code = (
        "import sys\n"
        "import simplyprint_ws_client.contrib.connection.mqtt.aio as m\n"
        "assert 'aiomqtt' not in sys.modules, 'aiomqtt eagerly imported'\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_async_mqtt_satisfies_the_contract():
    assert issubclass(AsyncMqttTransport, AsyncTransport)
    # Every abstract hook is implemented -- it is a usable transport.
    assert AsyncMqttTransport.__abstractmethods__ == frozenset()


# --------------------------------------------------------------------------- #
# Fake aiomqtt client: an async context manager exposing .messages (an async
# iterator), .subscribe and .publish -- the slice the transport actually uses.
# --------------------------------------------------------------------------- #


class _StreamingClient:
    """Yields a fixed list of messages, then (optionally) drops the link."""

    def __init__(self, inbound, *, drop: bool = True):
        self._inbound = list(inbound)
        self._drop = drop
        self.subscribed = []
        self.published = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def subscribe(self, topic):
        self.subscribed.append(topic)

    async def publish(self, topic, payload):
        self.published.append((topic, payload))

    @property
    def messages(self):
        return self._gen()

    async def _gen(self):
        for message in self._inbound:
            yield message
        if self._drop:
            raise RuntimeError("broker dropped the connection")


class _HoldingClient:
    """Stays 'connected' (blocks in .messages) until ``release`` is set."""

    def __init__(self, release: asyncio.Event):
        self._release = release
        self.subscribed = []
        self.published = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def subscribe(self, topic):
        self.subscribed.append(topic)

    async def publish(self, topic, payload):
        self.published.append((topic, payload))

    @property
    def messages(self):
        return self._gen()

    async def _gen(self):
        await self._release.wait()
        return
        yield  # marks _gen as an async generator


@pytest.mark.asyncio
async def test_transport_emits_neutral_events_and_supervises_reconnect():
    seen = []
    made = []

    def client_factory(params, logger):
        client = _StreamingClient(["m1", "m2"], drop=True)
        made.append(client)
        return client

    transport = AsyncMqttTransport(
        MqttParams("broker", 8883, "user", "pw"),
        client_factory=client_factory,
        backoff=ConstantBackoff(0),
    )
    transport.events.on(Connected, lambda e: seen.append("connected"))
    transport.events.on(MessageReceived, lambda e: seen.append(("msg", e.payload)))
    transport.events.on(Disconnected, lambda e: seen.append(("disc", e.transient)))
    transport.events.on(StateChanged, lambda e: seen.append(("state", e.state)))

    transport.subscribe("printer/+/report")
    transport.start()

    # Let it connect, stream both messages, drop, and reconnect at least once.
    for _ in range(200):
        await asyncio.sleep(0.005)
        if len(made) >= 2:
            break
    transport.stop()
    await asyncio.sleep(0.02)

    assert "connected" in seen
    assert ("msg", "m1") in seen and ("msg", "m2") in seen
    assert ("state", ConnectionState.ONLINE) in seen
    assert ("disc", True) in seen  # the drop is surfaced as a transient disconnect
    assert len(made) >= 2  # supervised: it rebuilt the client and reconnected
    assert made[0].subscribed == ["printer/+/report"]  # topic re-asserted on connect


@pytest.mark.asyncio
async def test_transport_send_publishes_to_the_live_client():
    release = asyncio.Event()
    client = _HoldingClient(release)
    connected = asyncio.Event()

    transport = AsyncMqttTransport(
        MqttParams("broker", 8883),
        client_factory=lambda params, logger: client,
        backoff=ConstantBackoff(0),
    )
    transport.events.on(Connected, lambda e: connected.set())
    transport.start()
    try:
        await asyncio.wait_for(connected.wait(), timeout=1.0)
        await transport.send(("cmd/topic", b"payload"))
        assert client.published == [("cmd/topic", b"payload")]
    finally:
        release.set()
        transport.stop()
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_pool_shares_one_transport_per_endpoint_and_refcounts():
    created = []

    def transport_factory(params):
        fake = _RecordingTransport(params)
        created.append(fake)
        return fake

    pool = AsyncMqttPool(transport_factory=transport_factory)
    a = MqttParams("a", 8883)
    b = MqttParams("b", 8883)

    c_a1 = await pool.connect(a)
    c_a2 = await pool.connect(a)  # same endpoint -> same transport, not a new one
    c_b = await pool.connect(b)
    t_a1 = created[0]
    t_b = created[1]

    assert t_a1 is not t_b
    assert len(created) == 2
    assert t_a1.started == 1  # started once despite two leases
    assert t_b.started == 1

    await c_a1.close()  # one holder remains
    assert t_a1.stopped == 0
    await c_a2.close()  # last holder leaves -> torn down
    assert t_a1.stopped == 1

    c_a3 = await pool.connect(a)  # re-acquired -> a fresh transport
    t_a3 = created[2]
    assert t_a3 is not t_a1
    assert t_a3.started == 1
    await c_a3.close()
    await c_b.close()


class _RecordingTransport:
    """Records start/stop so the pool's lifecycle can be asserted."""

    def __init__(self, params):
        self.params = params
        self.events = EventBus()
        self.state = ConnectionState.OFFLINE
        self.started = 0
        self.stopped = 0
        self._connected = False

    @property
    def connected(self):
        return self._connected

    def start(self):
        self.started += 1
        self._connected = True

    def stop(self):
        self.stopped += 1
        self._connected = False
