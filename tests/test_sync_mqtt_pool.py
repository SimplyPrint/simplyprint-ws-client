"""Brand-free tests for the sync paho ``MqttPool`` on the Courier.

The point of the sync family: paho fires its callbacks on its own network
thread, yet a lease's handlers must run on the consumer loop. These tests use a
fake paho client that fires callbacks from a *background thread* and assert the
events arrive on the loop thread (via the courier), routed to the right lease --
with no broker and no paho installed.
"""

import asyncio
import threading

import pytest

from simplyprint_ws_client.contrib.connection.async_mqtt import MqttParams
from simplyprint_ws_client.contrib.connection.sync_mqtt import (
    MqttPool,
    PahoMqttTransport,
)
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider


class _Msg:
    def __init__(self, topic, payload=b""):
        self.topic = topic
        self.payload = payload


class _Info:
    rc = 0


class FakePahoClient:
    """A paho-shaped client whose callbacks tests can fire from any thread."""

    def __init__(self):
        self.on_connect = None
        self.on_connect_fail = None
        self.on_message = None
        self.on_disconnect = None
        self._connected = False
        self.subscriptions = []
        self.unsubscriptions = []
        self.published = []
        self.loop_started = 0

    def username_pw_set(self, *_a, **_k):
        pass

    def connect_async(self, *_a, **_k):
        pass

    def loop_start(self):
        self.loop_started += 1
        return 0

    def loop_stop(self):
        pass

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def subscribe(self, topic):
        self.subscriptions.append(topic)

    def unsubscribe(self, topic):
        self.unsubscriptions.append(topic)

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload))
        return _Info()

    def fire_connect(self, reason_code=0):
        self._connected = True
        self.on_connect(self, None, {}, reason_code, None)

    def fire_message(self, topic, payload=b""):
        self.on_message(self, None, _Msg(topic, payload))

    def fire_disconnect(self):
        self._connected = False
        self.on_disconnect(self, None, None, 0, None)


def _pool(loop, clients=None, **pool_kwargs):
    """An MqttPool whose transports use FakePahoClient; records created clients."""
    created = clients if clients is not None else []

    def transport_factory(params):
        client = FakePahoClient()
        created.append(client)
        return PahoMqttTransport(params, client_factory=lambda p, log: client)

    pool = MqttPool(
        transport_factory=transport_factory,
        event_loop_provider=EventLoopProvider(loop=loop),
        **pool_kwargs,
    )
    return pool, created


def _from_thread(fn, *args):
    t = threading.Thread(target=fn, args=args)
    t.start()
    t.join(1.0)


@pytest.mark.asyncio
async def test_message_from_paho_thread_arrives_on_the_loop():
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    pool, clients = _pool(loop)

    got = []
    lease = pool.connect(MqttParams("broker", 8883), route="printer/a/report")
    lease.on_message(lambda msg: got.append((msg.topic, threading.get_ident())))

    client = clients[0]
    _from_thread(client.fire_connect)
    _from_thread(client.fire_message, "printer/a/report", b"hello")

    for _ in range(200):
        await asyncio.sleep(0.005)
        if got:
            break

    assert len(got) == 1
    topic, ran_thread = got[0]
    assert topic == "printer/a/report"
    assert ran_thread == loop_thread  # delivered on the loop, not the paho thread

    await asyncio.sleep(0)
    lease.close()
    pool.stop()


@pytest.mark.asyncio
async def test_route_subscription_is_refcounted_and_removed():
    loop = asyncio.get_running_loop()
    pool, clients = _pool(loop)
    params = MqttParams("broker", 8883)

    a = pool.connect(params, route="printer/a/report")
    b = pool.connect(params, route="printer/a/report")
    client = clients[0]
    _from_thread(client.fire_connect)
    await asyncio.sleep(0.02)

    assert client.subscriptions == ["printer/a/report"]

    a.close()
    assert client.unsubscriptions == []

    b.close()
    assert client.unsubscriptions == ["printer/a/report"]
    pool.stop()


@pytest.mark.asyncio
async def test_message_backlog_is_bounded_but_lifecycle_is_lossless():
    loop = asyncio.get_running_loop()
    pool, clients = _pool(loop, message_maxsize=1)
    lease = pool.connect(MqttParams("broker", 8883), route="printer/a/report")

    messages = []
    lifecycle = []
    lease.on_message(lambda msg: messages.append(msg.payload))
    lease.on_connected(lambda: lifecycle.append("up"))

    client = clients[0]

    def burst():
        for i in range(20):
            client.fire_message("printer/a/report", bytes([i]))
        client.fire_connect()

    _from_thread(burst)

    for _ in range(200):
        await asyncio.sleep(0.005)
        if lifecycle and messages:
            break

    assert lifecycle == ["up"]
    assert len(messages) == 1
    assert messages[0] == bytes([19])

    lease.close()
    pool.stop()


@pytest.mark.asyncio
async def test_routing_and_lifecycle_across_two_leases():
    loop = asyncio.get_running_loop()
    pool, clients = _pool(loop)

    a_msgs, b_msgs, events = [], [], []
    a = pool.connect(MqttParams("broker", 8883), route="p/a/#")
    b = pool.connect(MqttParams("broker", 8883), route="p/b/#")
    a.on_message(lambda m: a_msgs.append(m.topic))
    b.on_message(lambda m: b_msgs.append(m.topic))
    a.on_connected(lambda: events.append("a-up"))
    b.on_connected(lambda: events.append("b-up"))

    assert len(clients) == 1  # one shared transport for both leases

    client = clients[0]
    _from_thread(client.fire_connect)
    _from_thread(client.fire_message, "p/a/report", b"1")
    _from_thread(client.fire_message, "p/b/report", b"2")

    for _ in range(200):
        await asyncio.sleep(0.005)
        if a_msgs and b_msgs:
            break

    assert a_msgs == ["p/a/report"]  # routed by topic
    assert b_msgs == ["p/b/report"]
    assert set(events) == {"a-up", "b-up"}  # connect fans to all leases

    a.close()
    b.close()
    pool.stop()


@pytest.mark.asyncio
async def test_refcount_shares_and_tears_down_transport():
    loop = asyncio.get_running_loop()
    pool, clients = _pool(loop)
    params = MqttParams("broker", 8883)

    a = pool.connect(params)
    b = pool.connect(params)
    assert len(clients) == 1
    assert clients[0].loop_started == 1

    a.close()
    assert params in pool._refs  # still pooled: b holds it
    assert len(clients) == 1  # no new transport built

    b.close()
    assert params not in pool._refs  # last lease left -> torn down


@pytest.mark.asyncio
async def test_send_publishes_through_the_paho_client():
    loop = asyncio.get_running_loop()
    pool, clients = _pool(loop)
    lease = pool.connect(MqttParams("broker", 8883))

    client = clients[0]
    _from_thread(client.fire_connect)  # publish requires a live link
    await asyncio.sleep(0)

    assert lease.send(("cmd/topic", b"payload")) is True
    assert client.published == [("cmd/topic", b"payload")]

    lease.close()
    pool.stop()
