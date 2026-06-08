"""Brand-free tests for the ``Pool.connect() -> Connection`` lease + routing.

These pin the front-door contract: one shared transport per endpoint
(ref-counted), per-lease topic routing (a client sees only its own messages, not
every other client's), clean lease teardown, and the ``submit_to_loop``
cross-thread coroutine hop that replaces every ad-hoc reach-into-a-client.

Exercised with a fake aiomqtt-shaped transport, so no broker is involved.
"""

import asyncio
import threading

import pytest

from simplyprint_ws_client.contrib.connection.mqtt import (
    AsyncMqttPool,
    MqttParams,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    AsyncTransport,
    Connected,
    Disconnected,
    MessageReceived,
    TransportEvent,
)
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider


class _Msg:
    """An aiomqtt-message stand-in: the router reads ``.topic``."""

    def __init__(self, topic, data=None):
        self.topic = topic
        self.data = data


class FakeAsyncTransport(AsyncTransport):
    """A broker-free async transport that records lifecycle + lets tests emit."""

    def __init__(self, params):
        self.params = params
        self.events: EventBus[TransportEvent] = EventBus()
        self.state = ConnectionState.OFFLINE
        self.started = 0
        self.stopped = 0
        self.subscriptions = []
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

    def subscribe(self, topic: str) -> None:
        self.subscriptions.append(topic)

    async def send(self, payload) -> None:
        self.sent.append(payload)

    def emit(self, event: TransportEvent) -> None:
        self.events.emit_sync(type(event), event)


def _pool_with_recorder(loop=None):
    """An AsyncMqttPool wired to FakeAsyncTransport, exposing every built one."""
    built = []

    def factory(params):
        transport = FakeAsyncTransport(params)
        built.append(transport)
        return transport

    provider = EventLoopProvider(loop=loop) if loop is not None else None
    pool = AsyncMqttPool(transport_factory=factory, event_loop_provider=provider)
    return pool, built


@pytest.mark.asyncio
async def test_connect_shares_one_transport_per_endpoint_and_refcounts():
    pool, built = _pool_with_recorder()
    params = MqttParams("broker", 8883)

    a = await pool.connect(params)
    b = await pool.connect(params)

    assert len(built) == 1  # one shared transport for both leases
    assert built[0].started == 1

    await a.close()
    assert built[0].stopped == 0  # still held by b

    await b.close()
    assert built[0].stopped == 1  # last lease left -> torn down


@pytest.mark.asyncio
async def test_distinct_endpoints_get_distinct_transports():
    pool, built = _pool_with_recorder()
    await pool.connect(MqttParams("a", 1))
    await pool.connect(MqttParams("b", 2))
    assert len(built) == 2


@pytest.mark.asyncio
async def test_messages_route_only_to_the_matching_lease():
    pool, built = _pool_with_recorder()
    params = MqttParams("broker", 8883)

    got_a, got_b = [], []
    a = await pool.connect(params, route="printer/a/report")
    b = await pool.connect(params, route="printer/b/report")
    a.on_message(got_a.append)
    b.on_message(got_b.append)

    transport = built[0]
    transport.emit(MessageReceived(_Msg("printer/a/report", "hello-a")))

    assert [m.data for m in got_a] == ["hello-a"]
    assert got_b == []  # b never saw a's message


@pytest.mark.asyncio
async def test_wildcard_route_matches():
    pool, built = _pool_with_recorder()
    params = MqttParams("broker", 8883)

    got = []
    lease = await pool.connect(params, route="printer/#")
    lease.on_message(got.append)

    built[0].emit(MessageReceived(_Msg("printer/xyz/report", "data")))
    assert [m.data for m in got] == ["data"]


@pytest.mark.asyncio
async def test_connect_and_disconnect_fan_to_all_leases():
    pool, built = _pool_with_recorder()
    params = MqttParams("broker", 8883)

    events_a, events_b = [], []
    a = await pool.connect(params, route="t/a")
    b = await pool.connect(params, route="t/b")
    a.on_connected(lambda: events_a.append("up"))
    b.on_connected(lambda: events_b.append("up"))
    a.on_disconnected(lambda e: events_a.append(("down", e.transient)))
    b.on_disconnected(lambda e: events_b.append(("down", e.transient)))

    built[0].emit(Connected())
    built[0].emit(Disconnected(transient=True))

    assert events_a == ["up", ("down", True)]
    assert events_b == ["up", ("down", True)]  # broker up/down is shared


@pytest.mark.asyncio
async def test_closing_a_lease_stops_its_delivery():
    pool, built = _pool_with_recorder()
    params = MqttParams("broker", 8883)

    got = []
    a = await pool.connect(params, route="t/a")
    b = await pool.connect(params, route="t/a")  # keep transport alive after a closes
    a.on_message(got.append)

    await a.close()
    built[0].emit(MessageReceived(_Msg("t/a", "after-close")))

    assert got == []  # a unsubscribed itself from routing
    await b.close()


@pytest.mark.asyncio
async def test_connect_with_route_subscribes_the_transport():
    pool, built = _pool_with_recorder()
    await pool.connect(MqttParams("broker", 8883), route="printer/+/report")
    assert built[0].subscriptions == ["printer/+/report"]


@pytest.mark.asyncio
async def test_send_delegates_to_the_shared_transport():
    pool, built = _pool_with_recorder()
    lease = await pool.connect(MqttParams("broker", 8883))
    await lease.send(("topic", b"payload"))
    assert built[0].sent == [("topic", b"payload")]


@pytest.mark.asyncio
async def test_async_message_handler_is_scheduled_on_the_loop():
    pool, built = _pool_with_recorder()
    got = []

    async def handler(msg):
        await asyncio.sleep(0)
        got.append(msg.data)

    lease = await pool.connect(MqttParams("broker", 8883), route="t/a")
    lease.on_message(handler)

    built[0].emit(MessageReceived(_Msg("t/a", "async-data")))
    await asyncio.sleep(0.01)  # let the scheduled task run

    assert got == ["async-data"]


@pytest.mark.asyncio
async def test_submit_to_loop_runs_off_thread_and_coalesces():
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    pool, _ = _pool_with_recorder(loop=loop)
    lease = await pool.connect(MqttParams("broker", 8883))

    calls = []
    ran_on = []
    running = threading.Event()
    allow_finish = threading.Event()

    async def work(tag):
        ran_on.append(threading.get_ident())
        calls.append(tag)
        running.set()
        while not allow_finish.is_set():
            await asyncio.sleep(0.005)

    def producer():
        # Two submissions with the same key while the first is in flight: the
        # second must be a no-op (coalesced). Both return immediately.
        lease.submit_to_loop(lambda: work("a"), coalesce_key="k")
        lease.submit_to_loop(lambda: work("b"), coalesce_key="k")

    t = threading.Thread(target=producer)
    t.start()
    t.join(1.0)
    assert not t.is_alive()  # submit_to_loop returned immediately

    for _ in range(200):
        await asyncio.sleep(0.005)
        if running.is_set():
            break

    assert calls == ["a"]  # "b" was coalesced away
    assert ran_on[0] == loop_thread  # ran ON the loop, not the producer thread

    allow_finish.set()
    await asyncio.sleep(0.02)  # let "a" finish and free the coalesce key

    lease.submit_to_loop(lambda: work("c"), coalesce_key="k")
    await asyncio.sleep(0.02)
    assert calls == ["a", "c"]  # key freed after completion -> runs again
