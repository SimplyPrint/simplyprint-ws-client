"""``AsyncMqtt`` over real loopback sockets (plain and TLS) against the test
broker, which ``test_mqtt_broker_conformance`` proves against paho.

Besides the protocol (handshake, subscriptions, QoS 0/1/2 both ways,
keepalive, refusals, reconnects), these pin the property the transport exists
for: no thread ever touches the socket, so concurrent publishes while reports
stream in can't corrupt the TLS session (the paho race found end to end).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import List

import pytest
import pytest_asyncio
import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff
from simplyprint_ws_client.wire import async_mqtt
from simplyprint_ws_client.wire import mqtt_codec as codec
from simplyprint_ws_client.wire.async_mqtt import AsyncMqtt
from simplyprint_ws_client.wire.errors import AuthenticationError
from simplyprint_ws_client.wire.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.messages import MqttMessage, QoS
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import NotConnected, TransientError
from tests.mqtt_broker import Broker

pytestmark = [pytest.mark.behavior, pytest.mark.asyncio]

REPORT = "device/SN/report"
REQUEST = "device/SN/request"


class Events:
    def __init__(self, transport: AsyncMqtt) -> None:
        self.connected: List[int] = []
        self.disconnected: List[Disconnected] = []
        self.messages: List[MqttMessage] = []
        self.changed = asyncio.Event()
        transport.events.on(Connected, self._note(self.connected, "generation"))
        transport.events.on(Disconnected, self._note(self.disconnected, None))
        transport.events.on(MessageReceived, self._note(self.messages, "message"))

    def _note(self, into: list, attribute):
        def handler(event) -> None:
            into.append(getattr(event, attribute) if attribute else event)
            self.changed.set()

        return handler

    async def wait_for(self, predicate, timeout: float = 5.0) -> None:
        async def wait() -> None:
            while not predicate():
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(wait(), timeout)


@pytest_asyncio.fixture(params=[False, True], ids=["tcp", "tls"])
async def broker(request):
    b = await Broker(tls=request.param).start()
    yield b
    await b.stop()


def make(broker: Broker, *, keepalive: int = 60, topics=(REPORT,)) -> AsyncMqtt:
    transport = AsyncMqtt(
        yarl.URL(broker.url),
        EventLoopProvider(asyncio.get_running_loop()),
        keepalive=keepalive,
        policy=RetryPolicy(ConstantBackoff(0)),
    )
    for topic in topics:
        transport.subscribe(topic)
    return transport


@pytest_asyncio.fixture
async def started(broker):
    transport = make(broker)
    events = Events(transport)
    transport.start()
    await events.wait_for(lambda: events.connected)
    await broker.wait_for(lambda: broker.sessions and broker.sessions[0].subscriptions)
    yield transport, events
    await transport.stop()


async def test_connects_with_credentials_and_resubscribes(broker, started):
    transport, events = started
    session = broker.connects[0]
    assert (session.username, session.password) == ("bblp", "secret")
    assert session.client_id == ""  # MQTT 3.1.1 clean session: the broker assigns
    assert session.keepalive == 60
    assert session.subscriptions == {REPORT}
    assert transport.connected and events.connected == [1]


async def test_inbound_qos_0_1_2_are_delivered_once_and_acknowledged(broker, started):
    transport, events = started
    broker.publish(REPORT, b"q0", 0)
    broker.publish(REPORT, b"q1", 1, packet_id=7)
    broker.publish(REPORT, b"q2", 2, packet_id=9)
    broker.publish(REPORT, b"q2", 2, packet_id=9, dup=True)  # a re-send before PUBREL
    await events.wait_for(lambda: len(events.messages) == 3)
    assert [(m.payload, m.qos) for m in events.messages] == [
        (b"q0", QoS.AT_MOST_ONCE),
        (b"q1", QoS.AT_LEAST_ONCE),
        (b"q2", QoS.AT_LEAST_ONCE),
    ]
    await broker.wait_for(lambda: broker.pubacks.get(7) == codec.PUBACK)
    await broker.wait_for(lambda: broker.pubacks.get(9) == codec.PUBCOMP)
    await asyncio.sleep(0.1)
    assert len(events.messages) == 3


async def test_qos0_send_and_qos1_send_awaits_the_puback(broker, started):
    transport, _events = started
    await transport.send(MqttMessage(REQUEST, b"zero"))
    await transport.send(MqttMessage(REQUEST, b"one", QoS.AT_LEAST_ONCE, retain=True))
    await broker.wait_for(lambda: len(broker.published) == 2)
    assert broker.published == [
        (REQUEST, b"zero", 0, False),
        (REQUEST, b"one", 1, True),
    ]


async def test_an_unacknowledged_publish_fails_and_replaces_the_connection(
    broker, started, monkeypatch
):
    transport, events = started
    monkeypatch.setattr(async_mqtt, "PUBLISH_ACK_TIMEOUT", 0.2)
    broker.ack_publishes = False
    with pytest.raises(TransientError):
        await transport.send(MqttMessage(REQUEST, b"lost", QoS.AT_LEAST_ONCE))
    await events.wait_for(lambda: events.connected == [1, 2])


async def test_subscriptions_follow_the_desired_set_across_reconnects(broker, started):
    transport, events = started
    transport.subscribe("device/OTHER/report")
    await broker.wait_for(
        lambda: broker.sessions[0].subscriptions == {REPORT, "device/OTHER/report"}
    )
    transport.unsubscribe(REPORT)
    await broker.wait_for(lambda: broker.sessions[0].subscriptions == {"device/OTHER/report"})

    broker.drop_all()
    await events.wait_for(lambda: events.connected == [1, 2])
    await broker.wait_for(
        lambda: broker.sessions and broker.sessions[0].subscriptions == {"device/OTHER/report"}
    )
    assert isinstance(events.disconnected[0].code, TransientError)


async def test_a_refused_subscription_is_logged_and_the_link_stays(broker, caplog):
    broker.refuse_topics = {"device/NOPE/report"}
    transport = make(broker, topics=(REPORT, "device/NOPE/report"))
    events = Events(transport)
    with caplog.at_level(logging.WARNING, logger="wire.mqtt"):
        transport.start()
        await events.wait_for(lambda: events.connected)
        await broker.wait_for(lambda: broker.packets(codec.SUBSCRIBE))
        await asyncio.sleep(0.1)
    try:
        assert "refused the subscription to device/NOPE/report" in caplog.text
        assert transport.connected and events.connected == [1]
    finally:
        await transport.stop()


async def test_wrong_credentials_are_an_authentication_error(broker):
    broker.connack_code = 5
    transport = make(broker)
    events = Events(transport)
    transport.start()
    try:
        await events.wait_for(lambda: events.disconnected)
        error = events.disconnected[0].code
        assert isinstance(error, AuthenticationError) and error.code == 5
        assert not transport.connected
    finally:
        await transport.stop()


async def test_a_broker_that_refuses_an_empty_client_id_gets_a_random_one(broker):
    broker.reject_empty_client_id = True
    transport = make(broker)
    events = Events(transport)
    transport.start()
    try:
        await events.wait_for(lambda: events.connected)
        ids = [session.client_id for session in broker.connects]
        assert ids[0] == "" and len(ids[-1]) == 22
    finally:
        await transport.stop()


async def test_keepalive_pings_and_a_silent_broker_is_dropped(broker):
    transport = make(broker, keepalive=1)
    events = Events(transport)
    transport.start()
    try:
        await events.wait_for(lambda: events.connected)
        await broker.wait_for(lambda: broker.packets(codec.PINGREQ), timeout=3)
        assert transport.connected
        broker.answer_pings = False
        await events.wait_for(lambda: events.disconnected, timeout=5)
        assert "PINGRESP" in str(events.disconnected[0].code)
        await events.wait_for(lambda: len(events.connected) == 2)
    finally:
        await transport.stop()


async def test_stop_says_goodbye_and_settles(broker, started):
    transport, _events = started
    await transport.stop()
    await broker.wait_for(lambda: broker.packets(codec.DISCONNECT))
    assert transport.state is ConnectionState.DISCONNECTED
    with pytest.raises(NotConnected):
        await transport.send(MqttMessage(REQUEST, b"late"))


async def test_no_thread_ever_touches_the_socket(broker):
    """The paho race: commands published while reports stream in, over TLS.
    Here every byte moves on the event loop; the transport starts no thread."""
    threads = set(threading.enumerate())
    transport = make(broker)
    events = Events(transport)
    transport.start()
    try:
        await events.wait_for(lambda: events.connected)
        await broker.wait_for(lambda: broker.sessions and broker.sessions[0].subscriptions)

        async def stream() -> None:
            for n in range(300):
                broker.publish(REPORT, b"x" * 2048 + str(n).encode(), n % 2)
                await asyncio.sleep(0)

        async def command(n: int) -> None:
            await transport.send(
                MqttMessage(REQUEST, f"cmd{n}".encode(), QoS(n % 2))
            )

        await asyncio.gather(stream(), *(command(n) for n in range(200)))
        await events.wait_for(lambda: len(events.messages) == 300, timeout=10)
        await broker.wait_for(lambda: len(broker.published) == 200, timeout=10)
        assert events.connected == [1] and not events.disconnected
        assert set(threading.enumerate()) - threads == set()
    finally:
        await transport.stop()


async def test_subscribe_from_another_thread_hops_onto_the_loop(broker, started):
    transport, _events = started
    await asyncio.to_thread(transport.subscribe, "device/THREAD/report")
    await broker.wait_for(
        lambda: "device/THREAD/report" in broker.sessions[0].subscriptions
    )


async def test_the_pool_front_door_builds_this_transport(broker):
    from simplyprint_ws_client.wire import mqtt
    from simplyprint_ws_client.wire.options import ConnectionOptions
    from simplyprint_ws_client.wire.pools import PoolRegistry

    pool = mqtt.pool_for(PoolRegistry(), EventLoopProvider(asyncio.get_running_loop()))
    one = mqtt.connect(broker.url, pool=pool, topics=["device/ONE/report"],
                       options=ConnectionOptions())
    two = mqtt.connect(broker.url, pool=pool, topics=["device/TWO/report"],
                       options=ConnectionOptions())
    got_one, got_two = [], []
    one.event_bus.on(MessageReceived, lambda e: got_one.append(e.message.payload))
    two.event_bus.on(MessageReceived, lambda e: got_two.append(e.message.payload))
    try:
        assert one.transport is two.transport
        assert isinstance(one.transport, AsyncMqtt)
        assert await asyncio.wait_for(one.ready(), 5)
        await broker.wait_for(
            lambda: broker.sessions
            and broker.sessions[0].subscriptions == {"device/ONE/report", "device/TWO/report"}
        )
        broker.publish("device/ONE/report", b"for one")
        broker.publish("device/TWO/report", b"for two")
        for _ in range(50):
            if got_one and got_two:
                break
            await asyncio.sleep(0.05)
        assert (got_one, got_two) == ([b"for one"], [b"for two"])
        assert len(broker.connects) == 1  # one socket for both leases
    finally:
        await one.close()
        await two.close()


# -- the connection layer: TLS, large packets, back-pressure ----------------------


async def test_verify_tls_refuses_a_self_signed_printer():
    broker = await Broker(tls=True).start()
    transport = AsyncMqtt(
        yarl.URL(broker.url),
        EventLoopProvider(asyncio.get_running_loop()),
        verify_tls=True,
        policy=RetryPolicy(ConstantBackoff(0)),
    )
    events = Events(transport)
    transport.start()
    try:
        await events.wait_for(lambda: events.disconnected)
        assert "CERTIFICATE_VERIFY_FAILED" in str(events.disconnected[0].code)
        assert events.connected == [] and broker.connects == []
    finally:
        await transport.stop()
        await broker.stop()


async def test_a_large_report_arrives_whole(broker, started):
    transport, events = started
    report = bytes(range(256)) * 4096  # 1 MiB: 64 TLS records
    broker.publish(REPORT, report, 1)
    await events.wait_for(lambda: events.messages, timeout=10)
    assert events.messages[0].payload == report


async def test_a_slow_consumer_pauses_reading_instead_of_buffering(broker):
    from simplyprint_ws_client.wire import mqtt_stream

    transport = make(broker)
    events = Events(transport)
    release = asyncio.Event()
    delivered: List[int] = []

    async def slow(event) -> None:
        await release.wait()
        delivered.append(len(event.message.payload))

    transport.events.on(MessageReceived, slow)
    transport.start()
    try:
        await events.wait_for(lambda: events.connected)
        await broker.wait_for(lambda: broker.sessions and broker.sessions[0].subscriptions)
        for _ in range(12):
            broker.publish(REPORT, b"r" * (256 * 1024))  # 3 MiB in all
        connection = transport.connection
        for _ in range(100):
            if connection._reading_paused:
                break
            await asyncio.sleep(0.02)
        assert connection._reading_paused
        assert connection._queued < 2 * mqtt_stream.READ_HIGH_WATER
        release.set()
        for _ in range(200):
            if len(delivered) == 12:
                break
            await asyncio.sleep(0.02)
        assert delivered == [256 * 1024] * 12
        assert not connection._reading_paused and events.connected == [1]
    finally:
        await transport.stop()


async def test_a_graceful_close_from_the_broker_reconnects(broker, started):
    transport, events = started
    broker.sessions[0].writer.close()  # TLS close_notify, then FIN
    await events.wait_for(lambda: events.connected == [1, 2])
    assert "closed" in str(events.disconnected[0].code)


async def test_a_malformed_packet_drops_the_connection_and_reconnects(broker, started):
    transport, events = started
    broker.sessions[0].send(bytes((0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F)))
    await events.wait_for(lambda: events.connected == [1, 2])
    assert "malformed" in str(events.disconnected[0].code)


# -- review round 1 (Astra): uvloop, endless outages, probes, acks, ids, codec ----



async def test_reconnecting_survives_more_than_a_thousand_refusals(broker):
    broker.connack_code = 3  # server unavailable, forever
    transport = AsyncMqtt(
        yarl.URL(broker.url), EventLoopProvider(asyncio.get_running_loop())
    )
    real_delay = transport.policy.delay
    calls = []

    def no_wait() -> float:
        calls.append(real_delay())  # the production arithmetic, minus the sleep
        return 0

    transport.policy.delay = no_wait
    transport.start()
    try:
        for _ in range(3000):
            if len(calls) > 1100:
                break
            await asyncio.sleep(0.01)
        assert len(calls) > 1100
        assert transport.supervising() and not transport.task.done()
    finally:
        await transport.stop()


async def test_a_probe_reaches_a_broker_that_refuses_an_empty_client_id(broker):
    from simplyprint_ws_client.wire.mqtt_probe import MqttProbeOutcome, probe_mqtt

    broker.reject_empty_client_id = True
    result = await probe_mqtt(
        broker.url + "/?topic=device/SN/report",
        connect_timeout=5,
        report_timeout=0,
        require_report=False,
    )
    assert result.outcome is MqttProbeOutcome.VERIFIED
    assert [s.client_id for s in broker.connects][0] == ""
    assert len(broker.connects[-1].client_id) == 22


async def test_an_ack_arrives_while_a_message_listener_awaits_it(broker):
    """A listener that sends at QoS 1 and awaits the PUBACK must not block the
    PUBACK: acknowledgements bypass the message queue."""
    transport = make(broker)
    events = Events(transport)
    acked: List[str] = []

    async def reply(event) -> None:
        await transport.send(MqttMessage(REQUEST, b"reply", QoS.AT_LEAST_ONCE))
        acked.append("acked")

    transport.events.on(MessageReceived, reply)
    transport.start()
    try:
        await events.wait_for(lambda: events.connected)
        await broker.wait_for(lambda: broker.sessions and broker.sessions[0].subscriptions)
        broker.publish(REPORT, b"please reply")
        for _ in range(100):
            if acked:
                break
            await asyncio.sleep(0.02)
        assert acked == ["acked"]
    finally:
        await transport.stop()


async def test_an_unacknowledged_unsubscribe_keeps_its_packet_id(broker, started):
    transport, _events = started
    broker.sessions[0].writer.transport.pause_reading()  # no UNSUBACK for now
    transport.unsubscribe(REPORT)
    (pending,) = transport._unsubscribing
    transport._next_packet_id = pending - 1 if pending > 1 else 0xFFFF
    transport.subscribe("device/NEXT/report")
    assert pending not in transport._subscribing


async def test_a_qos2_resend_queued_before_its_pubrel_is_still_deduplicated(broker):
    """Round 2: the PUBREL must not overtake a queued duplicate and clear the
    dedupe state early (which delivered it twice)."""
    transport = make(broker)
    events = Events(transport)
    release = asyncio.Event()

    async def slow(event) -> None:
        if event.message.payload == b"block":
            await release.wait()

    transport.events.on(MessageReceived, slow)
    transport.start()
    try:
        await events.wait_for(lambda: events.connected)
        await broker.wait_for(lambda: broker.sessions and broker.sessions[0].subscriptions)
        broker.publish(REPORT, b"once", 2, packet_id=5)
        await events.wait_for(lambda: len(events.messages) == 1)
        broker.publish(REPORT, b"block", 0)
        broker.publish(REPORT, b"once", 2, packet_id=5, dup=True)  # queued behind block
        broker.sessions[0].send(codec.encode_ack(codec.PUBREL, 5))
        await asyncio.sleep(0.1)
        release.set()
        await asyncio.sleep(0.2)
        assert [m.payload for m in events.messages] == [b"once", b"block"]
    finally:
        await transport.stop()
