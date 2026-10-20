"""``probe_mqtt`` over real loopback sockets: the test broker (TCP and TLS) for
the protocol paths, a silent server for timeouts/cancellation, and a closed
port for a refused connection."""

from __future__ import annotations

import asyncio
import socket

import pytest
import pytest_asyncio
import yarl

from simplyprint_ws_client.wire import async_mqtt
from simplyprint_ws_client.wire import mqtt_codec as codec
from simplyprint_ws_client.wire import mqtt_probe
from simplyprint_ws_client.wire.messages import MqttMessage, QoS
from simplyprint_ws_client.wire.mqtt_probe import MqttProbeOutcome, probe_mqtt
from tests.mqtt_broker import Broker
from tests.silent_server import SilentServer

pytestmark = pytest.mark.behavior

REPORT = "device/SN/report"


@pytest_asyncio.fixture
async def broker():
    b = await Broker().start()
    yield b
    await b.stop()


@pytest_asyncio.fixture(params=[False, True], ids=["tcp", "tls"])
async def any_broker(request):
    b = await Broker(tls=request.param).start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def silent():
    server = await SilentServer().start()
    yield server
    await server.stop()


def probe_url(broker: Broker, topic: str = "status") -> yarl.URL:
    return yarl.URL(broker.url + "/").with_query(topic=topic)


def subscribed(broker: Broker) -> list:
    """Every topic the client asked the broker for, in order."""
    topics = []
    for _kind, _flags, body in broker.packets(codec.SUBSCRIBE):
        offset = 2  # past the packet id
        while offset < len(body):
            length = int.from_bytes(body[offset : offset + 2], "big")
            topics.append(body[offset + 2 : offset + 2 + length].decode())
            offset += 2 + length + 1  # topic + requested QoS
    return topics


async def settled(broker: Broker) -> None:
    """The probe leaves nothing behind: the broker saw its session end."""
    await broker.wait_for(lambda: not broker.sessions)


@pytest.mark.asyncio
async def test_probe_uses_owner_pool_front_door_and_publishes_before_report(
    any_broker,
):
    broker = any_broker
    commands = (
        MqttMessage("device/SN/request", b"start"),
        MqttMessage("device/SN/request", b"pushall"),
    )

    async def answer_pushall() -> None:
        await broker.wait_for(lambda: len(broker.published) == 2)
        broker.publish(REPORT, b"report")

    responder = asyncio.ensure_future(answer_pushall())
    try:
        result = await probe_mqtt(
            probe_url(broker, REPORT),
            connect_timeout=5,
            report_timeout=5,
            require_report=True,
            initial_messages=commands,
        )
    finally:
        responder.cancel()

    assert result.outcome is MqttProbeOutcome.VERIFIED
    assert result.report == MqttMessage(REPORT, b"report")
    assert result.messages_seen == 1
    session = broker.connects[0]
    assert (session.username, session.password) == ("bblp", "secret")
    assert session.keepalive == 20
    assert subscribed(broker) == [REPORT]
    assert [(topic, payload) for topic, payload, _q, _r in broker.published] == [
        ("device/SN/request", b"start"),
        ("device/SN/request", b"pushall"),
    ]
    assert len(broker.connects) == 1
    await broker.wait_for(lambda: broker.packets(codec.DISCONNECT))
    await settled(broker)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connack_code", "outcome"),
    [
        (1, MqttProbeOutcome.CONNECT_FAILED),  # unacceptable protocol version
        (2, MqttProbeOutcome.CONNECT_FAILED),  # identifier rejected
        (3, MqttProbeOutcome.CONNECT_FAILED),  # server unavailable
        pytest.param(
            4,
            MqttProbeOutcome.AUTH_FAILED,  # bad user name or password
        ),
        pytest.param(
            5,
            MqttProbeOutcome.AUTH_FAILED,  # not authorized
        ),
    ],
)
async def test_probe_classifies_mqtt_connack_failures(broker, connack_code, outcome):
    broker.connack_code = connack_code

    result = await probe_mqtt(
        probe_url(broker),
        connect_timeout=5,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is outcome
    assert result.reason_code == connack_code
    await settled(broker)


@pytest.mark.asyncio
async def test_probe_classifies_async_mqtt_authentication_error(broker):
    """The production wire's own refusal (an AuthenticationError from a real
    CONNACK) must classify as an auth failure, whatever form its code takes."""
    broker.connack_code = 5

    result = await probe_mqtt(
        probe_url(broker),
        connect_timeout=5,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.AUTH_FAILED
    assert result.reason == "MQTT connection refused (code 5)"


@pytest.mark.asyncio
async def test_probe_reports_first_transient_connect_failure():
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()  # nothing listens here: the connect is refused

    result = await probe_mqtt(
        f"mqtt://127.0.0.1:{port}/?topic=status",
        connect_timeout=5,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.CONNECT_FAILED
    assert result.reason
    assert result.reason_code is None


@pytest.mark.asyncio
async def test_probe_distinguishes_connect_and_required_report_timeouts(
    broker, silent
):
    connect_result = await probe_mqtt(
        silent.url(query="?topic=status"),
        connect_timeout=0.2,
        report_timeout=0,
        require_report=True,
    )
    assert connect_result.outcome is MqttProbeOutcome.CONNECT_TIMEOUT
    await silent.wait_for(lambda: silent.closed == silent.accepted == 1)

    report_result = await probe_mqtt(
        probe_url(broker),
        connect_timeout=5,
        report_timeout=0.1,
        require_report=True,
    )
    assert report_result.outcome is MqttProbeOutcome.REPORT_TIMEOUT
    await settled(broker)


@pytest.mark.asyncio
async def test_probe_accepts_missing_optional_report(broker):
    result = await probe_mqtt(
        probe_url(broker, "elegoo/#"),
        connect_timeout=5,
        report_timeout=0.05,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.VERIFIED
    assert result.report is None
    assert subscribed(broker) == ["elegoo/#"]
    await settled(broker)


@pytest.mark.asyncio
async def test_probe_reports_initial_publish_failure(broker, monkeypatch):
    monkeypatch.setattr(async_mqtt, "PUBLISH_ACK_TIMEOUT", 0.1)
    broker.ack_publishes = False  # the QoS 1 command is never acknowledged

    result = await probe_mqtt(
        probe_url(broker),
        connect_timeout=5,
        report_timeout=5,
        require_report=True,
        initial_messages=(MqttMessage("request", b"state", QoS.AT_LEAST_ONCE),),
    )

    assert result.outcome is MqttProbeOutcome.PUBLISH_FAILED
    assert "PUBACK" in (result.reason or "")
    await settled(broker)


@pytest.mark.asyncio
async def test_probe_cancellation_releases_lease_and_stops_transport(silent):
    probe = asyncio.create_task(
        probe_mqtt(
            silent.url(query="?topic=status"),
            connect_timeout=30,
            report_timeout=30,
            require_report=True,
        )
    )
    await silent.wait_for(lambda: silent.accepted == 1)
    probe.cancel()

    with pytest.raises(asyncio.CancelledError):
        await probe

    # The socket was closed, not leaked: the server saw the connection end,
    # after the CONNECT the probe sent (it never got a CONNACK).
    await silent.wait_for(lambda: silent.closed == 1)
    assert silent.received[0][:1] == bytes((codec.CONNECT << 4,))
