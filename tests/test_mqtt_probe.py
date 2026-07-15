from __future__ import annotations

import asyncio
import logging
import ssl
from types import SimpleNamespace

import pytest
import yarl
from paho.mqtt.client import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode

from simplyprint_ws_client.wire import mqtt
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff
from simplyprint_ws_client.wire.messages import MqttMessage
from simplyprint_ws_client.wire.mqtt_probe import MqttProbeOutcome, probe_mqtt
from simplyprint_ws_client.wire.options import WireKeepalive
from simplyprint_ws_client.wire.paho import default_paho_client
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.pools import PoolRegistry


class FakePublishInfo:
    def __init__(self, rc: int) -> None:
        self.rc = rc

    def wait_for_publish(self, timeout=None) -> None:
        return None

    def is_published(self) -> bool:
        return self.rc == 0


class FakePahoClient:
    def __init__(
        self,
        *,
        reason_code: int | ReasonCode | None = 0,
        report_topic: str | None = None,
        publish_rc: int = 0,
        connect_failure: bool = False,
    ) -> None:
        self.reason_code = reason_code
        self.report_topic = report_topic
        self.publish_rc = publish_rc
        self.connect_failure = connect_failure
        self.connected = False
        self.credentials = None
        self.connect_args = None
        self.subscriptions: list[str] = []
        self.published: list[tuple[str, bytes]] = []
        self.stopped = 0
        self.on_pre_connect = None
        self.on_connect = None
        self.on_connect_fail = None
        self.on_message = None
        self.on_disconnect = None

    def username_pw_set(self, username, password) -> None:
        self.credentials = username, password

    def connect_async(self, host, port, keepalive) -> None:
        self.connect_args = host, port, keepalive

    def loop_start(self) -> int:
        if self.connect_failure:
            self.on_connect_fail(self, None)
            return 0
        if self.reason_code is None:
            return 0
        self.on_pre_connect(self, None)
        if self.reason_code == 0:
            self.connected = True
            self.on_connect(self, None, {}, 0, None)
            if self.report_topic is not None:
                self.on_message(
                    self,
                    None,
                    SimpleNamespace(
                        topic=self.report_topic,
                        payload=b"report",
                        qos=0,
                        retain=False,
                    ),
                )
        else:
            self.on_connect(self, None, {}, self.reason_code, None)
        return 0

    def loop_stop(self) -> int:
        self.stopped += 1
        return 0

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def subscribe(self, topic: str):
        self.subscriptions.append(topic)
        return 0, len(self.subscriptions)

    def unsubscribe(self, topic: str):
        return 0, 1

    def publish(self, topic, payload, *, qos, retain) -> FakePublishInfo:
        self.published.append((topic, payload))
        return FakePublishInfo(self.publish_rc)


def install_client(monkeypatch, client: FakePahoClient) -> None:
    monkeypatch.setattr(mqtt, "default_paho_client", lambda _url, _logger, **_k: client)


def test_default_paho_client_matches_released_bambu_wire_parameters():
    client = default_paho_client(
        yarl.URL("mqtts://printer"),
        logging.getLogger("test.paho.defaults"),
    )

    assert client._callback_api_version is CallbackAPIVersion.VERSION2
    assert client._reconnect_on_failure is True
    assert (client._reconnect_min_delay, client._reconnect_max_delay) == (1, 5)
    assert client._tls_insecure is True
    assert client._ssl_context.verify_mode == ssl.CERT_NONE
    assert client._logger is None


def test_default_paho_client_maps_an_explicit_retry_policy():
    client = default_paho_client(
        yarl.URL("mqtt://printer"),
        logging.getLogger("test.paho.retry"),
        retry=RetryPolicy(backoff=ConstantBackoff(7)),
    )

    assert (client._reconnect_min_delay, client._reconnect_max_delay) == (7, 7)


@pytest.mark.asyncio
async def test_probe_uses_owner_pool_front_door_and_publishes_before_report(
    monkeypatch,
):
    client = FakePahoClient(report_topic="device/SN/report")
    install_client(monkeypatch, client)
    pools = PoolRegistry()
    commands = (
        MqttMessage("device/SN/request", b"start"),
        MqttMessage("device/SN/request", b"pushall"),
    )

    result = await probe_mqtt(
        pools,
        yarl.URL("mqtts://bblp:secret@printer:8883/?topic=device/SN/report"),
        connect_timeout=0.1,
        report_timeout=0.1,
        require_report=True,
        initial_messages=commands,
    )

    assert result.outcome is MqttProbeOutcome.VERIFIED
    assert result.report == MqttMessage("device/SN/report", b"report")
    assert result.messages_seen == 1
    assert client.credentials == ("bblp", "secret")
    assert client.connect_args == ("printer", 8883, 20)
    assert "device/SN/report" in client.subscriptions
    assert client.published == [
        ("device/SN/request", b"start"),
        ("device/SN/request", b"pushall"),
    ]
    assert client.stopped == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason_code", "outcome"),
    [
        (134, MqttProbeOutcome.AUTH_FAILED),
        (135, MqttProbeOutcome.AUTH_FAILED),
        (140, MqttProbeOutcome.AUTH_FAILED),
        (136, MqttProbeOutcome.CONNECT_FAILED),
    ],
)
async def test_probe_classifies_mqtt_connack_failures(
    monkeypatch, reason_code, outcome
):
    client = FakePahoClient(reason_code=reason_code)
    install_client(monkeypatch, client)

    result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://user:secret@printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is outcome
    assert result.reason_code == reason_code
    assert client.stopped == 1


@pytest.mark.asyncio
async def test_probe_classifies_real_paho_reason_code(monkeypatch):
    client = FakePahoClient(reason_code=ReasonCode(PacketTypes.CONNACK, identifier=135))
    install_client(monkeypatch, client)

    result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://user:secret@printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.AUTH_FAILED
    assert result.reason_code == 135


@pytest.mark.asyncio
async def test_probe_reports_first_transient_connect_failure(monkeypatch):
    client = FakePahoClient(connect_failure=True)
    install_client(monkeypatch, client)

    result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.CONNECT_FAILED
    assert result.reason == "paho connect failed"


@pytest.mark.asyncio
async def test_probe_distinguishes_connect_and_required_report_timeouts(monkeypatch):
    silent = FakePahoClient(reason_code=None)
    install_client(monkeypatch, silent)
    connect_result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://printer/?topic=status",
        connect_timeout=0.001,
        report_timeout=0,
        require_report=True,
    )
    assert connect_result.outcome is MqttProbeOutcome.CONNECT_TIMEOUT

    connected = FakePahoClient()
    install_client(monkeypatch, connected)
    report_result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0.001,
        require_report=True,
    )
    assert report_result.outcome is MqttProbeOutcome.REPORT_TIMEOUT


@pytest.mark.asyncio
async def test_probe_accepts_missing_optional_report(monkeypatch):
    client = FakePahoClient()
    install_client(monkeypatch, client)

    result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://printer/?topic=elegoo/%23",
        connect_timeout=0.1,
        report_timeout=0.001,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.VERIFIED
    assert result.report is None


@pytest.mark.asyncio
async def test_probe_reports_initial_publish_failure(monkeypatch):
    client = FakePahoClient(report_topic="status", publish_rc=4)
    install_client(monkeypatch, client)

    result = await probe_mqtt(
        PoolRegistry(),
        "mqtt://printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0.1,
        require_report=True,
        initial_messages=(MqttMessage("request", b"state"),),
    )

    assert result.outcome is MqttProbeOutcome.PUBLISH_FAILED


@pytest.mark.asyncio
async def test_probe_cancellation_releases_lease_and_stops_transport(monkeypatch):
    client = FakePahoClient(reason_code=None)
    install_client(monkeypatch, client)
    pools = PoolRegistry()

    probe = asyncio.create_task(
        probe_mqtt(
            pools,
            "mqtt://printer/?topic=status",
            connect_timeout=30,
            report_timeout=30,
            require_report=True,
        )
    )
    await asyncio.sleep(0)
    probe.cancel()

    with pytest.raises(asyncio.CancelledError):
        await probe

    provider = EventLoopProvider(asyncio.get_running_loop())
    pool = mqtt.pool_for(pools, provider, WireKeepalive(interval=20))
    assert pool.endpoints == {}
    assert client.stopped == 1
