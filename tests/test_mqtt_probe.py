from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
import yarl
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode

from simplyprint_ws_client.wire import mqtt
from simplyprint_ws_client.wire.messages import MqttMessage
from simplyprint_ws_client.wire.mqtt_probe import MqttProbeOutcome, probe_mqtt


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
        self.disconnects = 0
        self.loop_exit = threading.Event()
        self.drove_connection = False
        self.reported = False
        self.on_pre_connect = None
        self.on_connect = None
        self.on_connect_fail = None
        self.on_message = None
        self.on_disconnect = None

    def username_pw_set(self, username, password) -> None:
        self.credentials = username, password

    def connect_async(self, host, port, keepalive) -> None:
        self.connect_args = host, port, keepalive

    def loop_forever(self, timeout=1.0, retry_first_connection=False) -> int:
        if not self.drove_connection:
            self.drove_connection = True
            if self.connect_failure:
                self.on_connect_fail(self, None)
            elif self.reason_code == 0:
                self.connected = True
                self.on_connect(self, None, {}, 0, None)
            elif self.reason_code is not None:
                self.on_connect(self, None, {}, self.reason_code, None)
        self.loop_exit.wait()
        return 4

    def disconnect(self) -> None:
        self.disconnects += 1
        self.connected = False
        self.loop_exit.set()

    def is_connected(self) -> bool:
        return self.connected

    def subscribe(self, topic: str):
        self.subscriptions.append(topic)
        return 0, len(self.subscriptions)

    def unsubscribe(self, topic: str):
        return 0, 1

    def publish(self, topic, payload, *, qos, retain) -> FakePublishInfo:
        self.published.append((topic, payload))
        if self.publish_rc == 0 and self.report_topic is not None and not self.reported:
            self.reported = True
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
        return FakePublishInfo(self.publish_rc)


def install_client(monkeypatch, client: FakePahoClient) -> None:
    monkeypatch.setattr(mqtt, "default_paho_client", lambda _url, _logger, **_k: client)


@pytest.mark.asyncio
async def test_probe_uses_owner_pool_front_door_and_publishes_before_report(
    monkeypatch,
):
    client = FakePahoClient(report_topic="device/SN/report")
    install_client(monkeypatch, client)
    commands = (
        MqttMessage("device/SN/request", b"start"),
        MqttMessage("device/SN/request", b"pushall"),
    )

    result = await probe_mqtt(
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
    assert client.disconnects == 1


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
        "mqtt://user:secret@printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is outcome
    assert result.reason_code == reason_code
    assert client.disconnects == 1


@pytest.mark.asyncio
async def test_probe_classifies_real_paho_reason_code(monkeypatch):
    client = FakePahoClient(reason_code=ReasonCode(PacketTypes.CONNACK, identifier=135))
    install_client(monkeypatch, client)

    result = await probe_mqtt(
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
        "mqtt://printer/?topic=status",
        connect_timeout=0.1,
        report_timeout=0,
        require_report=False,
    )

    assert result.outcome is MqttProbeOutcome.CONNECT_FAILED
    assert result.reason == "paho connection failed"


@pytest.mark.asyncio
async def test_probe_distinguishes_connect_and_required_report_timeouts(monkeypatch):
    silent = FakePahoClient(reason_code=None)
    install_client(monkeypatch, silent)
    connect_result = await probe_mqtt(
        "mqtt://printer/?topic=status",
        connect_timeout=0.001,
        report_timeout=0,
        require_report=True,
    )
    assert connect_result.outcome is MqttProbeOutcome.CONNECT_TIMEOUT

    connected = FakePahoClient()
    install_client(monkeypatch, connected)
    report_result = await probe_mqtt(
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
    probe = asyncio.create_task(
        probe_mqtt(
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

    assert client.disconnects == 1
