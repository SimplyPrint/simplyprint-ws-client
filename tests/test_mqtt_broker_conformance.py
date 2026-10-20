"""The test broker against paho, the reference MQTT client.

If paho can connect, subscribe, publish at QoS 0/1, receive QoS 0/1/2 and
ping through :mod:`tests.mqtt_broker`, then the broker (and the codec it is
built on) speak real MQTT 3.1.1, and the ``AsyncMqtt`` tests that run
against it test a real protocol, not a mirror of the client's own mistakes.
"""

from __future__ import annotations

import asyncio
import ssl
import threading

import pytest
import pytest_asyncio

from simplyprint_ws_client.wire import mqtt_codec as codec
from tests.mqtt_broker import Broker

paho = pytest.importorskip("paho.mqtt.client")

pytestmark = [pytest.mark.behavior, pytest.mark.asyncio]


@pytest_asyncio.fixture(params=[False, True], ids=["tcp", "tls"])
async def broker(request):
    b = await Broker(tls=request.param).start()
    yield b
    await b.stop()


def _paho(broker: Broker):
    client = paho.Client(
        callback_api_version=paho.CallbackAPIVersion.VERSION2,
        protocol=paho.MQTTv311,
        clean_session=True,
    )
    if broker.tls:
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
    client.username_pw_set("bblp", "secret")
    return client


async def test_paho_round_trip(broker: Broker):
    received, connected, subscribed = [], threading.Event(), threading.Event()
    client = _paho(broker)
    client.on_connect = lambda c, u, f, rc, p: connected.set()
    client.on_subscribe = lambda c, u, mid, codes, p: subscribed.set()
    client.on_message = lambda c, u, m: received.append((m.topic, m.payload, m.qos))
    # paho's connect blocks (TCP and TLS handshake): keep it off this loop,
    # which the broker needs to answer.
    client.connect_async("127.0.0.1", broker.port, keepalive=1)
    client.loop_start()
    try:
        await asyncio.to_thread(connected.wait, 5)
        client.subscribe("device/+/report")
        await asyncio.to_thread(subscribed.wait, 5)
        assert broker.connects[0].username == "bblp"
        assert broker.connects[0].password == "secret"
        assert broker.connects[0].keepalive == 1

        client.publish("device/A/request", b"zero", qos=0)
        info = client.publish("device/A/request", b"one", qos=1)
        await asyncio.to_thread(info.wait_for_publish, 5)
        await broker.wait_for(lambda: len(broker.published) == 2)
        assert broker.published == [
            ("device/A/request", b"zero", 0, False),
            ("device/A/request", b"one", 1, False),
        ]

        for qos in (0, 1, 2):
            broker.publish("device/A/report", f"q{qos}".encode(), qos)
        await asyncio.sleep(0.5)
        assert received == [
            ("device/A/report", b"q0", 0),
            ("device/A/report", b"q1", 1),
            ("device/A/report", b"q2", 2),
        ]
        await broker.wait_for(lambda: codec.PUBCOMP in broker.pubacks.values())
        await broker.wait_for(lambda: broker.packets(codec.PINGREQ), timeout=5)
    finally:
        client.disconnect()
        await asyncio.to_thread(client.loop_stop)
