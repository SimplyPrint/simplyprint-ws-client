"""What a device link's reconnect builds anew and what it reuses: the TLS
context (built once), the TLS session (resumed on the next connect), and
the tasks one connect costs."""

from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest
import yarl

from simplyprint_ws_client.wire import async_mqtt
from simplyprint_ws_client.wire import mqtt_codec as codec
from simplyprint_ws_client.wire.async_mqtt import AsyncMqtt, mqtt_ssl_context
from simplyprint_ws_client.wire.options import TlsClientAuth
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.transport import TransientError


def test_contexts_without_a_client_certificate_are_shared():
    plain = mqtt_ssl_context(verify_tls=False, tls_client_auth=None)

    assert mqtt_ssl_context(verify_tls=False, tls_client_auth=None) is plain
    assert mqtt_ssl_context(verify_tls=True, tls_client_auth=None) is not plain


@pytest.mark.asyncio
async def test_a_transport_builds_its_client_certificate_context_once(monkeypatch):
    built: List[object] = []

    def build(auth):
        built.append(auth)
        return object()

    monkeypatch.setattr(async_mqtt, "client_cert_ssl_context", build)
    auth = TlsClientAuth(ca_pem="ca", cert_pem="cert", key_pem="key")
    transport = AsyncMqtt(yarl.URL("mqtts://printer.local"), tls_client_auth=auth)

    first = transport.ssl_context()

    assert transport.ssl_context() is first
    assert built == [auth]


@pytest.mark.asyncio
async def test_plain_mqtt_has_no_context():
    assert AsyncMqtt(yarl.URL("mqtt://printer.local")).ssl_context() is None


class _FakeStream:
    """The part of ``MqttStream`` a handshake and a close use. The broker
    answers CONNECT with CONNACK (accepted)."""

    opened: List[dict] = []

    def __init__(self, on_control, session) -> None:
        self._on_control = on_control
        self.session = session
        self.tls_session = object()  # what this connection's TLS negotiated
        self.failed: asyncio.Future = asyncio.get_running_loop().create_future()

    @classmethod
    async def open(cls, host, port, *, ssl_context, server_hostname, on_control, session):
        cls.opened.append({"context": ssl_context, "session": session})
        return cls(on_control, session)

    def write(self, data: bytes) -> None:
        if data[0] >> 4 == codec.CONNECT:
            packet = codec.Packet(type=codec.CONNACK, flags=0, body=b"\x00\x00")
            asyncio.get_running_loop().call_soon(self._on_control, packet)

    def is_closing(self) -> bool:
        return False

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    def abort(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


@pytest.fixture
def fake_stream(monkeypatch):
    _FakeStream.opened = []
    monkeypatch.setattr(async_mqtt, "MqttStream", _FakeStream)
    return _FakeStream


@pytest.mark.asyncio
async def test_a_reconnect_reuses_the_context_and_resumes_the_tls_session(
    fake_stream,
):
    transport = AsyncMqtt(yarl.URL("mqtts://printer.local"))

    await transport.open()
    first: Optional[_FakeStream] = transport.connection
    await transport.aclose()
    await transport.open()
    await transport.aclose()

    one, two = fake_stream.opened
    assert one["context"] is two["context"] is transport.ssl_context()
    assert one["session"] is None
    assert two["session"] is first.tls_session


@pytest.mark.asyncio
async def test_plain_mqtt_resumes_nothing(fake_stream):
    transport = AsyncMqtt(yarl.URL("mqtt://printer.local"))

    await transport.open()
    await transport.aclose()
    await transport.open()
    await transport.aclose()

    assert [o["session"] for o in fake_stream.opened] == [None, None]


def _count_tasks(loop: asyncio.AbstractEventLoop) -> List[asyncio.Task]:
    created: List[asyncio.Task] = []

    def factory(loop, coro, **kwargs):
        task = asyncio.Task(coro, loop=loop, **kwargs)
        created.append(task)
        return task

    loop.set_task_factory(factory)
    return created


@pytest.mark.asyncio
async def test_an_mqtt_connect_costs_one_task_its_keepalive(fake_stream):
    loop = asyncio.get_running_loop()
    transport = AsyncMqtt(yarl.URL("mqtts://printer.local"), open_timeout=None)
    created = _count_tasks(loop)
    try:
        await transport.open()
    finally:
        loop.set_task_factory(None)
    await transport.aclose()

    # No reader task to wait for CONNACK: only the keepalive pinger.
    assert len(created) == 1


class _Wire(Reconnecting):
    async def open(self) -> None:
        pass

    async def recv(self) -> Optional[object]:
        await asyncio.Event().wait()
        return None

    async def write(self, message: object) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_consuming_a_connection_costs_one_task_and_a_trip_still_ends_it():
    loop = asyncio.get_running_loop()
    wire = _Wire(yarl.URL("ws://printer.local"))
    wire._tripped = loop.create_future()
    consuming = asyncio.ensure_future(wire.consume())
    created = _count_tasks(loop)
    try:
        await asyncio.sleep(0)  # consume starts its stream
        wire.trip(wire.generation, TransientError("send failed"))
        with pytest.raises(TransientError):
            await asyncio.wait_for(consuming, timeout=2.0)
    finally:
        loop.set_task_factory(None)

    assert len(created) == 1  # the stream; the tripwire is a plain future
