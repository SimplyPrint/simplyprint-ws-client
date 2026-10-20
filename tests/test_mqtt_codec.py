"""The pieces under ``AsyncMqtt`` that need no broker: the codec's checks on
what a broker may send, the stream's transport contract, and the reconnect
backoff over an endless outage."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from simplyprint_ws_client.wire import mqtt_codec as codec

pytestmark = pytest.mark.behavior


def test_the_backoff_never_overflows_in_an_endless_outage():
    from simplyprint_ws_client.common.utils.backoff import ExponentialBackoff

    backoff = ExponentialBackoff(base=1.0, maximum=5.0)
    delays = [backoff.delay() for _ in range(5000)]  # 2**1024 used to raise
    assert max(delays) <= 5.0 and min(delays[10:]) >= 2.5


def test_the_stream_takes_any_transport_with_the_interface():
    """uvloop's TCP transport implements the interface without subclassing
    asyncio.Transport; the stream must not insist on the subclass."""
    from simplyprint_ws_client.wire.mqtt_stream import MqttStream

    class DuckTransport:
        def __init__(self) -> None:
            self.written: List[bytes] = []

        def write(self, data: bytes) -> None:
            self.written.append(data)

        def is_closing(self) -> bool:
            return False

    loop = asyncio.new_event_loop()
    try:
        stream = MqttStream(loop)
        transport = DuckTransport()
        stream.connection_made(transport)  # type: ignore[arg-type]
        stream.write(b"\xc0\x00")
        assert transport.written == [b"\xc0\x00"]
    finally:
        loop.close()


@pytest.mark.parametrize(
    "packet",
    [
        bytes((0x20, 2, 0x02, 0)),  # CONNACK with reserved flags
        bytes((0x40, 2, 0, 0)),  # PUBACK for packet id 0
        bytes((0x41, 2, 0, 1)),  # PUBACK with flags set
        bytes((0x30, 5, 0, 1, ord("+"), 0, 0)),  # PUBLISH to a wildcard topic
        bytes((0x10, 0)),  # a CONNECT from the broker
    ],
    ids=["connack-flags", "packet-id-0", "puback-flags", "wildcard-topic", "connect"],
)
def test_packets_a_broker_may_not_send_are_rejected(packet):
    buffer = bytearray(packet)
    with pytest.raises(codec.MalformedPacket):
        _packet, used = codec.split_packet(buffer)
        if _packet is not None and _packet.type == codec.PUBLISH:
            codec.decode_publish(_packet.flags, _packet.body)
        if _packet is not None and _packet.type == codec.CONNACK:
            codec.decode_connack(_packet.body)
        if _packet is not None and _packet.type == codec.PUBACK:
            codec.decode_packet_id(_packet.body)


@pytest.mark.parametrize(
    "packet",
    [
        bytes((0x90, 3, 0, 1, 3)),  # SUBACK return code 3
        bytes((0x50, 2, 0, 0)),  # PUBREC packet id 0
        bytes((0x70, 2, 0, 0)),  # PUBCOMP packet id 0
    ],
    ids=["suback-code-3", "pubrec-id-0", "pubcomp-id-0"],
)
def test_control_packets_are_fully_validated(packet):
    with pytest.raises(codec.MalformedPacket):
        codec.split_packet(bytearray(packet))


def test_pubrel_stays_in_wire_order_with_publishes():
    assert codec.PUBREL in codec.IN_ORDER and codec.PUBLISH in codec.IN_ORDER
    assert codec.PUBACK not in codec.IN_ORDER
