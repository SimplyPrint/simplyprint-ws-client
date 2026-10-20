"""MQTT 3.1.1 packets as bytes, without I/O.

Just what a client of a printer or cloud broker needs: CONNECT/CONNACK,
PUBLISH and its QoS 1/2 acknowledgements, SUBSCRIBE/SUBACK,
UNSUBSCRIBE/UNSUBACK, PINGREQ/PINGRESP and DISCONNECT. Encoding returns one
``bytes`` per packet, so a packet always reaches the socket in one write.
Decoding works on one packet already split off the stream by
:func:`read_packet`.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CONNACK",
    "PUBLISH",
    "PUBACK",
    "PUBREC",
    "PUBREL",
    "PUBCOMP",
    "SUBACK",
    "UNSUBACK",
    "PINGRESP",
    "PINGREQ_PACKET",
    "DISCONNECT_PACKET",
    "MAX_PACKET_SIZE",
    "MalformedPacket",
    "Packet",
    "Publish",
    "encode_connect",
    "encode_publish",
    "encode_subscribe",
    "encode_unsubscribe",
    "encode_ack",
    "decode_connack",
    "decode_publish",
    "decode_packet_id",
    "decode_suback",
    "read_packet",
    "split_packet",
]

# Control packet types (high nibble of the first byte).
CONNECT = 1
CONNACK = 2
PUBLISH = 3
PUBACK = 4
PUBREC = 5
PUBREL = 6
PUBCOMP = 7
SUBSCRIBE = 8
SUBACK = 9
UNSUBSCRIBE = 10
UNSUBACK = 11
PINGREQ = 12
PINGRESP = 13
DISCONNECT = 14

PINGREQ_PACKET = bytes((PINGREQ << 4, 0))
DISCONNECT_PACKET = bytes((DISCONNECT << 4, 0))

#: Largest packet accepted from a broker. Printer reports are tens of KiB; a
#: corrupt length must not make the client allocate the protocol's 256 MiB.
MAX_PACKET_SIZE = 16 * 1024 * 1024


class MalformedPacket(ValueError):
    """The broker sent bytes that are not a valid MQTT 3.1.1 packet."""


@dataclass(frozen=True)
class Packet:
    """One control packet split off the stream: its type, flags and body."""

    type: int
    flags: int
    body: bytes


@dataclass(frozen=True)
class Publish:
    topic: str
    payload: bytes
    qos: int
    retain: bool
    dup: bool
    packet_id: Optional[int]


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    if len(data) > 0xFFFF:
        raise ValueError("MQTT string longer than 65535 bytes")
    return struct.pack("!H", len(data)) + data


def _binary(value: bytes) -> bytes:
    if len(value) > 0xFFFF:
        raise ValueError("MQTT binary field longer than 65535 bytes")
    return struct.pack("!H", len(value)) + value


def _remaining_length(length: int) -> bytes:
    if length > 268_435_455:
        raise ValueError("MQTT packet too large")
    out = bytearray()
    while True:
        byte, length = length % 128, length // 128
        out.append(byte | (0x80 if length else 0))
        if not length:
            return bytes(out)


def _packet(kind: int, flags: int, body: bytes) -> bytes:
    return bytes((kind << 4 | flags,)) + _remaining_length(len(body)) + body


def encode_connect(
    client_id: str,
    *,
    keepalive: int,
    username: Optional[str] = None,
    password: Optional[str] = None,
    clean_session: bool = True,
) -> bytes:
    flags = 0x02 if clean_session else 0
    payload = _string(client_id)
    if username is not None:
        flags |= 0x80
        payload += _string(username)
        if password is not None:
            flags |= 0x40
            payload += _binary(password.encode("utf-8"))
    header = _string("MQTT") + bytes((4, flags)) + struct.pack("!H", keepalive)
    return _packet(CONNECT, 0, header + payload)


def encode_publish(
    topic: str,
    payload: bytes,
    *,
    qos: int = 0,
    retain: bool = False,
    packet_id: Optional[int] = None,
    dup: bool = False,
) -> bytes:
    flags = (0x08 if dup else 0) | (qos << 1) | (0x01 if retain else 0)
    body = _string(topic)
    if qos:
        if packet_id is None:
            raise ValueError("a QoS 1/2 PUBLISH needs a packet id")
        body += struct.pack("!H", packet_id)
    return _packet(PUBLISH, flags, body + bytes(payload))


def encode_subscribe(packet_id: int, topics: Sequence[str], qos: int = 0) -> bytes:
    body = struct.pack("!H", packet_id)
    for topic in topics:
        body += _string(topic) + bytes((qos,))
    return _packet(SUBSCRIBE, 0x02, body)


def encode_unsubscribe(packet_id: int, topics: Iterable[str]) -> bytes:
    body = struct.pack("!H", packet_id) + b"".join(_string(t) for t in topics)
    return _packet(UNSUBSCRIBE, 0x02, body)


def encode_ack(kind: int, packet_id: int) -> bytes:
    """PUBACK, PUBREC, PUBREL (flags 0b0010) or PUBCOMP."""
    return _packet(kind, 0x02 if kind == PUBREL else 0, struct.pack("!H", packet_id))


#: Packets that must be handled in wire order with PUBLISH (a PUBREL ends the
#: QoS 2 exchange of a message that may still be queued); every other packet
#: from a broker is a control that can be handled the moment it arrives.
IN_ORDER = frozenset((PUBLISH, PUBREL))
#: Packet types whose body starts with a packet identifier.
WITH_PACKET_ID = frozenset((PUBACK, PUBREC, PUBREL, PUBCOMP, SUBACK, UNSUBACK))
#: SUBACK return codes: granted QoS 0, 1, 2, or failure.
SUBACK_CODES = frozenset((0, 1, 2, 0x80))

#: Fixed-header flags a broker must use per packet type (PUBLISH's vary).
SERVER_FLAGS = {
    CONNACK: 0,
    PUBACK: 0,
    PUBREC: 0,
    PUBREL: 0x02,
    PUBCOMP: 0,
    SUBACK: 0,
    UNSUBACK: 0,
    PINGRESP: 0,
}
#: Exact body sizes of the fixed-size packets a broker sends.
SERVER_SIZES = {
    CONNACK: 2,
    PUBACK: 2,
    PUBREC: 2,
    PUBREL: 2,
    PUBCOMP: 2,
    UNSUBACK: 2,
    PINGRESP: 0,
}


def validate(packet: Packet) -> None:
    """Reject packets a broker may not send (MQTT 3.1.1, broker to client)."""
    if packet.type == PUBLISH:
        return  # flags and body checked by decode_publish
    if packet.type not in SERVER_FLAGS:
        raise MalformedPacket(f"a broker can't send packet type {packet.type}")
    if packet.flags != SERVER_FLAGS[packet.type]:
        raise MalformedPacket(f"bad flags {packet.flags:#x} on packet type {packet.type}")
    size = SERVER_SIZES.get(packet.type)
    if size is not None and len(packet.body) != size:
        raise MalformedPacket(f"packet type {packet.type} must be {size} bytes")
    if packet.type == SUBACK:
        if len(packet.body) < 3:
            raise MalformedPacket("SUBACK without return codes")
        if not set(packet.body[2:]) <= SUBACK_CODES:
            raise MalformedPacket("SUBACK with an invalid return code")
    if packet.type in WITH_PACKET_ID:
        decode_packet_id(packet.body)  # rejects packet id 0


def decode_connack(body: bytes) -> Tuple[bool, int]:
    """``(session_present, return_code)``."""
    if len(body) != 2:
        raise MalformedPacket("CONNACK must be 2 bytes")
    if body[0] & 0xFE:
        raise MalformedPacket("CONNACK reserved flags set")
    return bool(body[0] & 0x01), body[1]


def decode_packet_id(body: bytes) -> int:
    if len(body) < 2:
        raise MalformedPacket("packet id missing")
    packet_id = struct.unpack_from("!H", body)[0]
    if packet_id == 0:
        raise MalformedPacket("packet id 0")
    return packet_id


def decode_suback(body: bytes) -> Tuple[int, List[int]]:
    """``(packet_id, return_codes)``; 0x80 is a refused subscription."""
    return decode_packet_id(body), list(body[2:])


def decode_publish(flags: int, body: bytes) -> Publish:
    qos = (flags >> 1) & 0x03
    if qos == 3:
        raise MalformedPacket("PUBLISH with QoS 3")
    if len(body) < 2:
        raise MalformedPacket("PUBLISH without a topic")
    (length,) = struct.unpack_from("!H", body)
    end = 2 + length
    if len(body) < end:
        raise MalformedPacket("PUBLISH topic overruns the packet")
    try:
        topic = body[2:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise MalformedPacket("PUBLISH topic is not UTF-8") from error
    if not topic or "+" in topic or "#" in topic or "\x00" in topic:
        raise MalformedPacket(f"invalid PUBLISH topic {topic!r}")
    if qos == 0 and flags & 0x08:
        raise MalformedPacket("DUP set on a QoS 0 PUBLISH")
    packet_id = None
    if qos:
        if len(body) < end + 2:
            raise MalformedPacket("PUBLISH without a packet id")
        (packet_id,) = struct.unpack_from("!H", body, end)
        if packet_id == 0:
            raise MalformedPacket("PUBLISH with packet id 0")
        end += 2
    return Publish(
        topic, bytes(body[end:]), qos, bool(flags & 0x01), bool(flags & 0x08), packet_id
    )


async def read_packet(reader: asyncio.StreamReader) -> Packet:
    """Read the next packet. Raises ``asyncio.IncompleteReadError`` at EOF."""
    first = (await reader.readexactly(1))[0]
    length, multiplier = 0, 1
    for _ in range(4):
        byte = (await reader.readexactly(1))[0]
        length += (byte & 0x7F) * multiplier
        if not byte & 0x80:
            break
        multiplier *= 128
    else:
        raise MalformedPacket("remaining length longer than 4 bytes")
    if length > MAX_PACKET_SIZE:
        raise MalformedPacket(f"packet of {length} bytes exceeds the limit")
    body = await reader.readexactly(length) if length else b""
    return Packet(first >> 4, first & 0x0F, body)


def split_packet(buffer: bytearray, offset: int = 0) -> Tuple[Optional[Packet], int]:
    """The packet starting at ``offset`` in ``buffer``, if it is complete, and
    how many bytes it takes; ``(None, 0)`` while more bytes are needed."""
    available = len(buffer) - offset
    if available < 2:
        return None, 0
    length, multiplier, index = 0, 1, offset + 1
    for _ in range(4):
        if index >= len(buffer):
            return None, 0
        byte = buffer[index]
        index += 1
        length += (byte & 0x7F) * multiplier
        if not byte & 0x80:
            break
        multiplier *= 128
    else:
        raise MalformedPacket("remaining length longer than 4 bytes")
    if length > MAX_PACKET_SIZE:
        raise MalformedPacket(f"packet of {length} bytes exceeds the limit")
    end = index + length
    if end > len(buffer):
        return None, 0
    first = buffer[offset]
    with memoryview(buffer) as view:
        body = bytes(view[index:end])  # one copy, released before the caller resizes
    packet = Packet(first >> 4, first & 0x0F, body)
    validate(packet)
    return packet, end - offset
