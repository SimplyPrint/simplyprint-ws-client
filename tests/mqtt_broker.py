"""A small MQTT 3.1.1 broker on real loopback sockets, for tests.

Enough of a broker to drive a client through everything printers and cloud
brokers do: CONNECT (with a chosen CONNACK code), SUBSCRIBE (optionally
refusing topics), UNSUBSCRIBE, QoS 0/1 publishes from the client, QoS 0/1/2
publishes to the client, PINGREQ, and abrupt drops. Optional TLS with a
throwaway self-signed certificate. ``tests/test_mqtt_broker_conformance.py``
checks it against paho, the reference client.
"""

from __future__ import annotations

import asyncio
import ssl
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from simplyprint_ws_client.wire import mqtt_codec as codec
from simplyprint_ws_client.wire.transport import topic_matches


def self_signed_context() -> ssl.SSLContext:
    directory = Path(tempfile.mkdtemp())
    key, cert = directory / "key.pem", directory / "cert.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "1",
            "-subj", "/CN=printer.local",
        ],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


def _packet(kind: int, flags: int, body: bytes) -> bytes:
    return codec._packet(kind, flags, body)


@dataclass
class Session:
    writer: asyncio.StreamWriter
    client_id: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    keepalive: int = 0
    subscriptions: Set[str] = field(default_factory=set)
    pubrel_waiting: Set[int] = field(default_factory=set)

    def send(self, data: bytes) -> None:
        if not self.writer.is_closing():
            self.writer.write(data)


class Broker:
    def __init__(self, *, tls: bool = False) -> None:
        self.tls = tls
        self.server: Optional[asyncio.AbstractServer] = None
        self.port = 0
        self.sessions: List[Session] = []
        #: Every packet clients sent: (type, flags, body).
        self.received: List[Tuple[int, int, bytes]] = []
        #: PUBLISHes from clients: (topic, payload, qos, retain).
        self.published: List[Tuple[str, bytes, int, bool]] = []
        self.connects: List[Session] = []
        self.connack_code = 0
        self.reject_empty_client_id = False
        self.refuse_topics: Set[str] = set()
        self.answer_pings = True
        self.ack_publishes = True
        self.pubacks: Dict[int, int] = {}
        self._next_id = 0
        self.changed = asyncio.Event()

    async def start(self) -> "Broker":
        context = self_signed_context() if self.tls else None
        self.server = await asyncio.start_server(
            self._serve, "127.0.0.1", 0, ssl=context
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    @property
    def url(self) -> str:
        scheme = "mqtts" if self.tls else "mqtt"
        return f"{scheme}://bblp:secret@127.0.0.1:{self.port}"

    async def stop(self) -> None:
        self.drop_all()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    def drop_all(self) -> None:
        for session in self.sessions:
            session.writer.transport.abort()
        self.sessions.clear()

    def publish(self, topic: str, payload: bytes, qos: int = 0, *, dup: bool = False,
                packet_id: Optional[int] = None) -> None:
        for session in self.sessions:
            if any(topic_matches(s, topic) for s in session.subscriptions):
                if qos:
                    if packet_id is None:
                        self._next_id = self._next_id % 0xFFFF + 1
                        pid = self._next_id
                    else:
                        pid = packet_id
                else:
                    pid = None
                session.send(
                    codec.encode_publish(topic, payload, qos=qos, packet_id=pid, dup=dup)
                )

    async def wait_for(self, predicate, timeout: float = 5.0) -> None:
        async def wait() -> None:
            while not predicate():
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(wait(), timeout)

    def _note(self) -> None:
        self.changed.set()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        session = Session(writer)
        try:
            packet = await codec.read_packet(reader)
            self.received.append((packet.type, packet.flags, packet.body))
            if packet.type != codec.CONNECT:
                return
            self._connect(session, packet.body)
            self.connects.append(session)
            code = self.connack_code
            if self.reject_empty_client_id and not session.client_id:
                code = 2
            session.send(_packet(codec.CONNACK, 0, bytes((0, code))))
            self._note()
            if code:
                await writer.drain()
                return
            self.sessions.append(session)
            while True:
                packet = await codec.read_packet(reader)
                self.received.append((packet.type, packet.flags, packet.body))
                self._handle(session, packet)
                self._note()
        except (asyncio.IncompleteReadError, ConnectionError, ssl.SSLError):
            pass
        finally:
            if session in self.sessions:
                self.sessions.remove(session)
            writer.close()
            self._note()

    def _connect(self, session: Session, body: bytes) -> None:
        offset = 0

        def string() -> bytes:
            nonlocal offset
            (length,) = struct.unpack_from("!H", body, offset)
            value = body[offset + 2 : offset + 2 + length]
            offset += 2 + length
            return value

        assert string() == b"MQTT"
        level, flags = body[offset], body[offset + 1]
        assert level == 4
        (session.keepalive,) = struct.unpack_from("!H", body, offset + 2)
        offset += 4
        session.client_id = string().decode()
        if flags & 0x80:
            session.username = string().decode()
        if flags & 0x40:
            session.password = string().decode()

    def _handle(self, session: Session, packet: codec.Packet) -> None:
        kind = packet.type
        if kind == codec.SUBSCRIBE:
            packet_id = codec.decode_packet_id(packet.body)
            offset, codes = 2, []
            while offset < len(packet.body):
                (length,) = struct.unpack_from("!H", packet.body, offset)
                topic = packet.body[offset + 2 : offset + 2 + length].decode()
                offset += 2 + length + 1
                refused = topic in self.refuse_topics
                if not refused:
                    session.subscriptions.add(topic)
                codes.append(0x80 if refused else 0)
            session.send(
                _packet(codec.SUBACK, 0, struct.pack("!H", packet_id) + bytes(codes))
            )
        elif kind == codec.UNSUBSCRIBE:
            packet_id = codec.decode_packet_id(packet.body)
            offset = 2
            while offset < len(packet.body):
                (length,) = struct.unpack_from("!H", packet.body, offset)
                session.subscriptions.discard(
                    packet.body[offset + 2 : offset + 2 + length].decode()
                )
                offset += 2 + length
            session.send(_packet(codec.UNSUBACK, 0, struct.pack("!H", packet_id)))
        elif kind == codec.PUBLISH:
            publish = codec.decode_publish(packet.flags, packet.body)
            self.published.append(
                (publish.topic, publish.payload, publish.qos, publish.retain)
            )
            if publish.qos == 1 and self.ack_publishes:
                session.send(codec.encode_ack(codec.PUBACK, publish.packet_id))
        elif kind == codec.PUBREC:
            packet_id = codec.decode_packet_id(packet.body)
            session.send(codec.encode_ack(codec.PUBREL, packet_id))
        elif kind in (codec.PUBACK, codec.PUBCOMP):
            packet_id = codec.decode_packet_id(packet.body)
            self.pubacks[packet_id] = kind
        elif kind == codec.PINGREQ:
            if self.answer_pings:
                session.send(bytes((codec.PINGRESP << 4, 0)))
        elif kind == codec.DISCONNECT:
            session.writer.close()

    def packets(self, kind: int) -> List[Tuple[int, int, bytes]]:
        return [p for p in self.received if p[0] == kind]
