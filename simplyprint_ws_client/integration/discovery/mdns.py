"""Brand-neutral mDNS / DNS-SD discovery.

Parses multicast-DNS responses into neutral records and drives one brand's
:class:`~simplyprint_ws_client.integration.discovery.spec.MDNSSpec`: periodically
multicast a set of PTR queries, parse every response, optionally chain follow-up
queries (DNS-SD service-type enumeration), map matches to a brand device record,
cache with a TTL, and emit the spec's event. No brand knowledge lives here --
query names, the follow-up rule, the record mapping and the cache key all come
from the spec. The wire format is generic DNS (RFC 1035 / 6762 / 6763); the only
constants are the standard mDNS group/port, carried by the spec.
"""

from __future__ import annotations

import asyncio
import errno
import socket
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import dns.flags
import dns.message
import dns.name
import dns.rdataclass
import dns.rdatatype

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.integration.discovery.multicast_base import (
    DEVICE_TTL,
    MulticastListenerBase,
)
from simplyprint_ws_client.integration.discovery.spec import MDNSSpec


def _clear_cache_flush(data: bytes) -> bytes:
    """Normalise the mDNS cache-flush / unicast-response bit out of the wire.

    Responders set the top bit of each resource record's CLASS field (so ``IN``
    becomes ``0x8001``); a plain DNS parser treats that as an unknown class and
    returns untyped generic rdata, losing the ``target``/``address`` fields. This
    walks the message and clears that bit on every RR class so the datagram parses
    as class ``IN`` (with proper name decompression). OPT records are left alone --
    their CLASS field is a UDP payload size, not a class. Returns the input
    unchanged if the structure can't be walked.
    """
    if len(data) < 12:
        return data
    try:
        qd, an, ns, ar = struct.unpack_from("!4H", data, 4)
    except struct.error:
        return data

    buf = bytearray(data)
    end = len(buf)

    def skip_name(p: int) -> int:
        while p < end:
            length = buf[p]
            if length == 0:
                return p + 1
            if length & 0xC0 == 0xC0:
                return p + 2
            p += length + 1
        return p

    pos = 12
    for _ in range(qd):
        pos = skip_name(pos) + 4  # qtype + qclass
        if pos > end:
            return data
    for _ in range(an + ns + ar):
        pos = skip_name(pos)
        if pos + 10 > end:
            return data
        rrtype = struct.unpack_from("!H", buf, pos)[0]
        if rrtype != 41:  # not OPT (whose CLASS field is the UDP payload size)
            rrclass = struct.unpack_from("!H", buf, pos + 2)[0]
            if rrclass & 0x8000:
                struct.pack_into("!H", buf, pos + 2, rrclass & 0x7FFF)
        rdlen = struct.unpack_from("!H", buf, pos + 8)[0]
        pos += 10 + rdlen
        if pos > end:
            return data
    return bytes(buf)


@dataclass(frozen=True)
class MDNSRecord:
    name: str
    rtype: str
    target: Optional[str] = None
    address: Optional[str] = None
    port: Optional[int] = None
    txt: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MDNSResponse:
    """A parsed multicast-DNS message flattened to neutral records.

    The helpers let a brand mapper resolve a PTR -> SRV -> A/TXT chain without
    touching the DNS wire library directly.
    """

    records: Tuple[MDNSRecord, ...]

    def ptr_targets(self, name: str) -> List[str]:
        return [
            r.target
            for r in self.records
            if r.rtype == "PTR" and r.name == name and r.target
        ]

    def all_ptr_targets(self) -> List[str]:
        return [r.target for r in self.records if r.rtype == "PTR" and r.target]

    def srv(self, name: str) -> Optional[Tuple[str, int]]:
        for r in self.records:
            if r.rtype == "SRV" and r.name == name and r.target and r.port is not None:
                return (r.target, r.port)
        return None

    def srv_records(self) -> List[MDNSRecord]:
        return [r for r in self.records if r.rtype == "SRV"]

    def address_for(self, target: str) -> Optional[str]:
        for r in self.records:
            if r.rtype == "A" and r.name == target and r.address:
                return r.address
        return None

    def txt_for(self, name: str) -> Dict[str, str]:
        for r in self.records:
            if r.rtype == "TXT" and r.name == name:
                return dict(r.txt)
        return {}


class MDNSResponseParser:
    """Parses DNS response datagrams and builds PTR query datagrams."""

    _RTYPE = {
        dns.rdatatype.PTR: "PTR",
        dns.rdatatype.SRV: "SRV",
        dns.rdatatype.A: "A",
        dns.rdatatype.TXT: "TXT",
    }

    @classmethod
    def parse(cls, data: bytes) -> Optional[MDNSResponse]:
        try:
            msg = dns.message.from_wire(_clear_cache_flush(data))
        except Exception:
            return None

        records: List[MDNSRecord] = []
        for rrset in list(msg.answer) + list(msg.additional):
            rtype = cls._RTYPE.get(rrset.rdtype)
            if rtype is None:
                continue
            name = rrset.name.to_text(omit_final_dot=True)
            for rdata in rrset:
                record = cls._record(name, rtype, rdata)
                if record is not None:
                    records.append(record)

        if not records:
            return None
        return MDNSResponse(tuple(records))

    @staticmethod
    def _record(name: str, rtype: str, rdata) -> Optional[MDNSRecord]:
        # A record whose mDNS cache-flush class bit could not be normalised parses
        # as an untyped generic rdata (no ``target``/``address``); skip it rather
        # than crash the whole datagram.
        try:
            if rtype == "PTR":
                return MDNSRecord(
                    name, rtype, target=rdata.target.to_text(omit_final_dot=True)
                )
            if rtype == "SRV":
                return MDNSRecord(
                    name,
                    rtype,
                    target=rdata.target.to_text(omit_final_dot=True),
                    port=rdata.port,
                )
            if rtype == "A":
                return MDNSRecord(name, rtype, address=rdata.address)
            if rtype == "TXT":
                txt: Dict[str, str] = {}
                for chunk in rdata.strings:
                    text = chunk.decode("utf-8", "replace")
                    if "=" in text:
                        key, value = text.split("=", 1)
                        txt[key] = value
                    else:
                        txt[text] = ""
                return MDNSRecord(name, rtype, txt=txt)
        except AttributeError:
            return None
        return MDNSRecord(name, rtype)

    @staticmethod
    def build_query(name: str) -> bytes:
        query = dns.message.make_query(
            dns.name.from_text(name), dns.rdatatype.PTR, dns.rdataclass.IN
        )
        query.id = 0
        query.flags = 0  # standard mDNS query (no recursion-desired bit)
        return query.to_wire()


class _MDNSProtocol(asyncio.DatagramProtocol):
    """Parses datagrams, chains DNS-SD follow-ups, caches devices, emits events."""

    def __init__(self, spec: MDNSSpec, devices: ExpiringDict, emit, logger, queried):
        self._spec = spec
        self._devices = devices
        self._emit = emit  # async callable(record) | None
        self._logger = logger
        self._queried = queried  # TTL'd set of already-issued follow-up names
        self._transport = None
        #: In-flight datagram handlers; retained because asyncio holds tasks
        #: weakly and an unreferenced one can be collected mid-flight.
        self._handle_tasks: "set[asyncio.Task]" = set()

    def connection_made(self, transport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        response = MDNSResponseParser.parse(data)
        if response is None:
            return
        task = asyncio.create_task(self._handle(response, addr))
        self._handle_tasks.add(task)
        task.add_done_callback(self._handle_tasks.discard)

    async def _handle(self, response, addr) -> None:
        self._chain_follow_ups(response)

        try:
            record = self._spec.mapper(response, addr)
        except Exception:
            self._logger.debug(
                "mdns mapper failed for %s", self._spec.brand, exc_info=True
            )
            return

        if record is None:
            return

        self._devices[self._spec.key(record)] = record
        if self._emit is not None:
            await self._emit(record)

    def _chain_follow_ups(self, response) -> None:
        try:
            names = self._spec.follow_up(response)
        except Exception:
            self._logger.debug(
                "mdns follow_up failed for %s", self._spec.brand, exc_info=True
            )
            return
        for name in names:
            if name in self._queried or self._transport is None:
                continue
            self._queried[name] = True
            self._transport.sendto(
                MDNSResponseParser.build_query(name),
                (self._spec.group, self._spec.port),
            )

    def error_received(self, exc) -> None:
        if isinstance(exc, OSError) and exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return
        self._logger.error("mdns error for %s", self._spec.brand, exc_info=exc)


class MDNSDiscoveryBackend(MulticastListenerBase):
    """One always-on mDNS listener (periodic query + response parsing) per brand."""

    spec: MDNSSpec

    _listen_label = "mdns discovery"
    _port_label = "mdns"

    def __init__(self, spec: MDNSSpec, event_bus: EventBus) -> None:
        super().__init__(spec, event_bus)
        # TTL'd like the device cache: a name can be re-queried once its
        # entry expires, and the set cannot grow for process lifetime.
        self._queried = ExpiringDict(ttl=DEVICE_TTL)

    def _protocol_factory(self) -> _MDNSProtocol:
        return _MDNSProtocol(
            self.spec, self.devices, self._emit, self.logger, self._queried
        )

    async def _bind(self, sock: socket.socket) -> None:
        # Bind the fixed mDNS port (shared with the system responder via
        # SO_REUSEPORT) so multicast responses and announcements are received;
        # retry briefly while a previous instance tears down.
        await self._bind_fixed_port(sock)

    def _join_group(self, sock: socket.socket) -> None:
        try:
            self._join_group_mreq(sock)
        except OSError:
            self.logger.warning("could not join mdns group for %s", self.spec.brand)

    async def _run_loop(self, transport: asyncio.DatagramTransport) -> None:
        while not self.is_stopped():
            for name in self.spec.queries:
                transport.sendto(
                    MDNSResponseParser.build_query(name),
                    (self.spec.group, self.spec.port),
                )
            await self.wait(self.spec.query_interval)
