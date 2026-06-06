"""Brand-neutral mDNS / DNS-SD discovery.

Parses multicast-DNS responses into neutral records and drives one brand's
:class:`~simplyprint_ws_client.contrib.discovery.spec.MDNSSpec`: periodically
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
import logging
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
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.shared.utils.stoppable import AsyncStoppable

from simplyprint_ws_client.contrib.discovery.spec import MDNSSpec

_DEVICE_TTL = 300


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
            msg = dns.message.from_wire(data)
        except Exception:
            return None

        records: List[MDNSRecord] = []
        for rrset in list(msg.answer) + list(msg.additional):
            rtype = cls._RTYPE.get(rrset.rdtype)
            if rtype is None:
                continue
            name = rrset.name.to_text(omit_final_dot=True)
            for rdata in rrset:
                records.append(cls._record(name, rtype, rdata))

        if not records:
            return None
        return MDNSResponse(tuple(records))

    @staticmethod
    def _record(name: str, rtype: str, rdata) -> MDNSRecord:
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
        self._queried = queried  # set of already-issued follow-up query names
        self._transport = None

    def connection_made(self, transport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        response = MDNSResponseParser.parse(data)
        if response is None:
            return
        asyncio.create_task(self._handle(response, addr))

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
            self._queried.add(name)
            self._transport.sendto(
                MDNSResponseParser.build_query(name),
                (self._spec.group, self._spec.port),
            )

    def error_received(self, exc) -> None:
        if isinstance(exc, OSError) and exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return
        self._logger.error("mdns error for %s", self._spec.brand, exc_info=exc)


class MDNSDiscoveryBackend(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop]
):
    """One always-on mDNS listener (periodic query + response parsing) per brand."""

    def __init__(self, spec: MDNSSpec, event_bus: EventBus) -> None:
        AsyncStoppable.__init__(self)
        EventLoopProvider.__init__(self)

        self.spec = spec
        self.event_bus = event_bus
        self.logger = logging.getLogger("discovery")
        self.devices = ExpiringDict(ttl=_DEVICE_TTL)
        self._emit = event_bus.emit_wrap(spec.event_type, blocking=True)
        self._queried: set = set()

    def get_devices(self) -> list:
        return self.devices.values()

    def _protocol_factory(self) -> _MDNSProtocol:
        return _MDNSProtocol(
            self.spec, self.devices, self._emit, self.logger, self._queried
        )

    def _make_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        if hasattr(socket, "SO_REUSEADDR"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if self.spec.multicast_ttl is not None:
            sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.spec.multicast_ttl
            )
        sock.setblocking(False)
        return sock

    async def _bind(self, sock: socket.socket) -> None:
        # Bind the fixed mDNS port (shared with the system responder via
        # SO_REUSEPORT) so multicast responses and announcements are received;
        # retry briefly while a previous instance tears down.
        while not self.is_stopped():
            try:
                sock.bind(("", self.spec.port))
                return
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    self.logger.warning(
                        "mdns port %s in use - retrying in 5s", self.spec.port
                    )
                    await self.wait(5)
                else:
                    raise

    def _join_group(self, sock: socket.socket) -> None:
        group = socket.inet_aton(self.spec.group)
        mreq = struct.pack("4sL", group, socket.INADDR_ANY)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError:
            self.logger.warning("could not join mdns group for %s", self.spec.brand)

    async def run(self) -> None:
        self.use_running_loop()

        sock = self._make_socket()
        await self._bind(sock)
        if self.is_stopped():
            sock.close()
            return

        self._join_group(sock)
        transport, _ = await self.event_loop.create_datagram_endpoint(
            self._protocol_factory, sock=sock
        )
        self.logger.info(
            "mdns discovery listening for %s on %s:%s",
            self.spec.brand,
            self.spec.group,
            self.spec.port,
        )

        try:
            while not self.is_stopped():
                for name in self.spec.queries:
                    transport.sendto(
                        MDNSResponseParser.build_query(name),
                        (self.spec.group, self.spec.port),
                    )
                await self.wait(self.spec.query_interval)
        finally:
            transport.close()

        self.logger.info("mdns discovery for %s stopping", self.spec.brand)

    def stop(self) -> None:
        if self.is_stopped():
            return
        if self.event_loop_is_running():
            self.event_loop.call_soon_threadsafe(super().stop)
        else:
            super().stop()
