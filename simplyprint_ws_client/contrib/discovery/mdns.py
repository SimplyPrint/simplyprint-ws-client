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

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import dns.flags
import dns.message
import dns.name
import dns.rdataclass
import dns.rdatatype


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
