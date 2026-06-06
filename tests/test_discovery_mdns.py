import dns.message
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.rrset

from simplyprint_ws_client.contrib.discovery.mdns import (
    MDNSResponse,
    MDNSResponseParser,
)


def _response_wire(records):
    """Build a DNS response packet from (name, rdtype, rdata-text) tuples."""
    msg = dns.message.Message()
    msg.flags = dns.flags.QR | dns.flags.AA
    for name, rdtype, text in records:
        rrset = dns.rrset.from_text(name, 120, dns.rdataclass.IN, rdtype, text)
        msg.answer.append(rrset)
    return msg.to_wire()


def test_build_query_is_a_ptr_question():
    wire = MDNSResponseParser.build_query("_ultimaker._tcp.local")
    msg = dns.message.from_wire(wire)
    assert msg.id == 0
    assert len(msg.question) == 1
    assert msg.question[0].rdtype == dns.rdatatype.PTR
    assert msg.question[0].name.to_text(omit_final_dot=True) == "_ultimaker._tcp.local"


def test_parse_ptr_srv_a_txt():
    wire = _response_wire([
        ("_ultimaker._tcp.local.", "PTR", "ultimaker._ultimaker._tcp.local."),
        ("ultimaker._ultimaker._tcp.local.", "SRV", "0 0 80 um3.local."),
        ("um3.local.", "A", "192.0.2.7"),
        ("ultimaker._ultimaker._tcp.local.", "TXT", '"name=My UM3" "type=ultimaker3"'),
    ])
    response = MDNSResponseParser.parse(wire)
    assert isinstance(response, MDNSResponse)
    assert response.ptr_targets("_ultimaker._tcp.local") == [
        "ultimaker._ultimaker._tcp.local"
    ]
    assert response.srv("ultimaker._ultimaker._tcp.local") == ("um3.local", 80)
    assert response.address_for("um3.local") == "192.0.2.7"
    assert response.txt_for("ultimaker._ultimaker._tcp.local") == {
        "name": "My UM3",
        "type": "ultimaker3",
    }


def test_parse_garbage_returns_none():
    assert MDNSResponseParser.parse(b"not-a-dns-packet") is None


def test_mdns_spec_defaults():
    from simplyprint_ws_client.contrib.discovery.spec import MDNSSpec
    from simplyprint_ws_client.events import Event

    spec = MDNSSpec(
        brand="acme",
        queries=("_acme._tcp.local",),
        event_type=Event,
        mapper=lambda response, addr: None,
        key=lambda record: "x",
    )
    assert spec.group == "224.0.0.251"
    assert spec.port == 5353
    assert spec.query_interval == 30.0
    assert spec.follow_up(object()) == ()  # default: no DNS-SD chaining


import asyncio
import pytest
from simplyprint_ws_client.events import Event, EventBus
from simplyprint_ws_client.events.event import sync_only


@sync_only
class _ProbeEvent(Event):
    ...


def _ultimaker_response():
    return _response_wire([
        ("_ultimaker._tcp.local.", "PTR", "um._ultimaker._tcp.local."),
        ("um._ultimaker._tcp.local.", "SRV", "0 0 80 um.local."),
        ("um.local.", "A", "192.0.2.7"),
    ])


def _single_stage_spec():
    from simplyprint_ws_client.contrib.discovery.spec import MDNSSpec

    def mapper(response, addr):
        for srv in response.srv_records():
            ip = response.address_for(srv.target) or addr[0]
            return {"host": ip, "name": srv.name}
        return None

    return MDNSSpec(
        brand="probe",
        queries=("_ultimaker._tcp.local",),
        event_type=_ProbeEvent,
        mapper=mapper,
        key=lambda record: record["host"],
    )


@pytest.mark.asyncio
async def test_backend_caches_and_emits():
    from simplyprint_ws_client.contrib.discovery.mdns import MDNSDiscoveryBackend

    bus = EventBus()
    received = []
    bus.on(_ProbeEvent, lambda record: received.append(record))

    backend = MDNSDiscoveryBackend(_single_stage_spec(), bus)
    protocol = backend._protocol_factory()
    protocol.datagram_received(_ultimaker_response(), ("192.0.2.7", 5353))
    await asyncio.sleep(0)

    devices = backend.get_devices()
    assert len(devices) == 1
    assert devices[0]["host"] == "192.0.2.7"
    assert len(received) == 1


def _dnssd_stage1():
    return _response_wire([
        ("_services._dns-sd._udp.local.", "PTR", "_acme-AA11._udp.local."),
    ])


def _dnssd_stage2():
    return _response_wire([
        ("_acme-AA11._udp.local.", "PTR", "p._acme-AA11._udp.local."),
        ("p._acme-AA11._udp.local.", "SRV", "0 0 80 p.local."),
        ("p.local.", "A", "192.0.2.9"),
    ])


class _FakeTransport:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))


@pytest.mark.asyncio
async def test_two_stage_follow_up_issues_stage2_query_and_maps():
    from simplyprint_ws_client.contrib.discovery.mdns import MDNSDiscoveryBackend
    from simplyprint_ws_client.contrib.discovery.spec import MDNSSpec

    _DNSSD = "_services._dns-sd._udp.local"
    _PREFIX = "_acme-"

    def follow_up(response):
        return tuple(
            t for t in response.ptr_targets(_DNSSD) if t.startswith(_PREFIX)
        )

    def mapper(response, addr):
        for srv in response.srv_records():
            ip = response.address_for(srv.target) or addr[0]
            return {"host": ip}
        return None

    spec = MDNSSpec(
        brand="probe2",
        queries=(_DNSSD,),
        event_type=_ProbeEvent,
        mapper=mapper,
        key=lambda record: record["host"],
        follow_up=follow_up,
    )

    backend = MDNSDiscoveryBackend(spec, EventBus())
    protocol = backend._protocol_factory()
    transport = _FakeTransport()
    protocol.connection_made(transport)

    # Stage 1 -> backend should issue a stage-2 PTR query for the discovered type.
    protocol.datagram_received(_dnssd_stage1(), ("192.0.2.9", 5353))
    await asyncio.sleep(0)
    assert len(transport.sent) == 1
    sent_query = dns.message.from_wire(transport.sent[0][0])
    assert sent_query.question[0].name.to_text(omit_final_dot=True) == "_acme-AA11._udp.local"

    # Stage 2 -> device is resolved and cached.
    protocol.datagram_received(_dnssd_stage2(), ("192.0.2.9", 5353))
    await asyncio.sleep(0)
    devices = backend.get_devices()
    assert len(devices) == 1
    assert devices[0]["host"] == "192.0.2.9"

    # Re-receiving stage 1 must NOT re-issue the same follow-up (dedup).
    protocol.datagram_received(_dnssd_stage1(), ("192.0.2.9", 5353))
    await asyncio.sleep(0)
    assert len(transport.sent) == 1


@pytest.mark.asyncio
async def test_discovery_service_builds_mdns_backend_and_snapshots():
    from simplyprint_ws_client.contrib.discovery.service import DiscoveryService

    service = DiscoveryService(mdns_specs=[_single_stage_spec()])
    backend = service._mdns["probe"]
    protocol = backend._protocol_factory()
    protocol.datagram_received(_ultimaker_response(), ("192.0.2.7", 5353))
    await asyncio.sleep(0)
    snapshot = service.snapshot("probe")
    assert len(snapshot) == 1
    assert snapshot[0]["host"] == "192.0.2.7"


def test_printer_client_spec_mdns_hook_is_opt_in():
    from simplyprint_ws_client.contrib.spec.client_spec import PrinterClientSpec

    assert PrinterClientSpec.mdns_spec() is None
    assert PrinterClientSpec.provides("mdns_spec") is False

    class _Brand(PrinterClientSpec):
        KEY = "brandx"

        @classmethod
        def mdns_spec(cls):
            return "a-spec"

    assert _Brand.provides("mdns_spec") is True
