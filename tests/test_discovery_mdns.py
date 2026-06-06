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
