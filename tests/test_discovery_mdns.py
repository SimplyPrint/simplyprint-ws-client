import asyncio

import dns.message
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import pytest

from simplyprint_ws_client.events import Event, EventBus
from simplyprint_ws_client.events.event import sync_only

from simplyprint_ws_client.integration.discovery.mdns import (
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
    wire = _response_wire(
        [
            ("_ultimaker._tcp.local.", "PTR", "ultimaker._ultimaker._tcp.local."),
            ("ultimaker._ultimaker._tcp.local.", "SRV", "0 0 80 um3.local."),
            ("um3.local.", "A", "192.0.2.7"),
            (
                "ultimaker._ultimaker._tcp.local.",
                "TXT",
                '"name=My UM3" "type=ultimaker3"',
            ),
        ]
    )
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


def _set_cache_flush(data):
    """Set the mDNS cache-flush bit (top of CLASS) on every RR — like a real
    mDNS responder does, which a plain DNS parser turns into untyped rdata."""
    import struct

    qd, an, ns, ar = struct.unpack_from("!4H", data, 4)
    buf = bytearray(data)
    end = len(buf)

    def skip_name(p):
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
        pos = skip_name(pos) + 4
    for _ in range(an + ns + ar):
        pos = skip_name(pos)
        cls = struct.unpack_from("!H", buf, pos + 2)[0]
        struct.pack_into("!H", buf, pos + 2, cls | 0x8000)
        rdlen = struct.unpack_from("!H", buf, pos + 8)[0]
        pos += 10 + rdlen
    return bytes(buf)


def test_parse_normalises_cache_flush_bit():
    # Real responders set the cache-flush bit; the parser must still return typed
    # records (regression for the GenericRdata AttributeError crash on live LANs).
    wire = _set_cache_flush(
        _response_wire(
            [
                ("svc._x._tcp.local.", "SRV", "0 0 80 host.local."),
                ("host.local.", "A", "192.0.2.5"),
            ]
        )
    )
    response = MDNSResponseParser.parse(wire)
    assert response is not None
    assert response.srv("svc._x._tcp.local") == ("host.local", 80)
    assert response.address_for("host.local") == "192.0.2.5"


def test_mdns_spec_defaults():
    from simplyprint_ws_client.integration.discovery.spec import MDNSSpec
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


@sync_only
class _ProbeEvent(Event): ...


def _ultimaker_response():
    return _response_wire(
        [
            ("_ultimaker._tcp.local.", "PTR", "um._ultimaker._tcp.local."),
            ("um._ultimaker._tcp.local.", "SRV", "0 0 80 um.local."),
            ("um.local.", "A", "192.0.2.7"),
        ]
    )


def _single_stage_spec():
    from simplyprint_ws_client.integration.discovery.spec import MDNSSpec

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
    from simplyprint_ws_client.integration.discovery.mdns import MDNSDiscoveryBackend

    bus = EventBus()
    received = []
    bus.on(_ProbeEvent, lambda record: received.append(record))

    backend = MDNSDiscoveryBackend(_single_stage_spec(), bus)
    protocol = backend.protocol_factory()
    protocol.datagram_received(_ultimaker_response(), ("192.0.2.7", 5353))
    await asyncio.sleep(0)

    devices = backend.get_devices()
    assert len(devices) == 1
    assert devices[0]["host"] == "192.0.2.7"
    assert len(received) == 1


def _dnssd_stage1():
    return _response_wire(
        [
            ("_services._dns-sd._udp.local.", "PTR", "_acme-AA11._udp.local."),
        ]
    )


def _dnssd_stage2():
    return _response_wire(
        [
            ("_acme-AA11._udp.local.", "PTR", "p._acme-AA11._udp.local."),
            ("p._acme-AA11._udp.local.", "SRV", "0 0 80 p.local."),
            ("p.local.", "A", "192.0.2.9"),
        ]
    )


class _FakeTransport:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))


@pytest.mark.asyncio
async def test_two_stage_follow_up_issues_stage2_query_and_maps():
    from simplyprint_ws_client.integration.discovery.mdns import MDNSDiscoveryBackend
    from simplyprint_ws_client.integration.discovery.spec import MDNSSpec

    _DNSSD = "_services._dns-sd._udp.local"
    _PREFIX = "_acme-"

    def follow_up(response):
        return tuple(t for t in response.ptr_targets(_DNSSD) if t.startswith(_PREFIX))

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
    protocol = backend.protocol_factory()
    transport = _FakeTransport()
    protocol.connection_made(transport)

    # Stage 1 -> backend should issue a stage-2 PTR query for the discovered type.
    protocol.datagram_received(_dnssd_stage1(), ("192.0.2.9", 5353))
    await asyncio.sleep(0)
    assert len(transport.sent) == 1
    sent_query = dns.message.from_wire(transport.sent[0][0])
    assert (
        sent_query.question[0].name.to_text(omit_final_dot=True)
        == "_acme-AA11._udp.local"
    )

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
    from simplyprint_ws_client.integration.discovery.service import DiscoveryService

    service = DiscoveryService(mdns_specs=[_single_stage_spec()])
    backend = service._mdns["probe"]
    protocol = backend.protocol_factory()
    protocol.datagram_received(_ultimaker_response(), ("192.0.2.7", 5353))
    await asyncio.sleep(0)
    snapshot = service.snapshot("probe")
    assert len(snapshot) == 1
    assert snapshot[0]["host"] == "192.0.2.7"


def test_discovery_service_merges_multicast_and_mdns_for_one_brand():
    from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice
    from simplyprint_ws_client.integration.discovery.service import DiscoveryService
    from simplyprint_ws_client.integration.discovery.spec import MDNSSpec, MulticastSpec

    multicast = MulticastSpec(
        brand="probe",
        group="239.255.255.250",
        port=1900,
        event_type=_ProbeEvent,
        mapper=lambda request, addr: None,
        key=lambda device: device.hardware_identity() or device.host,
    )
    mdns = MDNSSpec(
        brand="probe",
        queries=("_octoprint._tcp.local",),
        event_type=_ProbeEvent,
        mapper=lambda response, addr: None,
        key=lambda device: device.hardware_identity() or device.host,
    )
    service = DiscoveryService(multicast_specs=[multicast], mdns_specs=[mdns])
    service._multicast["probe"].devices["192.0.2.7"] = DiscoveredDevice(
        host="192.0.2.7", name="OctoPrint", extra={"source": "ssdp"}
    )
    service._mdns["probe"].devices["uuid-7"] = DiscoveredDevice(
        host="192.0.2.7",
        hardware_id="uuid-7",
        extra={"model": "Raspberry Pi"},
    )

    assert service.snapshot("probe") == [
        DiscoveredDevice(
            host="192.0.2.7",
            name="OctoPrint",
            hardware_id="uuid-7",
            extra={"model": "Raspberry Pi", "source": "ssdp"},
        )
    ]


def test_discovery_service_keeps_distinct_hardware_identities():
    from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice
    from simplyprint_ws_client.integration.discovery.service import DiscoveryService
    from simplyprint_ws_client.integration.discovery.spec import MDNSSpec, MulticastSpec

    multicast = MulticastSpec(
        brand="probe",
        group="239.255.255.250",
        port=1900,
        event_type=_ProbeEvent,
        mapper=lambda request, addr: None,
        key=lambda device: device.hardware_identity() or device.host,
    )
    mdns = MDNSSpec(
        brand="probe",
        queries=("_octoprint._tcp.local",),
        event_type=_ProbeEvent,
        mapper=lambda response, addr: None,
        key=lambda device: device.hardware_identity() or device.host,
    )
    service = DiscoveryService(multicast_specs=[multicast], mdns_specs=[mdns])
    service._multicast["probe"].devices["uuid-a"] = DiscoveredDevice(
        host="192.0.2.7", hardware_id="uuid-a"
    )
    service._mdns["probe"].devices["uuid-b"] = DiscoveredDevice(
        host="192.0.2.7", hardware_id="uuid-b"
    )

    assert len(service.snapshot("probe")) == 2


def test_integration_spec_mdns_field_is_nullable():
    from simplyprint_ws_client.core.client import Client
    from simplyprint_ws_client.core.config import PrinterConfig
    from simplyprint_ws_client.integration.spec import (
        IntegrationId,
        IntegrationSpec,
        ProductMetadata,
    )

    metadata = ProductMetadata(
        display_name="Brand",
        image_url="/brand.png",
        supported_transports=(),
        capabilities=(),
    )
    base = IntegrationSpec(
        id=IntegrationId("brandx"),
        client_factory=Client,
        config_factory=PrinterConfig,
        metadata=metadata,
    )
    configured = IntegrationSpec(
        id=IntegrationId("brandy"),
        client_factory=Client,
        config_factory=PrinterConfig,
        metadata=metadata,
        mdns="a-spec",
    )

    assert base.mdns is None
    assert configured.mdns == "a-spec"
