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
