"""Tests for the shared SSDP request parser."""

from simplyprint_ws_client.contrib.discovery import (
    DiscoveredDevice,
    SSDPRequest,
    SSDPRequestParser,
)


_SAMPLE_PACKET = (
    b"NOTIFY * HTTP/1.1\r\n"
    b"HOST: 239.255.255.250:1900\r\n"
    b"Server: Example Device\r\n"
    b"Location: 192.168.1.50\r\n"
    b"NT: urn:example-com:device:printer:1\r\n"
    b"USN: 00M00A000000000\r\n"
    b"Cache-Control: max-age=1800\r\n"
    b"Device-Model: C11\r\n"
    b"Device-Name: MyPrinter\r\n"
    b"\r\n"
)


def test_parse_returns_ssdp_request_with_expected_fields():
    request = SSDPRequestParser.parse(_SAMPLE_PACKET)

    assert isinstance(request, SSDPRequest)
    assert request.method == "NOTIFY"
    assert request.uri == "*"
    assert request.version == "HTTP/1.1"


def test_discovered_device_is_a_neutral_discovery_dto():
    device = DiscoveredDevice(host="192.168.1.50", name="Printer", serial="SN1")

    assert device.host == "192.168.1.50"
    assert device.name == "Printer"
    assert device.serial == "SN1"
    assert device.extra == {}


def test_parse_lowercases_header_names_and_strips_values():
    request = SSDPRequestParser.parse(_SAMPLE_PACKET)

    assert request.headers["host"] == "239.255.255.250:1900"
    assert request.headers["server"] == "Example Device"
    assert request.headers["location"] == "192.168.1.50"
    assert request.headers["nt"] == "urn:example-com:device:printer:1"
    assert request.headers["usn"] == "00M00A000000000"
    assert request.headers["cache-control"] == "max-age=1800"
    assert request.headers["device-model"] == "C11"
    assert request.headers["device-name"] == "MyPrinter"


def test_parse_returns_none_on_non_utf8_payload():
    assert SSDPRequestParser.parse(b"\xff\xfe\xfa") is None


def test_parse_returns_none_on_malformed_request_line():
    assert SSDPRequestParser.parse(b"NOTIFY *\r\n\r\n") is None


def test_header_parsing_stops_at_blank_line():
    headers = SSDPRequestParser._parse_headers(["A: 1", "B: 2", "", "C: 3"])

    assert headers == {"a": "1", "b": "2"}
