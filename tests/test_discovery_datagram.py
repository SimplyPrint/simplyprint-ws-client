"""The raw datagram family: one payload out, each raw reply to a parser.

Brand-neutral. The collector and the always-on protocol are I/O-free; the
sweep is pinned once on loopback against an in-loop responder.
"""

import asyncio

import pytest

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.scope import Scope
from simplyprint_ws_client.integration.discovery.datagram import (
    DatagramCollector,
    datagram_listener,
    sweep,
)
from simplyprint_ws_client.integration.discovery.listener import (
    DatagramReceived,
    ListenerStarted,
    ListenerTick,
    SendTo,
    Sighted,
    WakeAfter,
)
from simplyprint_ws_client.integration.discovery.service import DiscoveryService
from simplyprint_ws_client.integration.discovery.spec import DatagramSpec
from simplyprint_ws_client.integration.spec import (
    IntegrationId,
    IntegrationSpec,
    ProductMetadata,
)


def _parse(data: bytes, source):
    text = data.decode()
    if text == "boom":
        raise ValueError("unreadable")
    if not text.startswith("id="):
        return None
    return {"id": text[3:], "host": source[0]}


def _spec(**kwargs) -> DatagramSpec:
    fields = dict(
        brand="alpha",
        ports=(3000,),
        parse=_parse,
        key=lambda record: record["id"],
        payload=b"M99999",
    )
    fields.update(kwargs)
    return DatagramSpec(**fields)


# -- the spec --------------------------------------------------------------------


def test_a_payload_without_interval_is_on_demand_only():
    assert _spec().always_on is False
    assert _spec(interval=10.0).always_on is True
    assert _spec(payload=None, listen_port=3001).always_on is True


# -- the collector ---------------------------------------------------------------


def test_collector_parses_each_reply_with_its_source_and_keeps_the_latest_per_key():
    collector = DatagramCollector(_spec())
    collector.feed(b"id=A", ("10.0.0.5", 3000))
    collector.feed(b"id=A", ("10.0.0.9", 3000))
    collector.feed(b"id=B", ("10.0.0.6", 3000))
    collector.feed(b"noise", ("10.0.0.7", 3000))

    assert collector.records() == [
        {"id": "A", "host": "10.0.0.9"},
        {"id": "B", "host": "10.0.0.6"},
    ]


def test_collector_treats_a_raising_parser_or_key_as_no_record():
    collector = DatagramCollector(_spec(key=lambda record: record["missing"]))
    assert collector.feed(b"boom", ("10.0.0.5", 3000)) is None
    assert collector.feed(b"id=A", ("10.0.0.5", 3000)) is None
    assert collector.records() == []


def test_collector_sends_the_payload_to_every_port_and_nothing_when_listen_only():
    sends = DatagramCollector(
        _spec(ports=(8899, 48899), address="10.0.0.255")
    ).sends()
    assert sends == [
        SendTo(b"M99999", "10.0.0.255", 8899),
        SendTo(b"M99999", "10.0.0.255", 48899),
    ]
    assert DatagramCollector(_spec(payload=None)).sends() == []


# -- the always-on listener ------------------------------------------------------


def test_listener_sends_a_round_and_wakes_after_the_interval():
    listener = datagram_listener(_spec(interval=15.0))
    protocol = listener.protocol()

    expected = [SendTo(b"M99999", "255.255.255.255", 3000), WakeAfter(15.0)]
    assert list(protocol.handle(ListenerStarted(0.0))) == expected
    assert list(protocol.handle(ListenerTick(15.0))) == expected


def test_listener_sights_parsed_replies_only():
    protocol = datagram_listener(_spec(interval=15.0)).protocol()

    assert list(
        protocol.handle(DatagramReceived(b"id=A", ("10.0.0.5", 3000), 1.0))
    ) == [Sighted({"id": "A", "host": "10.0.0.5"})]
    assert list(protocol.handle(DatagramReceived(b"noise", ("10.0.0.5", 3000), 1.0))) == []
    assert list(protocol.handle(DatagramReceived(b"boom", ("10.0.0.5", 3000), 1.0))) == []


def test_a_listen_only_listener_sends_nothing_and_never_wakes():
    protocol = datagram_listener(_spec(payload=None, listen_port=3001)).protocol()
    assert list(protocol.handle(ListenerStarted(0.0))) == []


def test_listener_spec_carries_the_socket_shape():
    spec = _spec(interval=5.0, listen_port=3001, group="239.0.0.1", multicast_ttl=4)
    listener = datagram_listener(spec)

    assert listener.brand == "alpha"
    assert listener.port == 3001
    assert listener.group == "239.0.0.1"
    assert listener.broadcast is True
    assert listener.multicast_ttl == 4
    assert listener.key({"id": "A"}) == "A"


# -- the service -----------------------------------------------------------------


_METADATA = ProductMetadata(
    display_name="Alpha",
    image_url="/alpha.png",
    supported_transports=(),
    capabilities=(),
)


def _integration(datagram: DatagramSpec) -> IntegrationSpec:
    return IntegrationSpec(
        id=IntegrationId(datagram.brand),
        client_factory=lambda config, *, context: None,
        config_factory=PrinterConfig,
        metadata=_METADATA,
        datagram=datagram,
    )


def test_an_on_demand_family_registers_a_sweep_and_no_listener():
    service = DiscoveryService()
    scope = Scope("alpha")
    service.add(_integration(_spec()), scope)

    assert service.datagram_keys() == ["alpha"]
    assert service.backend_keys() == []
    assert scope.close("unload").ok
    assert service.datagram_keys() == []


def test_an_always_on_family_also_runs_a_listener_until_the_scope_closes():
    service = DiscoveryService()
    scope = Scope("alpha")
    service.add(_integration(_spec(interval=10.0)), scope)

    assert service.datagram_keys() == ["alpha"]
    assert service.backend_keys() == ["alpha"]
    assert scope.close("unload").ok
    assert service.datagram_keys() == []
    assert service.backend_keys() == []


def test_a_taken_datagram_key_is_rejected():
    service = DiscoveryService()
    service.add(_integration(_spec()), Scope("a"))
    with pytest.raises(ValueError, match="duplicate datagram discovery spec: alpha"):
        service.add(_integration(_spec()), Scope("b"))


@pytest.mark.asyncio
async def test_sweeping_an_unknown_brand_finds_nothing():
    assert await DiscoveryService().sweep("nobody", timeout=0.0) == []


# -- the sweep, on loopback ------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_sends_once_and_collects_every_reply():
    loop = asyncio.get_running_loop()
    heard = []

    class _Printer(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            heard.append(data)
            self.transport.sendto(b"noise", addr)
            self.transport.sendto(b"id=P1", addr)

    transport, _ = await loop.create_datagram_endpoint(
        _Printer, local_addr=("127.0.0.1", 0)
    )
    try:
        port = transport.get_extra_info("sockname")[1]
        spec = _spec(address="127.0.0.1", ports=(port,))
        records = await sweep(spec, timeout=0.5)
    finally:
        transport.close()

    assert heard == [b"M99999"]
    assert records == [{"id": "P1", "host": "127.0.0.1"}]
