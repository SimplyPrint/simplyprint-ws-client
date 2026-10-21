"""Brand-owned always-on listeners: a sans-I/O protocol the host drives.

Brand-neutral. The session tests use a fake send, a fake timer and a fake
clock, so they are deterministic. The service tests pin the scoped
registration (key space, removal) without sockets; one loopback test binds a
real UDP socket on the shared host loop, like ``test_discovery_host.py``.
"""

import socket
import time
from typing import List

import pytest

from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.scope import Scope
from simplyprint_ws_client.events import Event
from simplyprint_ws_client.integration.discovery.listener import (
    DatagramReceived,
    Forget,
    ListenerSession,
    ListenerStarted,
    ListenerTick,
    SendTo,
    Sighted,
    WakeAfter,
)
from simplyprint_ws_client.integration.discovery.service import DiscoveryService
from simplyprint_ws_client.integration.discovery.spec import (
    ListenerSpec,
    MulticastSpec,
)
from simplyprint_ws_client.integration.spec import (
    IntegrationId,
    IntegrationSpec,
    ProductMetadata,
)


class _Echo:
    """Announces on start, re-announces on each tick, sights ``id=<x>`` replies,
    forgets ``bye=<x>``, and raises on ``boom``."""

    def __init__(self, announce_port: int = 9) -> None:
        self.events: List[object] = []
        self.announce_port = announce_port

    def handle(self, event):
        self.events.append(event)
        if isinstance(event, (ListenerStarted, ListenerTick)):
            return [
                SendTo(b"hello?", "255.255.255.255", self.announce_port),
                WakeAfter(30),
            ]
        if isinstance(event, DatagramReceived):
            text = event.data.decode()
            if text == "boom":
                raise RuntimeError("bad datagram")
            if text.startswith("bye="):
                return [Forget(text[4:])]
            if text.startswith("id="):
                return [Sighted({"id": text[3:], "host": event.source[0]})]
        return []


def _spec(brand: str = "alpha.raw", protocol=_Echo, **kwargs) -> ListenerSpec:
    return ListenerSpec(
        brand=brand, protocol=protocol, key=lambda record: record["id"], **kwargs
    )


class _Harness:
    def __init__(self, spec: ListenerSpec) -> None:
        self.sent: list = []
        self.wakes: list = []
        self.published: list = []
        self.now = 100.0
        self.devices = ExpiringDict(ttl=300)
        self.session = ListenerSession(
            spec,
            self.devices,
            send=lambda data, addr: self.sent.append((data, addr)),
            wake=self.wakes.append,
            publish=self.published.append,
            clock=lambda: self.now,
        )


# -- the session -----------------------------------------------------------------


def test_start_performs_the_protocols_sends_and_arms_its_timer():
    h = _Harness(_spec())
    h.session.start()

    assert h.sent == [(b"hello?", ("255.255.255.255", 9))]
    assert h.wakes == [30.0]
    assert h.session.protocol.events == [ListenerStarted(100.0)]


def test_a_tick_is_fed_as_a_tick_event_with_the_clock():
    h = _Harness(_spec())
    h.session.start()
    h.now = 130.0
    h.session.tick()

    assert h.session.protocol.events[-1] == ListenerTick(130.0)
    assert len(h.sent) == 2


def test_a_sighting_is_cached_by_the_specs_key_then_published():
    h = _Harness(_spec())
    h.session.start()
    h.session.received(b"id=A1", ("10.0.0.5", 3000))

    record = {"id": "A1", "host": "10.0.0.5"}
    assert h.devices.get("A1") == record
    assert h.published == [record]
    assert h.session.protocol.events[-1] == DatagramReceived(
        b"id=A1", ("10.0.0.5", 3000), 100.0
    )


def test_forget_drops_a_cached_key_and_tolerates_an_unknown_one():
    h = _Harness(_spec())
    h.session.received(b"id=A1", ("10.0.0.5", 3000))
    h.session.received(b"bye=A1", ("10.0.0.5", 3000))
    h.session.received(b"bye=nobody", ("10.0.0.5", 3000))

    assert h.devices.get("A1") is None


def test_a_raising_protocol_loses_only_that_event():
    h = _Harness(_spec())
    h.session.received(b"boom", ("10.0.0.5", 3000))
    h.session.received(b"id=B2", ("10.0.0.6", 3000))

    assert [r["id"] for r in h.published] == ["B2"]


def test_a_failing_send_does_not_stop_the_other_actions():
    def refuse(data, addr):
        raise OSError("network unreachable")

    wakes: list = []
    session = ListenerSession(
        _spec(),
        ExpiringDict(ttl=300),
        send=refuse,
        wake=wakes.append,
        publish=lambda record: None,
        clock=lambda: 0.0,
    )
    session.start()

    assert wakes == [30.0]


def test_a_negative_wake_fires_at_once_and_stop_disarms():
    class _Soon:
        def handle(self, event):
            return [WakeAfter(-5)]

    h = _Harness(_spec(protocol=_Soon))
    h.session.start()
    h.session.stop()

    assert h.wakes == [0.0, None]


def test_each_session_builds_a_fresh_protocol():
    spec = _spec()
    first, second = _Harness(spec), _Harness(spec)

    assert first.session.protocol is not second.session.protocol


# -- the service: scoped registration --------------------------------------------


class _FoundEvent(Event): ...


def _multicast(brand: str) -> MulticastSpec:
    return MulticastSpec(
        brand=brand,
        group="239.255.255.250",
        port=1900,
        event_type=_FoundEvent,
        mapper=lambda request, addr: None,
        key=lambda record: record,
    )


def test_add_listener_registers_until_the_scope_closes():
    service = DiscoveryService()
    scope = Scope("plugin")
    service.add_listener(_spec(), scope)

    assert service.backend_keys() == ["alpha.raw"]
    service._listeners["alpha.raw"].devices["A1"] = {"id": "A1"}
    assert service.snapshot("alpha.raw") == [{"id": "A1"}]

    assert scope.close("unload").ok
    assert service.backend_keys() == []
    assert service.snapshot("alpha.raw") == []


def test_listener_keys_share_the_space_of_every_always_on_listener():
    service = DiscoveryService(multicast_specs=[_multicast("alpha")])
    service.add_listener(_spec("beta"), Scope("a"))

    with pytest.raises(ValueError, match="duplicate discovery listener key: alpha"):
        service.add_listener(_spec("alpha"), Scope("b"))
    with pytest.raises(ValueError, match="duplicate discovery listener key: beta"):
        service.add_listener(_spec("beta"), Scope("c"))


_METADATA = ProductMetadata(
    display_name="Alpha",
    image_url="/alpha.png",
    supported_transports=(),
    capabilities=(),
)


def _integration(key: str, *listeners: ListenerSpec, multicast=None) -> IntegrationSpec:
    return IntegrationSpec(
        id=IntegrationId(key),
        client_factory=lambda config, *, context: None,
        config_factory=PrinterConfig,
        metadata=_METADATA,
        multicast=multicast,
        listeners=tuple(listeners),
    )


def test_integration_listeners_are_added_and_removed_with_its_scope():
    service = DiscoveryService()
    scope = Scope("alpha")
    service.add(
        _integration("alpha", _spec("alpha.raw"), multicast=_multicast("alpha")), scope
    )

    assert service.backend_keys() == ["alpha", "alpha.raw"]
    assert scope.close("unload").ok
    assert service.backend_keys() == []


def test_an_integration_listener_may_not_reuse_its_own_multicast_key():
    service = DiscoveryService()
    with pytest.raises(ValueError, match="duplicate discovery listener key: alpha"):
        service.add(
            _integration("alpha", _spec("alpha"), multicast=_multicast("alpha")),
            Scope("alpha"),
        )
    assert service.backend_keys() == []


# -- hosted, on loopback ---------------------------------------------------------


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_for(predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_hosted_listener_sends_hears_and_stops_with_its_scope():
    responder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    responder.bind(("127.0.0.1", 0))
    responder.settimeout(8.0)
    responder_port = responder.getsockname()[1]

    class _Unicast(_Echo):
        def handle(self, event):
            if isinstance(event, ListenerStarted):
                return [SendTo(b"hello?", "127.0.0.1", responder_port)]
            return super().handle(event)

    port = _free_udp_port()
    service = DiscoveryService(restart_interval=0.2)
    service.start()
    scope = Scope("plugin")
    try:
        service.add_listener(_spec(protocol=_Unicast, port=port), scope)

        # The protocol's send left the host's socket.
        data, _addr = responder.recvfrom(1024)
        assert data == b"hello?"

        def heard() -> bool:
            responder.sendto(b"id=L1", ("127.0.0.1", port))
            return bool(service.snapshot("alpha.raw"))

        assert _wait_for(heard)
        assert service.snapshot("alpha.raw")[0]["id"] == "L1"

        task = service._host._tasks["alpha.raw"]
        assert scope.close("unload", timeout=10).ok
        assert task.done()
        assert "alpha.raw" not in service._host._tasks
    finally:
        service.stop()
        responder.close()
