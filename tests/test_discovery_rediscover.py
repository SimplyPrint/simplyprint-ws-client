"""Routing a passive sighting back to the printer that moved.

Brand-neutral and socket-free: a sighting is fed where a listener hands it on
(``DiscoveryService._sighted`` / a backend's ``_publish``), and the app and
host halves run against small stand-ins.
"""

import logging
import threading
from types import SimpleNamespace

import pytest

from simplyprint_ws_client.core.app import ClientApp
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.host import Host
from simplyprint_ws_client.core.scope import Scope
from simplyprint_ws_client.events import Event, EventBus
from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice
from simplyprint_ws_client.integration.discovery.listener import (
    ListenerDiscoveryBackend,
)
from simplyprint_ws_client.integration.discovery.service import DiscoveryService
from simplyprint_ws_client.integration.discovery.spec import (
    DatagramSpec,
    ListenerSpec,
    MulticastSpec,
)
from simplyprint_ws_client.integration.spec import (
    IntegrationId,
    IntegrationSpec,
    ProductMetadata,
)


class _FoundEvent(Event): ...


def _multicast(brand: str) -> MulticastSpec:
    return MulticastSpec(
        brand=brand,
        group="239.255.255.250",
        port=1900,
        event_type=_FoundEvent,
        mapper=lambda request, addr: None,
        key=lambda record: record["serial"],
    )


def _listener(brand: str) -> ListenerSpec:
    return ListenerSpec(brand=brand, protocol=object, key=lambda r: r["serial"])


_METADATA = ProductMetadata(
    display_name="Alpha",
    image_url="/alpha.png",
    supported_transports=(),
    capabilities=(),
)


def _to_device(record):
    if not record.get("serial"):
        return None
    return DiscoveredDevice(host=record["host"], serial=record["serial"])


def _integration(key: str = "alpha", **kwargs) -> IntegrationSpec:
    return IntegrationSpec(
        id=IntegrationId(key),
        client_factory=lambda config, *, context: None,
        config_factory=PrinterConfig,
        metadata=_METADATA,
        **kwargs,
    )


# -- DiscoveryService.watch ------------------------------------------------------


def test_a_watcher_gets_every_sighting_of_its_key_until_its_scope_closes():
    service = DiscoveryService()
    scope = Scope("w")
    seen: list = []
    service.watch("alpha", seen.append, scope)

    service._sighted("alpha", {"serial": "S1"})
    service._sighted("beta", {"serial": "S2"})
    assert seen == [{"serial": "S1"}]

    assert scope.close("unload").ok
    service._sighted("alpha", {"serial": "S3"})
    assert seen == [{"serial": "S1"}]
    assert service._watchers == {}


def test_a_raising_watcher_does_not_stop_the_others():
    service = DiscoveryService()
    seen: list = []

    def boom(record):
        raise RuntimeError("watcher bug")

    service.watch("alpha", boom, Scope("a"))
    service.watch("alpha", seen.append, Scope("b"))
    service._sighted("alpha", {"serial": "S1"})

    assert seen == [{"serial": "S1"}]


def test_closing_one_watch_keeps_the_other_on_the_same_key():
    service = DiscoveryService()
    first, second = Scope("a"), Scope("b")
    kept: list = []
    service.watch("alpha", lambda record: None, first)
    service.watch("alpha", kept.append, second)

    assert first.close("unload").ok
    service._sighted("alpha", {"serial": "S1"})
    assert kept == [{"serial": "S1"}]


@pytest.mark.asyncio
async def test_every_hosted_backend_hands_its_sightings_to_the_service():
    service = DiscoveryService(multicast_specs=[_multicast("alpha")])
    service.add_listener(_listener("beta"), Scope("beta"))
    seen: list = []
    service.watch("alpha", lambda record: seen.append(("alpha", record)), Scope("a"))
    service.watch("beta", lambda record: seen.append(("beta", record)), Scope("b"))

    await service._multicast["alpha"]._publish({"serial": "S1"})
    await service._listeners["beta"]._publish({"serial": "S2"})

    assert seen == [("alpha", {"serial": "S1"}), ("beta", {"serial": "S2"})]


@pytest.mark.asyncio
async def test_a_backend_without_a_watcher_hook_or_event_only_caches():
    backend = ListenerDiscoveryBackend(_listener("solo"), EventBus())
    await backend._publish({"serial": "S1"})  # no hook, no event: a no-op


def test_listener_keys_of_lists_every_always_on_listener_a_spec_contributes():
    datagram = DatagramSpec(
        brand="alpha.udp",
        ports=(3000,),
        parse=lambda data, source: None,
        key=lambda record: record,
        payload=b"?",
        interval=10.0,
    )
    spec = _integration(
        multicast=_multicast("alpha"),
        listeners=(_listener("alpha.raw"),),
        datagram=datagram,
    )
    assert DiscoveryService.listener_keys_of(spec) == [
        "alpha",
        "alpha.raw",
        "alpha.udp",
    ]
    on_demand = _integration(datagram=DatagramSpec(
        brand="alpha.udp",
        ports=(3000,),
        parse=lambda data, source: None,
        key=lambda record: record,
        payload=b"?",
    ))
    assert DiscoveryService.listener_keys_of(on_demand) == []


# -- ClientApp.route_discovered --------------------------------------------------


class _Printer:
    def __init__(self, unique_id: str, serial, changes: bool = True) -> None:
        self.unique_id = unique_id
        self.config = SimpleNamespace(hardware_identity=lambda: serial)
        self.applied: list = []
        self._changes = changes

    def apply_discovered(self, device) -> bool:
        self.applied.append(device)
        return self._changes


def _app(domains: dict, printers: list) -> SimpleNamespace:
    return SimpleNamespace(
        _app_lock=threading.Lock(),
        _client_domains=domains,
        client_list={p.unique_id: p for p in printers},
        logger=logging.getLogger("test"),
    )


def test_a_sighting_reaches_only_the_matching_printer_of_its_integration():
    mine = _Printer("u1", "SN-1")
    other = _Printer("u2", "SN-2")
    elsewhere = _Printer("u3", "SN-1")  # same serial, another integration
    app = _app({"u1": "alpha", "u2": "alpha", "u3": "beta"}, [mine, other, elsewhere])
    device = DiscoveredDevice(host="10.0.0.9", serial=" sn-1 ")

    assert ClientApp.route_discovered(app, "alpha", device) == 1
    assert mine.applied == [device]
    assert other.applied == []
    assert elsewhere.applied == []


def test_an_unchanged_or_failing_printer_counts_as_not_changed():
    quiet = _Printer("u1", "SN-1", changes=False)

    class _Broken(_Printer):
        def apply_discovered(self, device):
            raise RuntimeError("driver bug")

    broken = _Broken("u2", "SN-1")
    app = _app({"u1": "alpha", "u2": "alpha"}, [quiet, broken])

    device = DiscoveredDevice(host="10.0.0.9", serial="SN-1")
    assert ClientApp.route_discovered(app, "alpha", device) == 0
    assert quiet.applied == [device]


def test_a_device_without_identity_reaches_nobody(monkeypatch):
    from simplyprint_ws_client.integration.discovery import mac

    monkeypatch.setattr(mac, "resolve_mac", lambda host, warm=False: None)
    printer = _Printer("u1", None)
    app = _app({"u1": "alpha"}, [printer])

    assert ClientApp.route_discovered(app, "alpha", DiscoveredDevice(host="10.0.0.9")) == 0
    assert printer.applied == []


# -- Host: the mount's watch -----------------------------------------------------


def _host() -> SimpleNamespace:
    routed: list = []
    app = SimpleNamespace(
        route_discovered=lambda key, device: routed.append((key, device))
    )
    return SimpleNamespace(discovery=DiscoveryService(), app=app, routed=routed)


def test_a_mount_with_rediscover_routes_its_listeners_sightings_until_unmounted():
    host = _host()
    spec = _integration(
        multicast=_multicast("alpha"),
        listeners=(_listener("alpha.raw"),),
        rediscover=_to_device,
    )
    scope = Scope("alpha")
    Host._mount_rediscovery(host, spec, scope)

    host.discovery._sighted("alpha", {"serial": "S1", "host": "10.0.0.5"})
    host.discovery._sighted("alpha.raw", {"serial": "S2", "host": "10.0.0.6"})
    host.discovery._sighted("alpha", {"serial": "", "host": "10.0.0.7"})  # no device

    assert host.routed == [
        ("alpha", DiscoveredDevice(host="10.0.0.5", serial="S1")),
        ("alpha", DiscoveredDevice(host="10.0.0.6", serial="S2")),
    ]
    assert scope.close("unload").ok
    host.discovery._sighted("alpha", {"serial": "S1", "host": "10.0.0.8"})
    assert len(host.routed) == 2


def test_a_mount_without_rediscover_watches_nothing():
    host = _host()
    Host._mount_rediscovery(host, _integration(multicast=_multicast("alpha")), Scope("a"))
    assert host.discovery._watchers == {}
