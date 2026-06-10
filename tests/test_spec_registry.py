"""SpecRegistry + Host seam pins: collection/projection shapes, the flow map,
and the add-printer persist seam (slot id once, hardware de-dup, dumb app.add)."""

from typing import Optional

import pytest

from simplyprint_ws_client.cloud.client import Client
from simplyprint_ws_client.cloud.config import PrinterConfig
from simplyprint_ws_client.integration.spec import PrinterSpec, ProductMetadata
from simplyprint_ws_client.runtime.host import DuplicatePrinter, Host
from simplyprint_ws_client.runtime.registry import SpecRegistry
from simplyprint_ws_client.runtime.settings import ClientSettings


class AlphaConfig(PrinterConfig):
    serial: Optional[str] = None
    host: Optional[str] = None

    def stable_hardware_id(self) -> Optional[str]:
        return self.serial


class _AlphaClient(Client):
    pass


def _alpha_client(config, **kwargs):
    return _AlphaClient(config, event_loop_provider=kwargs.get("event_loop_provider"))


class AlphaSpec(PrinterSpec):
    KEY = "alpha"
    metadata = ProductMetadata(
        display_name="Alpha",
        image_url="/img/alpha.png",
        supported_transports=("websocket",),
        capabilities=(),
    )

    @classmethod
    def build(cls):
        return cls(
            key=cls.KEY, client_factory=_alpha_client, config_factory=AlphaConfig
        )

    @classmethod
    def subnet_spec(cls):
        return object()

    @classmethod
    def add_printer_flow(cls):
        return object()


class BetaSpec(PrinterSpec):
    KEY = "beta"
    metadata = ProductMetadata(
        display_name="Beta",
        image_url="/img/beta.png",
        supported_transports=("http",),
        capabilities=(),
    )

    @classmethod
    def build(cls):
        return cls(
            key=cls.KEY, client_factory=_alpha_client, config_factory=AlphaConfig
        )


def test_registry_collects_only_providing_types():
    registry = SpecRegistry.of(AlphaSpec, BetaSpec)
    assert registry.keys() == ("alpha", "beta")
    assert len(registry.collect("subnet_spec")) == 1
    assert list(registry.collect_map("subnet_spec")) == ["alpha"]
    assert registry.metadata()["beta"].display_name == "Beta"


def test_registry_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        SpecRegistry.of(AlphaSpec, AlphaSpec)


def test_flow_projection_uses_the_stable_flow_ids():
    registry = SpecRegistry.of(AlphaSpec, BetaSpec)
    assert registry.brand_flows("alpha") == ["add-printer"]
    assert registry.list_flows() == {"alpha": ["add-printer"]}
    assert registry.flow_brands() == ["alpha"]
    assert registry.flow("beta", "add-printer") is None
    assert registry.flow("alpha", "no-such-flow") is None


def test_discoverers_ask_discover_not_provides():
    # Alpha rides the spec DEFAULT discover (it declares a subnet spec) -- the
    # registry must include it even though provides("discover") is False.
    registry = SpecRegistry.of(AlphaSpec, BetaSpec)
    discoverers = registry.discoverers()
    assert "alpha" in discoverers and "beta" not in discoverers
    assert not AlphaSpec.provides("discover")


@pytest.fixture
def host():
    registry = SpecRegistry.of(AlphaSpec)
    h = Host(registry, ClientSettings(name="test-host"))
    yield h
    h.app.stop()


def test_host_fills_settings_from_the_registry(host):
    assert host.settings.client_specs == host.registry.runtime_specs()


def test_add_printer_assigns_slot_id_once_and_rejects_duplicates(host):
    first = AlphaConfig.get_blank()
    first.serial = "S-1"
    host.add_printer("alpha", first)
    assert first.unique_id  # slot id assigned at the seam

    again = AlphaConfig.get_blank()
    again.serial = "S-1"
    with pytest.raises(DuplicatePrinter) as excinfo:
        host.add_printer("alpha", again)
    assert excinfo.value.field == "hardware id"
    assert excinfo.value.existing.unique_id == first.unique_id

    # The slot id is stable for life: present -> preserved by the seam.
    third = AlphaConfig.get_blank()
    third.serial = "S-2"
    third.unique_id = "existing-slot"
    host.add_printer("alpha", third)
    assert third.unique_id == "existing-slot"
