"""Value registry + Host seam contract pins."""

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Optional

import pytest

from simplyprint_ws_client.core.client import Client
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.host import DuplicatePrinter, Host
from simplyprint_ws_client.core.registry import SpecRegistry
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.integration.spec import (
    IntegrationCapability,
    IntegrationId,
    IntegrationSpec,
    IntegrationTransport,
    ProductMetadata,
)
from simplyprint_ws_client.integration.discovery import DiscoveryService
from simplyprint_ws_client.integration.discovery.spec import SubnetScanSpec


class AlphaConfig(PrinterConfig):
    serial: Optional[str] = None
    host: Optional[str] = None

    def hardware_identity(self) -> Optional[str]:
        return self.serial


class _AlphaClient(Client):
    pass


def _alpha_client(config, *, context):
    return _AlphaClient(config, context=context)


_flow_builds = {"count": 0, "context": None}


def _build_alpha_flow(context):
    _flow_builds["count"] += 1
    _flow_builds["context"] = context
    return SimpleNamespace(id="add-printer")


async def _discover_alpha(_discovery_service, _timeout):
    return []


async def _probe_alpha(_host):
    return None


ALPHA = IntegrationSpec(
    id=IntegrationId("alpha"),
    client_factory=_alpha_client,
    config_factory=AlphaConfig,
    metadata=ProductMetadata(
        display_name="Alpha",
        image_url="/img/alpha.png",
        supported_transports=(IntegrationTransport.WEBSOCKET,),
        capabilities=(),
    ),
    subnet=SubnetScanSpec(brand="alpha", probe=_probe_alpha, key=lambda record: record),
    discover=_discover_alpha,
    add_printer_flow_factory=_build_alpha_flow,
)
BETA = IntegrationSpec(
    id=IntegrationId("beta"),
    client_factory=_alpha_client,
    config_factory=AlphaConfig,
    metadata=ProductMetadata(
        display_name="Beta",
        image_url="/img/beta.png",
        supported_transports=(IntegrationTransport.HTTP,),
        capabilities=(),
    ),
)


def test_registry_projects_typed_discovery_specs():
    registry = SpecRegistry.of(ALPHA, BETA)
    assert registry.ids() == ("alpha", "beta")
    assert len(registry.subnet_specs()) == 1
    assert registry.multicast_specs() == ()
    assert registry.mdns_specs() == ()
    assert registry.metadata()["beta"].display_name == "Beta"


def test_registry_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        SpecRegistry.of(ALPHA, ALPHA)


def test_integration_values_are_immutable():
    with pytest.raises(FrozenInstanceError):
        ALPHA.name = "renamed"


def test_product_metadata_domains_serialize_as_wire_strings():
    metadata = ProductMetadata(
        display_name="Closed",
        image_url="/closed.png",
        supported_transports=(IntegrationTransport.HTTP,),
        capabilities=(IntegrationCapability.FILE_UPLOAD,),
    )

    assert metadata.supported_transports == (IntegrationTransport.HTTP,)
    assert metadata.capabilities == (IntegrationCapability.FILE_UPLOAD,)
    payload = json.loads(metadata.model_dump_json())
    assert payload["supported_transports"] == ["http"]
    assert payload["capabilities"] == ["file_upload"]


def test_product_metadata_domains_are_closed():
    assert {transport.value for transport in IntegrationTransport} == {
        "http",
        "websocket",
        "mqtt",
        "ftps",
    }
    assert {capability.value for capability in IntegrationCapability} == {
        "camera",
        "file_upload",
        "ams",
        "cloud_account",
        "lan_access_code",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("supported_transports", "smtp"), ("capabilities", "teleport")],
)
def test_product_metadata_rejects_unknown_domain_values(field, value):
    values = {
        "supported_transports": (),
        "capabilities": (),
        field: (value,),
    }
    with pytest.raises(ValueError):
        ProductMetadata(
            display_name="Invalid",
            image_url="/invalid.png",
            **values,
        )


def test_metadata_domains_are_public_exports():
    from simplyprint_ws_client import (
        IntegrationCapability as PublicCapability,
        IntegrationTransport as PublicTransport,
    )

    assert PublicTransport is IntegrationTransport
    assert PublicCapability is IntegrationCapability


def test_flow_projection_uses_the_stable_flow_ids():
    registry = SpecRegistry.of(ALPHA, BETA)
    discovery = DiscoveryService()
    context = ClientContext(discovery_service=discovery)
    _flow_builds["count"] = 0
    assert registry.brand_flows("alpha") == ["add-printer"]
    assert registry.list_flows() == {"alpha": ["add-printer"]}
    assert registry.flow_brands() == ["alpha"]
    assert registry.flow("beta", "add-printer", context) is None
    assert registry.flow("alpha", "no-such-flow", context) is None
    assert _flow_builds["count"] == 0
    assert registry.flow("alpha", "add-printer", context).id == "add-printer"
    assert _flow_builds["count"] == 1


def test_discoverers_project_only_explicit_callables():
    registry = SpecRegistry.of(ALPHA, BETA)
    assert registry.discoverers() == {"alpha": _discover_alpha}


@pytest.fixture
def host():
    registry = SpecRegistry.of(ALPHA)
    instance = Host(registry, ClientSettings(name="test-host"))
    yield instance
    instance.stop()


def test_host_fills_settings_from_the_registry(host):
    assert host.settings.integrations == host.registry.values()


def test_unstarted_host_stop_closes_its_owned_app(host):
    loop = host.app._app_event_loop

    host.stop()
    host.stop()

    assert loop.is_closed()


def test_host_owns_and_injects_one_restartable_discovery_service(host):
    service = host.discovery
    assert host.app.client_context.discovery_service is service

    flow = host.flow("alpha", "add-printer")
    assert flow.id == "add-printer"
    flow_context = _flow_builds["context"]
    assert flow_context.discovery_service is service

    host.start_discovery()
    first_runner = service._host
    host.start_discovery()
    assert service._host is first_runner

    host.stop_discovery()
    host.stop_discovery()
    assert service.is_stopped()
    assert host.discovery is service

    host.start_discovery()
    assert not service.is_stopped()
    assert service._host is not None
    assert service._host is not first_runner
    host.stop_discovery()


def test_host_rejects_a_second_integration_source():
    with pytest.raises(ValueError, match="must match exactly"):
        Host(
            SpecRegistry.of(ALPHA),
            ClientSettings(integrations=(BETA,)),
        )


def test_add_printer_assigns_slot_id_once_and_rejects_duplicates(host):
    first = AlphaConfig.get_blank()
    first.serial = "S-1"
    host.add_printer("alpha", first)
    assert first.unique_id

    again = AlphaConfig.get_blank()
    again.serial = "S-1"
    with pytest.raises(DuplicatePrinter) as excinfo:
        host.add_printer("alpha", again)
    assert excinfo.value.field == "hardware id"
    assert excinfo.value.existing.unique_id == first.unique_id

    third = AlphaConfig.get_blank()
    third.serial = "S-2"
    third.unique_id = "existing-slot"
    host.add_printer("alpha", third)
    assert third.unique_id == "existing-slot"


# ---------------------------------------------------------------------------
# SpecRegistry.add: scoped, copy-on-write
# ---------------------------------------------------------------------------


def test_add_registers_until_the_scope_closes():
    from simplyprint_ws_client.core.scope import Scope

    registry = SpecRegistry.of(ALPHA)
    scope = Scope("beta")
    registry.add(BETA, scope)

    assert registry.ids() == ("alpha", "beta")
    assert registry.get("beta") is BETA

    assert scope.close("unload").ok
    assert registry.ids() == ("alpha",)
    assert registry.get("beta") is None


def test_add_is_copy_on_write_for_readers():
    from simplyprint_ws_client.core.scope import Scope

    registry = SpecRegistry.of(ALPHA)
    before = registry.integrations()
    scope = Scope("beta")
    snapshot = registry._integrations
    registry.add(BETA, scope)

    assert registry._integrations is not snapshot
    assert list(snapshot) == ["alpha"]  # a reader's view never changes
    assert list(before) == ["alpha"]
    scope.close("unload")


def test_add_rejects_another_spec_under_a_taken_id():
    import dataclasses

    from simplyprint_ws_client.core.scope import Scope

    registry = SpecRegistry.of(ALPHA)
    impostor = dataclasses.replace(ALPHA, name="other")
    with pytest.raises(ValueError, match="duplicate integration id"):
        registry.add(impostor, Scope("x"))
    assert registry.get("alpha") is ALPHA


def test_add_adopts_the_spec_already_registered():
    from simplyprint_ws_client.core.scope import Scope

    registry = SpecRegistry.of(ALPHA, BETA)
    scope = Scope("root")
    registry.add(ALPHA, scope)

    assert registry.ids() == ("alpha", "beta")
    scope.close("stop")
    assert registry.ids() == ("beta",)


def test_the_undo_leaves_a_replacement_alone():
    import dataclasses

    from simplyprint_ws_client.core.scope import Scope

    registry = SpecRegistry()
    old = Scope("old")
    registry.add(BETA, old)
    # An update path that already swapped in a new spec must not lose it to
    # the old scope's late undo.
    replacement = dataclasses.replace(BETA, name="v2")
    registry._discard("beta", BETA)
    registry.add(replacement, Scope("new"))

    old.close("late")
    assert registry.get("beta") is replacement


def test_register_still_works_without_a_scope():
    registry = SpecRegistry()
    registry.register(ALPHA)
    assert registry.ids() == ("alpha",)
    with pytest.raises(ValueError):
        registry.register(ALPHA)
