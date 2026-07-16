"""Immutable integration descriptors.

An integration is one :class:`IntegrationSpec` value.  The value contains the
runtime factories and the optional capabilities that the host projects; there
is no descriptor subclass, hook discovery, dotted import string, or build step.

Factories that intentionally defer a heavy import are ordinary callables with a
local import.  This keeps laziness visible at the application composition
boundary and preserves normal Python identity and static analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    List,
    NewType,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict

from simplyprint_ws_client._compat import StrEnum
from simplyprint_ws_client.core.client_context import (
    BackgroundService,
    ClientContext,
)

__all__ = [
    "IntegrationId",
    "IntegrationTransport",
    "IntegrationCapability",
    "IntegrationSpec",
    "BetaInfo",
    "ProductMetadata",
    "discover_from_service",
    "model_aware_presentation",
]

if TYPE_CHECKING:
    from simplyprint_ws_client.core.client import Client
    from simplyprint_ws_client.core.config import ConfigManagerType, PrinterConfig
    from simplyprint_ws_client.integration.accounts import AccountProvider
    from simplyprint_ws_client.integration.camera.base import BaseCameraProtocol
    from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice
    from simplyprint_ws_client.integration.discovery.service import DiscoveryService
    from simplyprint_ws_client.integration.discovery.spec import (
        MDNSSpec,
        MulticastSpec,
        NetworkServiceSpec,
        SubnetScanSpec,
    )
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.model_catalogue import ModelCatalogue
    from simplyprint_ws_client.integration.presentation import PrinterPresentation
    from simplyprint_ws_client.integration.tasks import TaskRegistry


IntegrationId = NewType("IntegrationId", str)

TAnyClient = TypeVar("TAnyClient", bound="Client")
TAnyPrinterConfig = TypeVar("TAnyPrinterConfig", bound="PrinterConfig")


class ClientFactory(Protocol):
    def __call__(
        self,
        config: TAnyPrinterConfig,
        *,
        context: ClientContext,
    ) -> TAnyClient: ...


TClientFactory = Union[Type[TAnyClient], ClientFactory]
TConfigFactory = Union[Type[TAnyPrinterConfig], Callable[..., TAnyPrinterConfig]]

Discoverer = Callable[["DiscoveryService", float], Awaitable[List["DiscoveredDevice"]]]
DiscoveryRefiner = Callable[["DiscoveredDevice"], Optional["DiscoveredDevice"]]
CameraProtocolsFactory = Callable[[], Tuple[Type["BaseCameraProtocol"], ...]]
BackgroundServiceFactory = Callable[[Any], "BackgroundService"]
AccountProviderFactory = Callable[["ConfigManagerType"], "AccountProvider"]
FlowFactory = Callable[[ClientContext], "Flow"]
TaskRegistrar = Callable[["TaskRegistry"], None]
PresentationFactory = Callable[["PrinterConfig"], "PrinterPresentation"]
MqttUrlFactory = Callable[["PrinterConfig", ClientContext], Optional[str]]


class LanMqttVerifier(Protocol):
    def __call__(
        self,
        url: str,
        *,
        timeout: float,
        expected_host: Optional[str],
        context: ClientContext,
    ) -> Awaitable[Optional[Any]]: ...


class IntegrationTransport(StrEnum):
    """Transport vocabulary exposed by product metadata."""

    HTTP = "http"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    FTPS = "ftps"


class IntegrationCapability(StrEnum):
    """Capability vocabulary exposed by product metadata."""

    CAMERA = "camera"
    FILE_UPLOAD = "file_upload"
    AMS = "ams"
    CLOUD_ACCOUNT = "cloud_account"
    LAN_ACCESS_CODE = "lan_access_code"


class BetaInfo(BaseModel):
    """Optional setup links shown for a beta integration."""

    model_config = ConfigDict(frozen=True)

    setup_guide_url: Optional[str] = None
    helpdesk_url: Optional[str] = None


class ProductMetadata(BaseModel):
    """Neutral product facts exposed by the integration catalogue."""

    model_config = ConfigDict(frozen=True)

    display_name: str
    image_url: str
    supported_transports: tuple[IntegrationTransport, ...]
    capabilities: tuple[IntegrationCapability, ...]
    beta: Optional[BetaInfo] = None


def no_camera_protocols() -> Tuple[Type["BaseCameraProtocol"], ...]:
    """The explicit camera capability for integrations without a camera."""

    return ()


@dataclass(frozen=True)
class IntegrationSpec:
    """The complete immutable description of one integration.

    Optional capabilities are data or callables.  ``None`` means the capability
    is absent; callers never infer support from subclass overrides.
    """

    id: IntegrationId
    client_factory: TClientFactory
    config_factory: TConfigFactory
    metadata: ProductMetadata

    name: Optional[str] = None
    config_manager_t: Optional["ConfigManagerType"] = None
    camera_protocols: CameraProtocolsFactory = no_camera_protocols

    background_service_factory: Optional[BackgroundServiceFactory] = None
    account_provider_factory: Optional[AccountProviderFactory] = None

    multicast: Optional["MulticastSpec"] = None
    mdns: Optional["MDNSSpec"] = None
    subnet: Optional["SubnetScanSpec"] = None
    network_services: Tuple["NetworkServiceSpec", ...] = ()
    discover: Optional[Discoverer] = None

    add_printer_flow_factory: Optional[FlowFactory] = None
    account_login_flow_factory: Optional[FlowFactory] = None
    register_tasks: Optional[TaskRegistrar] = None

    presentation: Optional[PresentationFactory] = None
    model_catalogue: Optional["ModelCatalogue"] = None
    discovery_priority: int = 0

    mqtt_url_from_config: Optional[MqttUrlFactory] = None
    verify_lan_mqtt_url: Optional[LanMqttVerifier] = None

    def storage_name(self, app_name: Optional[str], multiple: bool) -> Optional[str]:
        if self.name is not None:
            return self.name
        if not multiple:
            return app_name
        return f"{app_name}-{self.id}" if app_name else str(self.id)


def discover_from_service(
    integration_id: IntegrationId,
    *,
    refine: Optional[DiscoveryRefiner] = None,
) -> Discoverer:
    """Build the standard discovery projection for one explicit integration id."""

    async def discover(
        service: "DiscoveryService", timeout: float
    ) -> List["DiscoveredDevice"]:
        from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice

        records = await service.scan(str(integration_id), timeout)
        devices = (
            DiscoveredDevice(
                host=record.host,
                name=record.name,
                serial=record.serial,
                hardware_id=record.hardware_id,
                extra=dict(record.extra),
            )
            for record in records
        )
        if refine is None:
            return list(devices)
        refined = (refine(device) for device in devices)
        return [device for device in refined if device is not None]

    return discover


def model_aware_presentation(
    metadata: ProductMetadata,
    catalogue: Optional["ModelCatalogue"],
    config: "PrinterConfig",
    device_type: Optional[str],
) -> "PrinterPresentation":
    """Build the common model-aware card without reflective config access."""

    from dataclasses import replace

    from simplyprint_ws_client.integration.presentation import (
        default_printer_presentation,
    )

    base = default_printer_presentation(metadata.image_url, config)
    if catalogue is None or not device_type or device_type == catalogue.unknown_value:
        return base
    return replace(
        base,
        image_url=catalogue.image_url(device_type) or base.image_url,
        model_name=catalogue.model_name(device_type),
    )
