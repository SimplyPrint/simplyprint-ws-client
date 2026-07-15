"""Public authoring API for SimplyPrint printer integrations.

Every name exported here is deliberate and statically traceable. Protocol and
runtime internals remain available from their owning modules; the package root
does not search modules for undocumented compatibility exports.
"""

from simplyprint_ws_client.core.app import ClientApp
from simplyprint_ws_client.core.client import (
    Client,
    ClientConfigChangedEvent,
    ClientState,
    ClientStateChangeEvent,
    PeripheralDefinitionEntry,
    PeripheralDefinitions,
)
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.config import (
    Config,
    ConfigManager,
    ConfigManagerType,
    PrinterConfig,
)
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.messages import (
    ConnectedMsg,
    FileDemandData,
    GcodeDemandData,
    MaterialDataMsg,
    MeshDataMsg,
    MMSMapEntry,
    ObjectsMsg,
    PeripheralActionDemandData,
    PeripheralDefinitionsMsg,
    PeripheralMsg,
    PluginInstallDemandData,
    RefreshPeripheralsDemandData,
    ResolveNotificationDemandData,
    SendLogsDemandData,
    SetMaterialDataDemandData,
    SkipObjectsDemandData,
)
from simplyprint_ws_client.core.protocol.models import PeripheralAction
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.core.state import (
    FileProgressState,
    FileProgressStateEnum,
    MaterialEntry,
    MaterialLayoutEntry,
    MultiMaterialSolution,
    NotificationEventPayload,
    NotificationEventSeverity,
    NotificationEventType,
    NozzleType,
    PrinterState,
    PrinterStatus,
)
from simplyprint_ws_client.integration.client import PrinterClient
from simplyprint_ws_client.integration.spec import (
    IntegrationCapability,
    IntegrationId,
    IntegrationSpec,
    IntegrationTransport,
    ProductMetadata,
    discover_from_service,
    model_aware_presentation,
)

__all__ = [
    "Client",
    "ClientApp",
    "ClientConfigChangedEvent",
    "ClientContext",
    "ClientSettings",
    "ClientState",
    "ClientStateChangeEvent",
    "Config",
    "ConfigManager",
    "ConfigManagerType",
    "ConnectedMsg",
    "ConnectionMode",
    "FileDemandData",
    "FileProgressState",
    "FileProgressStateEnum",
    "GcodeDemandData",
    "IntegrationCapability",
    "IntegrationId",
    "IntegrationSpec",
    "IntegrationTransport",
    "MaterialDataMsg",
    "MaterialEntry",
    "MaterialLayoutEntry",
    "MeshDataMsg",
    "MMSMapEntry",
    "MultiMaterialSolution",
    "NotificationEventPayload",
    "NotificationEventSeverity",
    "NotificationEventType",
    "NozzleType",
    "ObjectsMsg",
    "PeripheralAction",
    "PeripheralActionDemandData",
    "PeripheralDefinitionEntry",
    "PeripheralDefinitions",
    "PeripheralDefinitionsMsg",
    "PeripheralMsg",
    "PluginInstallDemandData",
    "PrinterClient",
    "PrinterConfig",
    "PrinterState",
    "PrinterStatus",
    "ProductMetadata",
    "RefreshPeripheralsDemandData",
    "ResolveNotificationDemandData",
    "SendLogsDemandData",
    "SetMaterialDataDemandData",
    "SkipObjectsDemandData",
    "discover_from_service",
    "model_aware_presentation",
]
