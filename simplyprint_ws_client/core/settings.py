from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Type

from simplyprint_ws_client.common.asyncio.event_loop_runner import EventLoopBackend
from simplyprint_ws_client.core.api.url_builder import (
    SimplyPrintBackend,
    SimplyPrintEndpoints,
    resolve_backend_endpoints,
)
from simplyprint_ws_client.core.config import ConfigManagerType
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.integration.camera.base import BaseCameraProtocol
from simplyprint_ws_client.integration.spec import IntegrationSpec

__all__ = ["ClientSettings"]


@dataclass
class ClientSettings:
    """Process settings and the exact integrations hosted by the process."""

    integrations: Optional[Sequence[IntegrationSpec]] = None
    name: Optional[str] = "printers"
    version: Optional[str] = "0.1"
    mode: ConnectionMode = ConnectionMode.SINGLE
    backend: Optional[SimplyPrintBackend] = None
    endpoints: SimplyPrintEndpoints = field(init=False)
    event_loop_backend: EventLoopBackend = EventLoopBackend.ASYNCIO
    development: bool = False
    config_manager_t: ConfigManagerType = ConfigManagerType.MEMORY
    allow_setup: bool = True
    max_clients_per_connection: Optional[int] = None
    tick_rate: float = 1.0
    sentry_dsn: Optional[str] = None
    camera_workers: Optional[int] = None
    camera_protocols: Optional[List[Type[BaseCameraProtocol]]] = None

    def __post_init__(self) -> None:
        self.backend, self.endpoints = resolve_backend_endpoints(self.backend)

    def resolved_integrations(self) -> tuple[IntegrationSpec, ...]:
        integrations = tuple(self.integrations or ())
        if not integrations:
            raise ValueError("At least one integration must be configured.")

        ids = {str(integration.id) for integration in integrations}
        if len(ids) != len(integrations):
            raise ValueError("Integration ids must be unique.")
        return integrations

    def get_integration(self, integration_id: Optional[str] = None) -> IntegrationSpec:
        integrations = self.resolved_integrations()
        if integration_id is None:
            if len(integrations) == 1:
                return integrations[0]
            raise ValueError(
                "Integration id is required when multiple integrations exist."
            )

        for integration in integrations:
            if integration.id == integration_id:
                return integration
        raise KeyError(f"Unknown integration: {integration_id}")

    def new_config_manager(self, integration_id: Optional[str] = None):
        integration = self.get_integration(integration_id)
        integrations = self.resolved_integrations()
        manager_t = integration.config_manager_t or self.config_manager_t
        return manager_t(
            name=integration.storage_name(self.name, multiple=len(integrations) > 1),
            config_t=integration.config_factory,
        )
