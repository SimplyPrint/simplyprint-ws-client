__all__ = ["ClientSettings"]

from dataclasses import dataclass
from typing import Optional, Type, List, Sequence

from simplyprint_ws_client.runtime.config import ConfigManagerType
from simplyprint_ws_client.cloud.protocol.connection import ConnectionMode
from simplyprint_ws_client.common.asyncio.event_loop_runner import EventLoopBackend
from simplyprint_ws_client.device.camera.base import BaseCameraProtocol
from simplyprint_ws_client.cloud.api.url_builder import SimplyPrintBackend
from simplyprint_ws_client.integration.spec import (
    PrinterSpec,
    TClientFactory,
    TConfigFactory,
)


@dataclass
class ClientSettings:
    client_factory: Optional[TClientFactory] = None
    config_factory: Optional[TConfigFactory] = None
    name: Optional[str] = "printers"
    version: Optional[str] = "0.1"
    mode: ConnectionMode = ConnectionMode.SINGLE
    backend: Optional[SimplyPrintBackend] = None
    event_loop_backend: EventLoopBackend = EventLoopBackend.ASYNCIO
    development: bool = False
    config_manager_t: ConfigManagerType = ConfigManagerType.MEMORY
    allow_setup: bool = True
    max_clients_per_connection: Optional[int] = None
    tick_rate = 1.0
    reconnect_timeout = 5.0
    sentry_dsn: Optional[str] = None
    camera_workers: Optional[int] = None
    camera_protocols: Optional[List[Type[BaseCameraProtocol]]] = None
    client_specs: Optional[Sequence[PrinterSpec]] = None

    def resolved_client_specs(self) -> tuple[PrinterSpec, ...]:
        if self.client_specs is not None:
            specs = tuple(self.client_specs)

            if not specs:
                raise ValueError("At least one client spec must be configured.")

            keys = {spec.key for spec in specs}

            if len(keys) != len(specs):
                raise ValueError("Client spec keys must be unique.")

            return specs

        if self.client_factory is None or self.config_factory is None:
            raise ValueError(
                "Either client_specs or both client_factory/config_factory must be set."
            )

        return (
            PrinterSpec(
                key="default",
                client_factory=self.client_factory,
                config_factory=self.config_factory,
                name=self.name,
                config_manager_t=self.config_manager_t,
                allow_setup=self.allow_setup,
            ),
        )

    def get_client_spec(self, key: Optional[str] = None) -> PrinterSpec:
        specs = self.resolved_client_specs()

        if key is None:
            if len(specs) == 1:
                return specs[0]

            raise ValueError("Client spec key is required when multiple specs exist.")

        for spec in specs:
            if spec.key == key:
                return spec

        raise KeyError(f"Unknown client spec: {key}")

    def new_config_manager(self, key: Optional[str] = None):
        spec = self.get_client_spec(key)
        specs = self.resolved_client_specs()
        manager_t = spec.config_manager_t or self.config_manager_t

        return manager_t(
            name=spec.storage_name(self.name, multiple=len(specs) > 1),
            config_t=spec.config_factory,
        )
