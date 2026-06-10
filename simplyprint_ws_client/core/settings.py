__all__ = ["ClientSettings", "ClientFactory", "ClientSpec"]

from dataclasses import dataclass
from typing import Optional, Type, Union, Callable, Protocol, TypeVar, List, Sequence

from simplyprint_ws_client.cloud.client import Client
from simplyprint_ws_client.cloud.config import PrinterConfig
from simplyprint_ws_client.core.config import ConfigManagerType
from simplyprint_ws_client.cloud.protocol.connection import ConnectionMode
from simplyprint_ws_client.common.asyncio.event_loop_runner import EventLoopBackend
from simplyprint_ws_client.device.camera.base import BaseCameraProtocol
from simplyprint_ws_client.cloud.api.url_builder import SimplyPrintBackend

TAnyClient = TypeVar("TAnyClient", bound=Client)
TAnyPrinterConfig = TypeVar("TAnyPrinterConfig", bound=PrinterConfig)


class ClientFactory(Protocol):
    def __call__(self, config: TAnyPrinterConfig, *args, **kwargs) -> TAnyClient: ...


TClientFactory = Union[Type[TAnyClient], ClientFactory]
TConfigFactory = Union[Type[TAnyPrinterConfig], Callable[..., TAnyPrinterConfig]]


@dataclass(frozen=True)
class ClientSpec:
    key: str
    client_factory: TClientFactory
    config_factory: TConfigFactory
    name: Optional[str] = None
    config_manager_t: Optional[ConfigManagerType] = None
    allow_setup: Optional[bool] = None
    #: Camera protocol classes this client type can drive. Generic (every entry
    #: is a library ``BaseCameraProtocol``), so an integration declares its
    #: per-client cameras here instead of in a parallel descriptor.
    camera_protocols: tuple[Type[BaseCameraProtocol], ...] = ()

    def storage_name(self, app_name: Optional[str], multiple: bool) -> Optional[str]:
        if self.name is not None:
            return self.name

        if not multiple:
            return app_name

        return f"{app_name}-{self.key}" if app_name else self.key


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
    client_specs: Optional[Sequence[ClientSpec]] = None

    def resolved_client_specs(self) -> tuple[ClientSpec, ...]:
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
            ClientSpec(
                key="default",
                client_factory=self.client_factory,
                config_factory=self.config_factory,
                name=self.name,
                config_manager_t=self.config_manager_t,
                allow_setup=self.allow_setup,
            ),
        )

    def get_client_spec(self, key: Optional[str] = None) -> ClientSpec:
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
