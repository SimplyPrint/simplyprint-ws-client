"""The one descriptor an integration writes per client type: ``PrinterSpec``.

Every client type an integration ships is described by exactly one
:class:`PrinterSpec` subclass -- the single source of truth for that type, and
the *only* descriptor (2.0 merged the old runtime ``PrinterSpec`` and the
authoring ``PrinterSpec`` into this one class). It carries the runtime
fields the app hands to ``ClientSettings`` (key, factories, cameras, name), and
it owns every per-type surface an app projects -- product metadata, the
background service, discovery specs, guided flows, account capability, periodic
tasks -- as hooks that default to "this type doesn't back that surface".

The subtlety is laziness. ``KEY`` and ``metadata`` are :class:`~typing.ClassVar`
literals and the capability hooks are classmethods, so a surface can ask *which*
client types back it (:meth:`provides`) and read their catalogue facts without
ever calling :meth:`build` -- the one place a type imports its cameras / runtime.
Declarative types skip writing ``build`` entirely: point :attr:`client` /
:attr:`config` / :attr:`cameras` at dotted paths via :func:`lazy` and the default
``build`` resolves them exactly once, at the same startup boundary.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from simplyprint_ws_client.cloud.client import Client
    from simplyprint_ws_client.cloud.config import PrinterConfig
    from simplyprint_ws_client.runtime.config import ConfigManagerType
    from simplyprint_ws_client.device.accounts import AccountProvider
    from simplyprint_ws_client.device.camera.base import BaseCameraProtocol
    from simplyprint_ws_client.device.discovery.spec import (
        MDNSSpec,
        MulticastSpec,
        NetworkServiceSpec,
        SubnetScanSpec,
    )
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.presentation import PrinterPresentation
    from simplyprint_ws_client.integration.tasks import TaskRegistry

TAnyClient = TypeVar("TAnyClient", bound="Client")
TAnyPrinterConfig = TypeVar("TAnyPrinterConfig", bound="PrinterConfig")


class ClientFactory(Protocol):
    def __call__(self, config: TAnyPrinterConfig, *args, **kwargs) -> TAnyClient: ...


TClientFactory = Union[Type[TAnyClient], ClientFactory]
TConfigFactory = Union[Type[TAnyPrinterConfig], Callable[..., TAnyPrinterConfig]]


class LazyRef:
    """A ``"package.module:attr"`` reference resolved on first use, then cached.

    The declarative side of :class:`PrinterSpec`: a spec points its
    :attr:`~PrinterSpec.client` / :attr:`~PrinterSpec.config` /
    :attr:`~PrinterSpec.cameras` at dotted paths, and the default ``build``
    resolves them -- so the catalogue never imports a type's heavy runtime, and
    the import is paid exactly once, at the same boundary a hand-written
    ``build`` paid it.
    """

    def __init__(self, target: str) -> None:
        if ":" not in target:
            raise ValueError(f"lazy ref needs 'package.module:attr', got {target!r}")
        self.target = target
        self._resolved: Any = None

    def resolve(self) -> Any:
        if self._resolved is None:
            module_path, attr = self.target.split(":", 1)
            value: Any = importlib.import_module(module_path)
            for part in attr.split("."):
                value = getattr(value, part)
            self._resolved = value
        return self._resolved

    def __repr__(self) -> str:
        return f"lazy({self.target!r})"


def lazy(target: str) -> LazyRef:
    """Declare a lazily-resolved ``"package.module:attr"`` reference."""
    return LazyRef(target)


class BackgroundService(Protocol):
    """Supervisor-owned service contract: registerable things can be stopped."""

    def stop(self) -> None: ...


class ProductMetadata(BaseModel):
    """Neutral, brand-agnostic product descriptor.

    Values are populated by each client type's spec as plain static
    strings/tuples; this container itself imports nothing brand-specific.
    """

    model_config = ConfigDict(frozen=True)

    display_name: str
    image_url: str
    supported_transports: tuple[str, ...]
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class PrinterSpec:
    """One client type's single source of truth -- the only descriptor.

    Subclasses live next to their client type's code, set the :attr:`KEY` /
    :attr:`metadata` class literals, point :attr:`client` / :attr:`config` /
    :attr:`cameras` at dotted paths (or implement :meth:`build` themselves for
    genuinely custom construction), and override only the capability hooks they
    back. The hooks below default to "absent"; a surface lists the types that
    back it via :meth:`provides` and builds none it doesn't need.
    """

    # -- the runtime fields the app consumes (the old PrinterSpec) -------------

    key: str
    client_factory: TClientFactory
    config_factory: TConfigFactory
    name: Optional[str] = None
    config_manager_t: Optional["ConfigManagerType"] = None
    allow_setup: Optional[bool] = None
    #: Camera protocol classes this client type can drive. Generic (every entry
    #: is a library ``BaseCameraProtocol``), so an integration declares its
    #: per-client cameras here instead of in a parallel descriptor.
    camera_protocols: Tuple[Type["BaseCameraProtocol"], ...] = field(default=())

    def storage_name(self, app_name: Optional[str], multiple: bool) -> Optional[str]:
        if self.name is not None:
            return self.name

        if not multiple:
            return app_name

        return f"{app_name}-{self.key}" if app_name else self.key

    # -- catalogue identity, read off the class without building --------------

    #: ``key`` is the per-instance runtime field; :meth:`build` sets it from this.
    KEY: ClassVar[str]
    #: Declarative product facts, read off the class without building the heavy
    #: runtime spec (so the catalogue never imports a brand's cameras/runtime).
    metadata: ClassVar[ProductMetadata]
    #: Optional storage name the default ``build`` passes through.
    NAME: ClassVar[Optional[str]] = None

    # -- declarative construction (the default ``build`` consumes these) ------

    #: ``lazy("pkg.module:PrinterClass")`` -- the client factory.
    client: ClassVar[Optional[LazyRef]] = None
    #: ``lazy("pkg.module:ConfigClass")`` -- the config class.
    config: ClassVar[Optional[LazyRef]] = None
    #: ``(lazy("pkg.module:CameraProtocol"), ...)`` -- camera protocol classes.
    cameras: ClassVar[Tuple[LazyRef, ...]] = ()

    @classmethod
    def build(cls) -> "PrinterSpec":
        """Construct the runtime spec for this client type.

        The default resolves the declarative :attr:`client` / :attr:`config` /
        :attr:`cameras` refs (the one place their imports are paid). A type whose
        construction is genuinely custom overrides this instead and returns
        ``cls(key=cls.KEY, ...)`` with the runtime fields filled.
        """
        if cls.client is None or cls.config is None:
            raise NotImplementedError(
                f"{cls.__name__} must point `client = lazy(...)` and "
                "`config = lazy(...)` at its classes, or override build()."
            )
        return cls(
            key=cls.KEY,
            name=cls.NAME,
            client_factory=cls.client.resolve(),
            config_factory=cls.config.resolve(),
            camera_protocols=tuple(ref.resolve() for ref in cls.cameras),
        )

    # -- capability hooks: default "this type does not back the surface" --------

    @classmethod
    def background_service(cls, event_loop_provider=None) -> "BackgroundService | None":
        """A process-wide background service this client type needs (for example a
        watchdog), or ``None``.

        Constructed once by the app's supervisor at startup and read back by the
        client factory -- never built in the factory, so two factory calls can't
        race to create one. Return a ready-to-register service (already started if
        it needs starting). Read off the *class* (no ``build``).

        ``event_loop_provider`` is the app's loop provider, for services that must
        deliver work onto that loop.
        """
        return None

    @classmethod
    def account_provider(cls) -> "Optional[AccountProvider]":
        """The brand's cloud-account capability, or ``None`` if it has no cloud."""
        return None

    @classmethod
    def multicast_spec(cls) -> "Optional[MulticastSpec]":
        """An always-on SSDP multicast discovery spec, or ``None``."""
        return None

    @classmethod
    def mdns_spec(cls) -> "Optional[MDNSSpec]":
        """An always-on mDNS / DNS-SD discovery spec, or ``None``."""
        return None

    @classmethod
    def subnet_spec(cls) -> "Optional[SubnetScanSpec]":
        """An on-demand active subnet-scan discovery spec, or ``None``."""
        return None

    @classmethod
    def network_services(cls) -> "tuple[NetworkServiceSpec, ...]":
        """Declarative LAN services useful for discovery, onboarding, and debug.

        This is intentionally separate from catalogue ``supported_transports``:
        those describe product capability, while this describes concrete local
        network endpoints a diagnostic/scanner may check.
        """
        return ()

    @classmethod
    def discover(cls) -> "Optional[Callable[[float], Awaitable[list]]]":
        """An ``async discover(timeout)`` listing devices via the shared discovery
        service, or ``None`` if this type does not discover over the LAN.

        The default works for every type that declares a discovery spec
        (multicast / mDNS / subnet): scan the shared service under this type's
        :attr:`KEY`, map records to neutral
        :class:`~simplyprint_ws_client.device.discovery.device.DiscoveredDevice` s,
        and pass each through :meth:`refine_discovered`. Types with no discovery
        spec return ``None`` -- so "can this type discover?" is asked as
        ``spec.discover() is not None``.
        """
        if not (
            cls.provides("multicast_spec")
            or cls.provides("mdns_spec")
            or cls.provides("subnet_spec")
        ):
            return None

        async def _discover(timeout: float) -> list:
            from simplyprint_ws_client.device.discovery.active import (
                active_discovery_service,
            )
            from simplyprint_ws_client.device.discovery.device import DiscoveredDevice

            records = await active_discovery_service().scan(cls.KEY, timeout)
            devices = (
                DiscoveredDevice(
                    host=record.host,
                    name=record.name,
                    serial=record.serial,
                    extra=dict(record.extra),
                )
                for record in records
            )
            refined = (cls.refine_discovered(device) for device in devices)
            return [device for device in refined if device is not None]

        return _discover

    @classmethod
    def refine_discovered(cls, device):
        """Polish (or drop, by returning ``None``) one discovered device before it
        reaches the candidate surface -- probe it, enrich the model name, ... The
        default keeps it as-is."""
        return device

    @classmethod
    def add_printer_flow(cls) -> "Optional[Flow]":
        """The guided add-printer :class:`Flow`, or ``None``."""
        return None

    @classmethod
    def account_login_flow(cls) -> "Optional[Flow]":
        """The guided account-login :class:`Flow` (cloud brands only), or ``None``."""
        return None

    @classmethod
    def register_tasks(cls, task_registry: "TaskRegistry") -> None:
        """Contribute periodic / on-demand ``TaskSpec`` s to the shared scheduler.
        No-op for client types with no background tasks."""

    @classmethod
    def printer_presentation(cls, config: "PrinterConfig") -> "PrinterPresentation":
        """Public printer-card presentation for one persisted config.

        The default is intentionally boring and metadata-driven. Brands that need
        to hide secrets, expose editable fields, or resolve model-specific photos
        override this hook in their own spec.
        """
        from simplyprint_ws_client.integration.presentation import (
            default_printer_presentation,
        )

        return default_printer_presentation(cls.metadata.image_url)

    @classmethod
    def provides(cls, capability: str) -> bool:
        """True if this type overrides the named capability hook (vs the base
        no-op) -- a cheap class-level check, so a surface can list the types that
        back it without building any of them.

        Note ``discover`` has a working default driven by the discovery-spec
        hooks: ask ``spec.discover() is not None`` instead of
        ``provides("discover")``.
        """
        own = getattr(cls, capability)
        base = getattr(PrinterSpec, capability)
        return getattr(own, "__func__", own) is not getattr(base, "__func__", base)
