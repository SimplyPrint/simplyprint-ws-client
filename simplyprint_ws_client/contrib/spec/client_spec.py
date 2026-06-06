"""The one descriptor an integration writes per client type: ``PrinterClientSpec``.

Every client type an integration ships is described by exactly one
:class:`PrinterClientSpec` subclass -- the single source of truth for that type,
and the *only* descriptor. It *is* a library :class:`ClientSpec` (key, factories,
cameras, name), so the app hands it straight to ``ClientSettings``; and it owns
every per-type surface an app projects -- product metadata, the background
service, discovery specs, guided flows, account capability, periodic tasks -- as
hooks that default to "this type doesn't back that surface".

The subtlety is laziness. ``KEY`` and ``metadata`` are :class:`~typing.ClassVar`
literals and the capability hooks are classmethods, so a surface can ask *which*
client types back it (:meth:`provides`) and read their catalogue facts without
ever calling :meth:`build` -- the one place a type imports its cameras / runtime.
Building the live spec list is the only path that pays for those heavy imports,
and it happens once at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    ClassVar,
    Optional,
    Protocol,
)

from pydantic import BaseModel, ConfigDict

from simplyprint_ws_client.core.settings import ClientSpec

if TYPE_CHECKING:
    from simplyprint_ws_client import PrinterConfig
    from simplyprint_ws_client.contrib.accounts import AccountProvider
    from simplyprint_ws_client.contrib.discovery.spec import (
        MulticastSpec,
        NetworkServiceSpec,
        SubnetScanSpec,
    )
    from simplyprint_ws_client.contrib.flow import Flow
    from simplyprint_ws_client.contrib.presentation import PrinterPresentation
    from simplyprint_ws_client.contrib.tasks import TaskRegistry


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
class PrinterClientSpec(ClientSpec):
    """One client type's single source of truth -- the only descriptor.

    Subclasses live next to their client type's code, set the :attr:`KEY` /
    :attr:`metadata` class literals, implement :meth:`build` to construct the
    runtime spec (lazy camera / config / factory imports), and override only the
    capability hooks they back. The hooks below default to "absent"; a surface
    lists the types that back it via :meth:`provides` and builds none it doesn't
    need.
    """

    #: Catalogue identity, read off the class without building. ``ClientSpec.key``
    #: is the per-instance runtime field; :meth:`build` sets it from this.
    KEY: ClassVar[str]
    #: Declarative product facts, read off the class without building the heavy
    #: runtime spec (so the catalogue never imports a brand's cameras/runtime).
    metadata: ClassVar[ProductMetadata]

    @classmethod
    def build(cls) -> "PrinterClientSpec":
        """Construct the runtime spec for this client type.

        Subclasses do the lazy imports here and return ``cls(key=cls.KEY, ...)``
        with the library ``ClientSpec`` fields (including ``camera_protocols``)
        filled.
        """
        raise NotImplementedError

    # -- capability hooks: default "this type does not back the surface" --------

    @classmethod
    def background_service(cls, event_loop_provider=None) -> "BackgroundService | None":
        """A process-wide background service this client type needs (a connection
        manager or watchdog), or ``None``.

        Constructed once by the app's supervisor at startup and read back by the
        client factory -- never built in the factory, so two factory calls can't
        race to create one. Return a ready-to-register service (already started if
        it needs starting). Read off the *class* (no ``build``).

        ``event_loop_provider`` is the app's loop provider, for a service whose
        transport pool must deliver onto that loop (e.g. a connection manager).
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
        service, or ``None`` if this type does not discover over the LAN."""
        return None

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
        from simplyprint_ws_client.contrib.presentation import (
            default_printer_presentation,
        )

        return default_printer_presentation(cls.metadata.image_url)

    @classmethod
    def provides(cls, capability: str) -> bool:
        """True if this type overrides the named capability hook (vs the base
        no-op) -- a cheap class-level check, so a surface can list the types that
        back it without building any of them."""
        own = getattr(cls, capability)
        base = getattr(PrinterClientSpec, capability)
        return getattr(own, "__func__", own) is not getattr(base, "__func__", base)
