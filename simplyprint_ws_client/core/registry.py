"""The spec registry: every client type an app ships, collected once.

An app (or a single-vendor integration) constructs one :class:`SpecRegistry`
from its :class:`~simplyprint_ws_client.integration.spec.PrinterSpec` classes —
the single place its types are listed — and every capability surface projects
off it: live runtime specs, catalogue metadata, flows, discovery, accounts,
periodic tasks. No type is named in here; the registry only iterates what it
was given, and each spec's hooks import their modules lazily, so projecting
never drags in a type's runtime.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

from simplyprint_ws_client.integration.spec import PrinterSpec

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.accounts import AccountProvider
    from simplyprint_ws_client.integration.camera.base import BaseCameraProtocol
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.spec import ProductMetadata
    from simplyprint_ws_client.integration.tasks import TaskRegistry

__all__ = ["FLOW_FIELDS", "SpecRegistry"]

#: Stable flow ids (the wire/route names) -> the ``PrinterSpec`` hook that
#: builds them. The id matches the ``Flow.id`` each builder sets.
FLOW_FIELDS = {
    "add-printer": "add_printer_flow",
    "account-login": "account_login_flow",
}


class SpecRegistry:
    """The collected ``{key: PrinterSpec class}`` map and its projections."""

    def __init__(self, specs: Iterable[Type[PrinterSpec]] = ()) -> None:
        self._types: Dict[str, Type[PrinterSpec]] = {}
        self._built: Optional[Tuple[PrinterSpec, ...]] = None
        for spec in specs:
            self.register(spec)

    @classmethod
    def of(cls, *specs: Type[PrinterSpec]) -> "SpecRegistry":
        """The explicit constructor an app's one type-naming module calls."""
        return cls(specs)

    def register(self, spec: Type[PrinterSpec]) -> None:
        """Add one spec class; duplicate keys are a wiring bug and raise."""
        key = spec.KEY
        if key in self._types:
            raise ValueError(f"duplicate client type key: {key!r}")
        self._types[key] = spec

    def types(self) -> Mapping[str, Type[PrinterSpec]]:
        """The ``{key: spec class}`` map, in registration order."""
        return dict(self._types)

    def get(self, key: str) -> Optional[Type[PrinterSpec]]:
        return self._types.get(key)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._types)

    def collect(self, capability: str) -> list:
        """``[spec.cap() for every type that backs cap]`` — the one loop the old
        per-surface registries each re-wrote."""
        return [
            getattr(spec, capability)()
            for spec in self._types.values()
            if spec.provides(capability)
        ]

    def collect_map(self, capability: str) -> dict:
        """Like :meth:`collect`, keyed by client type."""
        return {
            key: getattr(spec, capability)()
            for key, spec in self._types.items()
            if spec.provides(capability)
        }

    def runtime_specs(self) -> Tuple[PrinterSpec, ...]:
        """The live client specs, one per type, built once (the lazy boundary)."""
        if self._built is None:
            self._built = tuple(spec.build() for spec in self._types.values())
        return self._built

    def metadata(self) -> Dict[str, "ProductMetadata"]:
        """Catalogue product metadata keyed by client type, without building."""
        return {key: spec.metadata for key, spec in self._types.items()}

    @staticmethod
    def camera_protocols(
        clients: Iterable[PrinterSpec], disabled: bool = False
    ) -> Tuple[Type["BaseCameraProtocol"], ...]:
        if disabled:
            return ()
        return tuple(
            protocol for client in clients for protocol in client.camera_protocols
        )

    def register_tasks(self, task_registry: "TaskRegistry") -> None:
        """Let every type contribute its periodic / on-demand tasks once."""
        for spec in self._types.values():
            if spec.provides("register_tasks"):
                spec.register_tasks(task_registry)

    def flow(self, key: str, flow_id: str) -> Optional["Flow"]:
        """Build ``flow_id`` for type ``key``, or ``None`` if it has no such flow."""
        spec = self._types.get(key)
        field = FLOW_FIELDS.get(flow_id)
        if spec is None or field is None or not spec.provides(field):
            return None
        return getattr(spec, field)()

    def brand_flows(self, key: str) -> List[str]:
        """The flow ids type ``key`` exposes (in declaration order)."""
        spec = self._types.get(key)
        if spec is None:
            return []
        return [
            flow_id for flow_id, field in FLOW_FIELDS.items() if spec.provides(field)
        ]

    def list_flows(self) -> Dict[str, List[str]]:
        """Every type that exposes at least one flow, mapped to its flow ids."""
        listing = {key: self.brand_flows(key) for key in self._types}
        return {key: flows for key, flows in listing.items() if flows}

    def flow_brands(self) -> List[str]:
        """Client types that expose a guided add-printer flow."""
        return sorted(
            key
            for key, spec in self._types.items()
            if spec.provides("add_printer_flow")
        )

    def discoverers(self) -> Dict[str, Callable]:
        """``{key: async discover(timeout)}`` for every type that can list LAN
        devices (overridden or the spec default — asked as ``discover() is not
        None``, never ``provides``)."""
        out: Dict[str, Callable] = {}
        for key, spec in self._types.items():
            discover = spec.discover()
            if discover is not None:
                out[key] = discover
        return out

    def account_providers(self) -> Dict[str, Callable[[], "AccountProvider"]]:
        """Account-capability factories keyed by client type (factories, so
        listing never builds a provider)."""
        return {
            key: spec.account_provider
            for key, spec in self._types.items()
            if spec.provides("account_provider")
        }
