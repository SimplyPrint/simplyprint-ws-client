"""Projection registry for immutable integration descriptors.

Applications name each shipped integration once by collecting its
:class:`~simplyprint_ws_client.integration.spec.IntegrationSpec` value here.
Every runtime surface then projects the corresponding explicit field; there is
no descriptor construction or hook discovery in the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Mapping, Tuple, Type

from simplyprint_ws_client.integration.spec import IntegrationSpec

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.camera.base import BaseCameraProtocol
    from simplyprint_ws_client.integration.discovery.spec import (
        MDNSSpec,
        MulticastSpec,
        NetworkServiceSpec,
        SubnetScanSpec,
    )
    from simplyprint_ws_client.core.client_context import ClientContext
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.spec import ProductMetadata
    from simplyprint_ws_client.integration.tasks import TaskRegistry

__all__ = ["SpecRegistry"]


class SpecRegistry:
    """The ordered ``{integration id: IntegrationSpec}`` map and projections."""

    def __init__(self, integrations: Iterable[IntegrationSpec] = ()) -> None:
        self._integrations: Dict[str, IntegrationSpec] = {}
        for integration in integrations:
            self.register(integration)

    @classmethod
    def of(cls, *integrations: IntegrationSpec) -> "SpecRegistry":
        return cls(integrations)

    def register(self, integration: IntegrationSpec) -> None:
        integration_id = str(integration.id)
        if integration_id in self._integrations:
            raise ValueError(f"duplicate integration id: {integration_id!r}")
        self._integrations[integration_id] = integration

    def integrations(self) -> Mapping[str, IntegrationSpec]:
        return dict(self._integrations)

    def get(self, integration_id: str) -> IntegrationSpec | None:
        return self._integrations.get(integration_id)

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._integrations)

    def values(self) -> Tuple[IntegrationSpec, ...]:
        return tuple(self._integrations.values())

    def multicast_specs(self) -> Tuple["MulticastSpec", ...]:
        return tuple(
            integration.multicast
            for integration in self._integrations.values()
            if integration.multicast is not None
        )

    def mdns_specs(self) -> Tuple["MDNSSpec", ...]:
        return tuple(
            integration.mdns
            for integration in self._integrations.values()
            if integration.mdns is not None
        )

    def subnet_specs(self) -> Tuple["SubnetScanSpec", ...]:
        return tuple(
            integration.subnet
            for integration in self._integrations.values()
            if integration.subnet is not None
        )

    def network_services(self) -> Dict[str, Tuple["NetworkServiceSpec", ...]]:
        return {
            integration_id: integration.network_services
            for integration_id, integration in self._integrations.items()
            if integration.network_services
        }

    def metadata(self) -> Dict[str, "ProductMetadata"]:
        return {
            integration_id: integration.metadata
            for integration_id, integration in self._integrations.items()
        }

    def camera_protocols(
        self, disabled: bool = False
    ) -> Tuple[Type["BaseCameraProtocol"], ...]:
        if disabled:
            return ()
        return tuple(
            protocol
            for integration in self._integrations.values()
            for protocol in integration.camera_protocols()
        )

    def register_tasks(self, task_registry: "TaskRegistry") -> None:
        for integration in self._integrations.values():
            if integration.register_tasks is not None:
                integration.register_tasks(task_registry)

    @staticmethod
    def _flow_factories(
        integration: IntegrationSpec,
    ) -> Tuple[Tuple[str, Callable[["ClientContext"], "Flow"]], ...]:
        return tuple(
            (flow_id, factory)
            for flow_id, factory in (
                ("add-printer", integration.add_printer_flow_factory),
                ("account-login", integration.account_login_flow_factory),
            )
            if factory is not None
        )

    def flow(
        self,
        integration_id: str,
        flow_id: str,
        context: "ClientContext",
    ) -> "Flow | None":
        integration = self._integrations.get(integration_id)
        if integration is None:
            return None
        factory = next(
            (
                factory
                for registered_id, factory in self._flow_factories(integration)
                if registered_id == flow_id
            ),
            None,
        )
        return factory(context) if factory is not None else None

    def brand_flows(self, integration_id: str) -> List[str]:
        integration = self._integrations.get(integration_id)
        if integration is None:
            return []
        return [flow_id for flow_id, _ in self._flow_factories(integration)]

    def list_flows(self) -> Dict[str, List[str]]:
        listing = {
            integration_id: self.brand_flows(integration_id)
            for integration_id in self._integrations
        }
        return {
            integration_id: flows for integration_id, flows in listing.items() if flows
        }

    def flow_brands(self) -> List[str]:
        return sorted(
            integration_id
            for integration_id, integration in self._integrations.items()
            if integration.add_printer_flow_factory is not None
        )

    def discoverers(self) -> Dict[str, Callable]:
        return {
            integration_id: integration.discover
            for integration_id, integration in self._integrations.items()
            if integration.discover is not None
        }
