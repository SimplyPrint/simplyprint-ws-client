"""The single, process-wide owner of LAN discovery.

One injected :class:`DiscoveryService` holds every brand's discovery backend and
a shared event bus. The always-on multicast backends run on a single harness
thread + loop owned by a :class:`DiscoveryServiceHost` (not a thread each);
printer clients subscribe to ``event_bus`` for live device updates and onboarding
reads :meth:`snapshot`. Active subnet scans run on demand via :meth:`scan`.

The service is a thin owner: the host runs + supervises (restarts on death) the
listener coroutines, and the service exposes a single synchronous :meth:`stop` so
the supervisor can shut discovery down like any other background service.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

from simplyprint_ws_client.events import EventBus

from simplyprint_ws_client.integration.discovery.host import DiscoveryServiceHost
from simplyprint_ws_client.integration.discovery.mdns import MDNSDiscoveryBackend
from simplyprint_ws_client.integration.discovery.multicast import (
    MulticastDiscoveryBackend,
)
from simplyprint_ws_client.integration.discovery.network import (
    HostDiagnostic,
    NetworkScanContext,
    diagnostic_status,
    service_diagnostic_check,
)
from simplyprint_ws_client.integration.discovery.spec import (
    MDNSSpec,
    MulticastSpec,
    NetworkServiceSpec,
    SubnetScanSpec,
)
from simplyprint_ws_client.integration.discovery.subnet import SubnetScanBackend


class DiscoveryService:
    def __init__(
        self,
        multicast_specs: Iterable[MulticastSpec] = (),
        subnet_specs: Iterable[SubnetScanSpec] = (),
        network_services: Mapping[str, tuple[NetworkServiceSpec, ...]] | None = None,
        mdns_specs: Iterable[MDNSSpec] = (),
        restart_interval: float = 5.0,
    ) -> None:
        self.logger = logging.getLogger("discovery")
        self.event_bus = EventBus()
        self._multicast = {}
        for spec in multicast_specs:
            if spec.brand in self._multicast:
                raise ValueError(f"duplicate multicast discovery spec: {spec.brand}")
            self._multicast[spec.brand] = MulticastDiscoveryBackend(
                spec, self.event_bus
            )
        self._mdns = {}
        for spec in mdns_specs:
            if spec.brand in self._mdns:
                raise ValueError(f"duplicate mdns discovery spec: {spec.brand}")
            self._mdns[spec.brand] = MDNSDiscoveryBackend(spec, self.event_bus)
        self._restart_interval = restart_interval
        self._host: DiscoveryServiceHost | None = None
        self._subnet = {spec.brand: spec for spec in subnet_specs}
        self._network_services = dict(network_services or {})
        for spec in self._subnet.values():
            if spec.services:
                self._network_services.setdefault(spec.brand, spec.services)
        self._scan_context: NetworkScanContext | None = None
        self._scan_context_users = 0
        self._stopped = False

    def _retain_scan_context(self) -> NetworkScanContext:
        if self._scan_context is None:
            self._scan_context = NetworkScanContext()
        self._scan_context_users += 1
        return self._scan_context

    def _release_scan_context(self, context: NetworkScanContext) -> None:
        self._scan_context_users -= 1
        if self._scan_context_users <= 0 and self._scan_context is context:
            self._scan_context = None
            self._scan_context_users = 0

    def start(self) -> None:
        """Start the host that runs (and restarts on death) every multicast backend."""
        if self._host is not None and not self._stopped:
            return
        self._host = DiscoveryServiceHost(
            list(self._multicast.values()) + list(self._mdns.values()),
            restart_interval=self._restart_interval,
            logger=self.logger,
        )
        self._host.start()
        self._stopped = False

    def is_stopped(self) -> bool:
        """True once :meth:`stop` has shut discovery down."""
        return self._stopped

    def snapshot(self, brand: str) -> list:
        """Devices currently in a brand's passive cache (empty if none)."""
        backend = self._multicast.get(brand) or self._mdns.get(brand)
        return list(backend.get_devices()) if backend is not None else []

    async def scan(self, brand: str, timeout: float = 5.0) -> list:
        """Run a brand's on-demand active subnet scan (empty if it has none)."""
        spec = self._subnet.get(brand)
        if spec is None:
            return []
        context = self._retain_scan_context()
        try:
            return await SubnetScanBackend(spec, context).scan(timeout)
        finally:
            self._release_scan_context(context)

    async def diagnose(
        self, brand: str, host: str, timeout: float = 5.0, *, run_probe: bool = True
    ) -> HostDiagnostic | None:
        """Diagnose one host for a brand's active subnet spec, if it has one."""
        spec = self._subnet.get(brand)
        if spec is not None:
            return await SubnetScanBackend(spec).diagnose(
                host, timeout, run_probe=run_probe
            )

        services = self._network_services.get(brand)
        if not services:
            return None

        context = await NetworkScanContext().host_context(host, services, timeout)
        service_results = tuple(context.services.values())
        checks = tuple(service_diagnostic_check(result) for result in service_results)
        if context.required_services_open:
            return HostDiagnostic(
                brand=brand,
                host=host,
                status=diagnostic_status(checks),
                services=service_results,
                checks=checks,
                matched=None,
                reason="required_services_open",
                message=f"{host} is answering on the expected {brand} port(s)",
            )

        closed = [
            result for result in service_results if result.required and not result.open
        ]
        labels = [
            result.label or f"{result.transport}:{result.port}" for result in closed
        ]
        return HostDiagnostic(
            brand=brand,
            host=host,
            status=diagnostic_status(checks),
            services=service_results,
            checks=checks,
            matched=False,
            reason="required_service_closed",
            message=(
                f"{host} is not answering on the expected {brand} "
                f"service port(s): {', '.join(labels)}"
            ),
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._host is not None:
            self._host.shutdown()
            self._host = None
