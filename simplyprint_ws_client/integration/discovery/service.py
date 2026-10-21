"""The Host-owned LAN discovery service.

One explicitly injected :class:`DiscoveryService` holds every brand's discovery backend and
a shared event bus. The always-on multicast backends run on a single harness
thread + loop owned by a :class:`DiscoveryServiceHost` (not a thread each);
printer clients subscribe to ``event_bus`` for live device updates and onboarding
reads :meth:`snapshot`. Active subnet scans run on demand via :meth:`scan`.

The service is a thin owner: the host runs + supervises (restarts on death) the
listener coroutines, and the service exposes a single synchronous :meth:`stop` so
the supervisor can shut discovery down like any other background service.

:meth:`DiscoveryService.add` contributes one integration's discovery under a
:class:`~simplyprint_ws_client.core.scope.Scope`, live: its listeners join the
running host loop, and closing the scope stops them and drops its subnet and
service entries. The maps are copy-on-write, so readers never lock.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

from simplyprint_ws_client.common.asyncio.concurrent import await_concurrent_future

from simplyprint_ws_client.events import EventBus

from simplyprint_ws_client.integration.discovery.host import DiscoveryServiceHost
from simplyprint_ws_client.integration.discovery.mdns import MDNSDiscoveryBackend
from simplyprint_ws_client.integration.discovery.multicast import (
    MulticastDiscoveryBackend,
)
from simplyprint_ws_client.integration.discovery.network import (
    DiagnosticReason,
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

if TYPE_CHECKING:
    from simplyprint_ws_client.core.scope import Scope
    from simplyprint_ws_client.integration.spec import IntegrationSpec


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
        # Serializes add/remove against start/stop. The maps themselves are
        # swapped whole (copy-on-write) so readers need no lock.
        self._lock = threading.Lock()

    def add(self, spec: "IntegrationSpec", scope: "Scope") -> None:
        """Contribute ``spec``'s discovery until ``scope`` closes.

        Adds its multicast and mDNS listeners (launched on the running host loop
        at once, or with the rest at start), its subnet scan and its network
        services. Closing the scope stops the listeners on the host loop and
        waits for them, then drops the entries. Raises on a key already taken.
        """
        integration_id = str(spec.id)
        multicast = (
            MulticastDiscoveryBackend(spec.multicast, self.event_bus)
            if spec.multicast is not None
            else None
        )
        mdns = (
            MDNSDiscoveryBackend(spec.mdns, self.event_bus)
            if spec.mdns is not None
            else None
        )
        subnet = spec.subnet
        services: Dict[str, Any] = {}
        if spec.network_services:
            services[integration_id] = spec.network_services
        if subnet is not None and subnet.services:
            services.setdefault(subnet.brand, subnet.services)

        with self._lock:
            if multicast is not None and multicast.spec.brand in self._multicast:
                raise ValueError(
                    f"duplicate multicast discovery spec: {multicast.spec.brand}"
                )
            if mdns is not None and mdns.spec.brand in self._mdns:
                raise ValueError(f"duplicate mdns discovery spec: {mdns.spec.brand}")
            if subnet is not None and subnet.brand in self._subnet:
                raise ValueError(f"duplicate subnet discovery spec: {subnet.brand}")
            services = {
                key: value
                for key, value in services.items()
                if key not in self._network_services
            }
            if multicast is not None:
                self._multicast = {**self._multicast, multicast.spec.brand: multicast}
            if mdns is not None:
                self._mdns = {**self._mdns, mdns.spec.brand: mdns}
            if subnet is not None:
                self._subnet = {**self._subnet, subnet.brand: subnet}
            if services:
                self._network_services = {**self._network_services, **services}
            host = self._host if not self._stopped else None
            backends = [b for b in (multicast, mdns) if b is not None]
            if host is not None:
                for backend in backends:
                    host.add_backend(backend)

        scope.on_close(
            lambda: self._remove(backends, subnet, services),
            name=f"discovery:{integration_id}",
        )

    def _remove(
        self,
        backends: List[Any],
        subnet: Optional[SubnetScanSpec],
        services: Mapping[str, Any],
    ):
        with self._lock:
            multicast = {
                k: v for k, v in self._multicast.items() if v not in backends
            }
            mdns = {k: v for k, v in self._mdns.items() if v not in backends}
            self._multicast, self._mdns = multicast, mdns
            if subnet is not None and self._subnet.get(subnet.brand) is subnet:
                self._subnet = {
                    k: v for k, v in self._subnet.items() if k != subnet.brand
                }
            self._network_services = {
                k: v
                for k, v in self._network_services.items()
                if not (k in services and services[k] is v)
            }
            host = self._host if not self._stopped else None
        if host is None:
            for backend in backends:
                backend.stop()
            return None
        futures = [
            future
            for future in (host.remove_backend(backend) for backend in backends)
            if future is not None
        ]
        if not futures:
            return None

        async def wait() -> None:
            for future in futures:
                await await_concurrent_future(future)

        return wait()

    def backend_keys(self) -> List[str]:
        """Keys of the multicast and mDNS listeners currently contributed."""
        return list(self._multicast) + list(self._mdns)

    def subnet_keys(self) -> List[str]:
        return list(self._subnet)

    def network_service_keys(self) -> List[str]:
        return list(self._network_services)

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
        with self._lock:
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
                reason=DiagnosticReason.REQUIRED_SERVICES_OPEN,
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
            reason=DiagnosticReason.REQUIRED_SERVICE_CLOSED,
            message=(
                f"{host} is not answering on the expected {brand} "
                f"service port(s): {', '.join(labels)}"
            ),
        )

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            host, self._host = self._host, None
        if host is not None:
            host.shutdown()
