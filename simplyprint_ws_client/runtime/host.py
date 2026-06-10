"""The headless host: one owner for startup order, shutdown order, and the
add-printer seam.

A :class:`Host` composes the things a running integration needs — the client
fleet (:class:`~simplyprint_ws_client.runtime.app.ClientApp`), the discovery
service, and each type's background services — without owning any thread
itself: every member owns its own sanctioned lifecycle, the Host owns *order*
(start phases forward, stop reversed) and the seams between them.

A vendor with only the library runs headless in a few lines::

    registry = SpecRegistry.of(MySpec)
    host = Host(registry, ClientSettings(name="vendor",
                                         config_manager_t=ConfigManagerType.JSON))
    host.start(detach_fleet=False)      # blocks; or detach and drive your own loop

Programmatic onboarding without a web layer: ``host.flow(key, "add-printer")``
hands back the guided :class:`Flow` (drive it with ``run_flow``/``advance_flow``),
``host.discover(key, timeout)`` lists LAN devices, and ``host.add_printer(key,
config)`` is THE persist seam — slot id assigned once (never re-keyed),
hardware-identity de-dup, then the fleet add. ``ClientApp.add`` stays dumb on
purpose: the startup config replay can never reach the identity/de-dup code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from simplyprint_ws_client.runtime.app import ClientApp
from simplyprint_ws_client.runtime.registry import SpecRegistry
from simplyprint_ws_client.runtime.settings import ClientSettings

if TYPE_CHECKING:
    from simplyprint_ws_client.cloud.client import Client
    from simplyprint_ws_client.cloud.config import PrinterConfig
    from simplyprint_ws_client.device.discovery import DiscoveryService
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.spec import BackgroundService

__all__ = ["DuplicatePrinter", "Host"]


class DuplicatePrinter(ValueError):
    """``add_printer`` rejected a config that is an already-registered printer.

    Carries what matched so a caller can render a precise message: the existing
    config, the matched field name (``"hardware id"`` or an address field), and
    the offending value.
    """

    def __init__(self, existing, field: str, value: str) -> None:
        super().__init__(f"already registered (matched {field}: {value})")
        self.existing = existing
        self.field = field
        self.value = value


class Host:
    """Composes the headless runtime; owns order, not threads.

    ``service_sink`` decides who STOPS background services: leave it ``None``
    and the Host stops them (newest-first) in :meth:`stop`; inject the app's
    sink (a supervisor's ``add_service``) and that owner stops them — exactly
    one stopper either way.
    """

    def __init__(
        self,
        registry: SpecRegistry,
        settings: ClientSettings,
        *,
        service_sink: Optional[
            Callable[["BackgroundService"], "BackgroundService"]
        ] = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        if settings.client_specs is None:
            settings.client_specs = registry.runtime_specs()
        if settings.camera_protocols is None:
            protocols = registry.camera_protocols(settings.client_specs)
            settings.camera_protocols = list(protocols) or None
        self.app = ClientApp(settings)
        self.discovery: Optional["DiscoveryService"] = None
        self.services: Dict[str, "BackgroundService"] = {}
        self._service_sink = service_sink
        self._fleet_detached = False

    # -- lifecycle phases -------------------------------------------------------

    def start(self, *, detach_fleet: bool = True) -> None:
        """Start phases forward: discovery -> background services -> the fleet.

        ``detach_fleet=True`` runs the fleet on its own thread and returns (an
        app interleaves its own startup around it); ``False`` blocks here until
        the fleet stops.
        """
        self.start_discovery()
        self.start_services()
        if detach_fleet:
            self._fleet_detached = True
            self.app.run_detached()
        else:
            self.app.run_blocking()

    def stop(self) -> None:
        """Stop phases reversed; safe to call once whatever start reached."""
        if self._fleet_detached:
            self.app.stop()
            self._fleet_detached = False
        if self._service_sink is None:
            for key in reversed(list(self.services)):
                service = self.services.pop(key)
                try:
                    service.stop()
                except Exception:  # noqa: BLE001 -- one bad service must not block the rest
                    pass
        self.stop_discovery()

    def start_discovery(self) -> None:
        """Build the one self-supervising discovery service from the registry's
        declared specs and publish it as the process's active service."""
        if self.discovery is not None:
            return
        from simplyprint_ws_client.device.discovery import DiscoveryService
        from simplyprint_ws_client.device.discovery.active import (
            set_active_discovery_service,
        )

        self.discovery = DiscoveryService(
            self.registry.collect("multicast_spec"),
            self.registry.collect("subnet_spec"),
            self.registry.collect_map("network_services"),
            mdns_specs=self.registry.collect("mdns_spec"),
        )
        set_active_discovery_service(self.discovery)
        self.discovery.start()

    def stop_discovery(self) -> None:
        from simplyprint_ws_client.device.discovery.active import (
            set_active_discovery_service,
        )

        discovery = self.discovery
        self.discovery = None
        if discovery is None:
            return
        set_active_discovery_service(None)
        discovery.stop()

    def start_services(self, event_loop_provider=None) -> None:
        """Construct every type's background service once (idempotent) and route
        it to the owning sink."""
        for key, spec in self.registry.types().items():
            if key in self.services:
                continue
            service = spec.background_service(event_loop_provider)
            if service is None:
                continue
            if self._service_sink is not None:
                service = self._service_sink(service)
            self.services[key] = service

    def service_for(self, key: str) -> Optional["BackgroundService"]:
        """The background service started for type ``key``, or ``None``."""
        return self.services.get(key)

    # -- headless onboarding ------------------------------------------------------

    async def discover(self, key: str, timeout: float) -> Optional[List]:
        """Devices discoverable for type ``key``.

        ``None`` for an unknown type (a caller maps that to not-found), an empty
        list for a known type with no LAN discovery, otherwise the devices.
        """
        if self.registry.get(key) is None:
            return None
        discover = self.registry.discoverers().get(key)
        return await discover(timeout) if discover is not None else []

    def flow(self, key: str, flow_id: str) -> Optional["Flow"]:
        """The guided flow ``flow_id`` for type ``key``, or ``None``."""
        return self.registry.flow(key, flow_id)

    def add_printer(self, key: str, config: "PrinterConfig") -> "Client":
        """THE persist seam: slot id once, hardware de-dup, then the fleet add.

        Raises :class:`DuplicatePrinter` when ``config`` is an already-registered
        printer (matched by hardware identity, else address).
        """
        from simplyprint_ws_client.device.discovery.identity import assign_unique_id
        from simplyprint_ws_client.device.discovery.reconcile import DeviceReconciler

        manager = self.app.get_config_manager(client_key=key)
        assign_unique_id(config)
        duplicate = DeviceReconciler(manager).duplicate_of(config)
        if duplicate is not None:
            existing, field, value = duplicate
            raise DuplicatePrinter(existing, field, value)
        return self.app.add(config, client_key=key)
