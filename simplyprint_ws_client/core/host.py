"""The headless host: one owner for startup order, shutdown order, and the
add-printer seam.

A :class:`Host` composes the things a running integration needs — the client
fleet (:class:`~simplyprint_ws_client.core.app.ClientApp`), the discovery
service, and each type's background services — without owning any thread
itself: every member owns its own sanctioned lifecycle, the Host owns *order*
(start phases forward, stop reversed) and the seams between them.

A vendor with only the library runs headless in a few lines::

    registry = SpecRegistry.of(MY_INTEGRATION)
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

from simplyprint_ws_client.core.app import ClientApp
from simplyprint_ws_client.core.registry import SpecRegistry
from simplyprint_ws_client.core.settings import ClientSettings

if TYPE_CHECKING:
    from simplyprint_ws_client.core.client import Client
    from simplyprint_ws_client.core.config import PrinterConfig
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.accounts import AccountProvider
    from simplyprint_ws_client.core.client_context import BackgroundService

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
        registered = registry.values()
        if settings.integrations is None:
            settings.integrations = registered
        elif tuple(settings.integrations) != registered:
            raise ValueError(
                "Host registry and ClientSettings integrations must match exactly."
            )
        if settings.camera_protocols is None:
            protocols = registry.camera_protocols()
            settings.camera_protocols = list(protocols) or None
        from simplyprint_ws_client.integration.discovery import DiscoveryService

        self.discovery = DiscoveryService(
            self.registry.multicast_specs(),
            self.registry.subnet_specs(),
            self.registry.network_services(),
            mdns_specs=self.registry.mdns_specs(),
        )
        self.services: Dict[str, "BackgroundService"] = {}
        self.account_providers: Dict[str, "AccountProvider"] = {
            integration_id: integration.account_provider_factory(
                integration.config_manager_t or settings.config_manager_t
            )
            for integration_id, integration in self.registry.integrations().items()
            if integration.account_provider_factory is not None
        }
        self.app = ClientApp(
            settings,
            discovery_service=self.discovery,
            account_providers=self.account_providers,
            background_services=self.services,
        )
        self._service_sink = service_sink

    def start(self, *, detach_fleet: bool = True) -> None:
        """Start phases forward: discovery -> background services -> the fleet.

        ``detach_fleet=True`` runs the fleet on its own thread and returns (an
        app interleaves its own startup around it); ``False`` blocks here until
        the fleet stops.
        """
        self.start_discovery()
        self.start_services()
        if detach_fleet:
            self.app.run_detached()
        else:
            self.app.run_blocking()

    def stop(self) -> None:
        """Stop phases reversed; safe to call once whatever start reached."""
        self.app.stop()
        if self._service_sink is None:
            for key in reversed(list(self.services)):
                service = self.services.pop(key)
                try:
                    service.stop()
                except Exception:  # noqa: BLE001 -- one bad service must not block the rest
                    pass
        self.stop_discovery()

    def start_discovery(self) -> None:
        """Start the one self-supervising service composed with this host."""
        self.discovery.start()

    def stop_discovery(self) -> None:
        """Stop the owned service; it can be started again idempotently."""
        self.discovery.stop()

    def start_services(self, event_loop_provider=None) -> None:
        """Construct every type's background service once (idempotent) and route
        it to the owning sink."""
        for integration_id, integration in self.registry.integrations().items():
            if integration_id in self.services:
                continue
            factory = integration.background_service_factory
            if factory is None:
                continue
            service = factory(event_loop_provider)
            if self._service_sink is not None:
                service = self._service_sink(service)
            self.services[integration_id] = service

    def service_for(self, key: str) -> Optional["BackgroundService"]:
        """The background service started for type ``key``, or ``None``."""
        return self.services.get(key)

    async def discover(self, key: str, timeout: float) -> Optional[List]:
        """Devices discoverable for type ``key``.

        ``None`` for an unknown type (a caller maps that to not-found), an empty
        list for a known type with no LAN discovery, otherwise the devices.
        """
        if self.registry.get(key) is None:
            return None
        discover = self.registry.discoverers().get(key)
        return await discover(self.discovery, timeout) if discover is not None else []

    def flow(self, key: str, flow_id: str) -> Optional["Flow"]:
        """The guided flow ``flow_id`` for type ``key``, or ``None``."""
        if self.registry.get(key) is None:
            return None
        return self.registry.flow(key, flow_id, self.app.context_for(key))

    def mqtt_url(self, key: str, config: "PrinterConfig") -> Optional[str]:
        """Resolve diagnostic MQTT credentials through the scoped integration."""
        integration = self.registry.get(key)
        if integration is None or integration.mqtt_url_from_config is None:
            return None
        return integration.mqtt_url_from_config(config, self.app.context_for(key))

    def add_printer(self, key: str, config: "PrinterConfig") -> "Client":
        """THE persist seam: slot id once, hardware de-dup, then the fleet add.

        Raises :class:`DuplicatePrinter` when ``config`` is an already-registered
        printer (matched by hardware identity, else address).
        """
        from simplyprint_ws_client.integration.discovery.identity import (
            assign_unique_id,
        )
        from simplyprint_ws_client.integration.discovery.reconcile import (
            DeviceReconciler,
        )

        manager = self.app.get_config_manager(integration_id=key)
        assign_unique_id(config)
        duplicate = DeviceReconciler(manager).duplicate_of(config)
        if duplicate is not None:
            existing, field, value = duplicate
            raise DuplicatePrinter(existing, field, value)
        return self.app.add(config, integration_id=key)
