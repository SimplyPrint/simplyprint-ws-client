"""The headless host: one owner for startup order, shutdown order, and the
add-printer seam.

A :class:`Host` composes the things a running integration needs — the client
fleet (:class:`~simplyprint_ws_client.core.app.ClientApp`), the discovery
service, and each type's background services — without owning any thread
itself: every member owns its own sanctioned lifecycle, the Host owns *order*
(start phases forward, stop reversed) and the seams between them.

Everything an integration contributes is MOUNTED: ``host.mount(spec, parent)``
registers it under a child :class:`~simplyprint_ws_client.core.scope.Scope` of
``parent`` and returns that scope; closing it undoes the lot, newest first.
Startup is the same call: the Host mounts every spec of its registry under its
own :attr:`Host.root` scope at construction, and closes that scope at
:meth:`Host.stop`. A plugin mounted later under ``host.root`` (or any scope
below it) is unloaded by closing its scope, with no restart.

A mount registers, in this order (so teardown runs it backwards and stops the
printers first, the registry entry last):

1. the registry entry;
2. the account provider;
3. the background service (built now if services already started, else by
   :meth:`Host.start_services`);
4. the camera protocols (when the settings did not fix them);
5. discovery, plus (with ``IntegrationSpec.rediscover``) a watch on its
   always-on listeners that routes each sighting to the printer it identifies;
6. tasks, under a ``"<id>."`` prefix, once a task scheduler is attached;
7. the optional ``setup`` hook;
8. the printers, through :meth:`ClientApp.mount_domain`.

A mount is transactional: if any step raises, the partial scope is closed and
the error re-raised.

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

import threading
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from simplyprint_ws_client.core.app import ClientApp
from simplyprint_ws_client.core.registry import SpecRegistry
from simplyprint_ws_client.core.scope import CloseReport, Scope
from simplyprint_ws_client.core.settings import ClientSettings

if TYPE_CHECKING:
    from simplyprint_ws_client.core.client import Client
    from simplyprint_ws_client.core.config import PrinterConfig
    from simplyprint_ws_client.integration.flow import Flow
    from simplyprint_ws_client.integration.accounts import AccountProvider
    from simplyprint_ws_client.core.client_context import BackgroundService
    from simplyprint_ws_client.integration.spec import IntegrationSpec
    from simplyprint_ws_client.integration.tasks import SchedulerService

__all__ = ["DuplicatePrinter", "Host"]

_V = TypeVar("_V")


class _LiveMapping(Mapping[str, _V]):
    """A read-only view of whichever dict the Host currently holds.

    The Host swaps its service and provider maps whole on mount/unmount; the
    app keeps this view, so it always reads the current one.
    """

    def __init__(self, read: Callable[[], Mapping[str, _V]]) -> None:
        self._read = read

    def __getitem__(self, key: str) -> _V:
        return self._read()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())


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

    ``service_sink`` decides who STOPS background services at :meth:`stop`:
    leave it ``None`` and the Host stops them (newest-first); inject the app's
    sink (a supervisor's ``add_service``) and that owner stops them — exactly
    one stopper either way. Closing one mount's scope before that always stops
    its service.

    ``task_scheduler`` (or a later :meth:`attach_task_scheduler`) is where
    mounted integrations' tasks go; without one, mounts register no tasks.
    """

    def __init__(
        self,
        registry: SpecRegistry,
        settings: ClientSettings,
        *,
        service_sink: Optional[
            Callable[["BackgroundService"], "BackgroundService"]
        ] = None,
        task_scheduler: Optional["SchedulerService"] = None,
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
        # Camera protocols set in the settings win, as before; otherwise each
        # mount contributes its own, scoped.
        self._mount_cameras = settings.camera_protocols is None
        from simplyprint_ws_client.integration.discovery import DiscoveryService

        self._lock = threading.Lock()
        # Serializes building background services (the factory runs unlocked
        # from the maps, but never twice for one id).
        self._service_build_lock = threading.Lock()
        self._service_sink = service_sink
        self._service_provider: Optional[object] = None
        self._services_started = False
        self._task_scheduler = task_scheduler
        self._stopping = False
        #: Ids with a mount in progress or done (the duplicate guard).
        self._reserved: Set[str] = set()
        #: Mounted specs whose service slot is registered (start_services reads it).
        self._mounts: Dict[str, "IntegrationSpec"] = {}
        #: Each mount's task slot, filled when a task scheduler is attached.
        self._task_slots: Dict[str, Tuple["IntegrationSpec", Scope]] = {}
        self._scopes: Dict[str, Scope] = {}

        self.discovery = DiscoveryService()
        # Swapped whole on mount/unmount; the app reads them through live views.
        self.services: Dict[str, "BackgroundService"] = {}
        self.account_providers: Dict[str, "AccountProvider"] = {}
        self.app = ClientApp(
            settings,
            discovery_service=self.discovery,
            account_providers=_LiveMapping(lambda: self.account_providers),
            background_services=_LiveMapping(lambda: self.services),
        )
        #: The scope every startup mount lives under; closed by :meth:`stop`.
        self.root = Scope("host", loop=self.app._app_event_loop)
        try:
            for integration in registered:
                self.mount(integration, self.root)
        except BaseException:
            self.root.close("host construction failed")
            self.app.stop()
            raise
        if self._mount_cameras:
            # Kept for readers of the settings: what startup contributed.
            settings.camera_protocols = list(registry.camera_protocols()) or None

    # ------------------------------------------------------------------ mounting

    def mount(
        self,
        spec: "IntegrationSpec",
        parent: Scope,
        *,
        setup: Optional[Callable[[Scope], None]] = None,
    ) -> Scope:
        """Mount ``spec`` under a child of ``parent`` and return that scope.

        Registers everything the spec contributes in the order of the module
        doc, so closing the scope stops its printers first. ``setup``, if
        given, is called with the scope after the tasks and before the
        printers. Transactional: if a step raises, the partial scope is closed
        and the error re-raised.
        """
        key = str(spec.id)
        with self._lock:
            if key in self._reserved:
                raise ValueError(f"integration {key!r} is already mounted")
            self._reserved.add(key)
        try:
            scope = parent.child(key)
        except BaseException:
            with self._lock:
                self._reserved.discard(key)
            raise
        try:
            scope.on_close(lambda: self._release(key, scope), name=f"host:{key}")
            with self._lock:
                self._scopes = {**self._scopes, key: scope}
            # 1. registry
            self.registry.add(spec, scope)
            # 2. account provider
            self._mount_account_provider(spec, scope)
            # 3. background service
            self._mount_service(spec, scope)
            # 4. camera protocols
            pool = self.app.camera_pool
            if self._mount_cameras and pool is not None:
                pool.add_protocols(spec.camera_protocols(), scope)
            # 5. discovery (and, with ``rediscover``, its sightings routed
            # to the printer that moved)
            self.discovery.add(spec, scope)
            self._mount_rediscovery(spec, scope)
            # 6. tasks
            self._mount_tasks(spec, scope)
            # 7. setup
            if setup is not None:
                setup(scope)
            # 8. printers, last
            self.app.mount_domain(spec, scope)
        except BaseException as error:
            report = scope.close(f"mount failed: {type(error).__name__}: {error}")
            for undo_error in report.errors:
                self.app.logger.warning(
                    "undoing the failed mount of %s also failed",
                    key,
                    exc_info=undo_error,
                )
            raise
        return scope

    def scope_for(self, key: str) -> Optional[Scope]:
        """The scope ``key`` is mounted under, or ``None``."""
        return self._scopes.get(key)

    def _release(self, key: str, scope: Scope) -> None:
        with self._lock:
            if self._scopes.get(key) is scope:
                self._scopes = {k: v for k, v in self._scopes.items() if k != key}
            self._reserved.discard(key)

    def _mount_rediscovery(self, spec: "IntegrationSpec", scope: Scope) -> None:
        """Watch ``spec``'s always-on listeners and route each sighting, as a
        device, to the running printer with its identity (see
        :meth:`ClientApp.route_discovered`). Nothing without ``rediscover``."""
        convert = spec.rediscover
        if convert is None:
            return
        key = str(spec.id)
        app = self.app

        def route(record) -> None:
            device = convert(record)
            if device is not None:
                app.route_discovered(key, device)

        for listener_key in self.discovery.listener_keys_of(spec):
            self.discovery.watch(listener_key, route, scope)

    def _mount_account_provider(self, spec: "IntegrationSpec", scope: Scope) -> None:
        factory = spec.account_provider_factory
        if factory is None:
            return
        key = str(spec.id)
        provider = factory(spec.config_manager_t or self.settings.config_manager_t)
        with self._lock:
            self.account_providers = {**self.account_providers, key: provider}

        def drop() -> None:
            with self._lock:
                if self.account_providers.get(key) is provider:
                    self.account_providers = {
                        k: v for k, v in self.account_providers.items() if k != key
                    }

        scope.on_close(drop, name=f"account-provider:{key}")

    def _mount_service(self, spec: "IntegrationSpec", scope: Scope) -> None:
        key = str(spec.id)
        scope.on_close(lambda: self._unmount_service(key, spec), name=f"service:{key}")
        with self._lock:
            self._mounts = {**self._mounts, key: spec}
            build = self._services_started
        if build:
            self._build_service(spec)

    def _build_service(self, spec: "IntegrationSpec") -> None:
        factory = spec.background_service_factory
        if factory is None:
            return
        key = str(spec.id)
        with self._service_build_lock:
            with self._lock:
                if key in self.services or self._mounts.get(key) is not spec:
                    return
                provider = self._service_provider
            service = factory(provider)
            if self._service_sink is not None:
                service = self._service_sink(service)
            with self._lock:
                mounted = self._mounts.get(key) is spec
                if mounted:
                    self.services = {**self.services, key: service}
        if not mounted:  # unmounted while its factory ran
            service.stop()

    def _unmount_service(self, key: str, spec: "IntegrationSpec") -> None:
        with self._lock:
            if self._mounts.get(key) is spec:
                self._mounts = {k: v for k, v in self._mounts.items() if k != key}
        with self._service_build_lock:
            with self._lock:
                service = self.services.get(key)
                if service is None:
                    return
                self.services = {k: v for k, v in self.services.items() if k != key}
                sink_stops_it = self._stopping and self._service_sink is not None
        if not sink_stops_it:
            service.stop()

    def _mount_tasks(self, spec: "IntegrationSpec", scope: Scope) -> None:
        key = str(spec.id)
        slot = scope.slot("tasks")

        def forget() -> None:
            with self._lock:
                if self._task_slots.get(key, (None, None))[1] is slot:
                    self._task_slots = {
                        k: v for k, v in self._task_slots.items() if k != key
                    }

        # Registered after the slot, so it runs just before the slot closes.
        scope.on_close(forget, name=f"task-slot:{key}")
        with self._lock:
            self._task_slots = {**self._task_slots, key: (spec, slot)}
            scheduler = self._task_scheduler
        if scheduler is not None:
            self._register_tasks(spec, slot, scheduler)

    @staticmethod
    def _register_tasks(
        spec: "IntegrationSpec", slot: Scope, scheduler: "SchedulerService"
    ) -> None:
        if spec.register_tasks is None:
            return
        from simplyprint_ws_client.integration.tasks import ScopedTaskRegistry

        spec.register_tasks(ScopedTaskRegistry(scheduler, slot, prefix=str(spec.id)))

    def attach_task_scheduler(self, scheduler: "SchedulerService") -> None:
        """Route every mount's tasks to ``scheduler`` (once), current and future.

        Mounts made before this register their tasks now, each in its own
        mount's place in the teardown order.
        """
        with self._lock:
            if self._task_scheduler is not None:
                if self._task_scheduler is scheduler:
                    return
                raise ValueError("a task scheduler is already attached")
            self._task_scheduler = scheduler
            slots = list(self._task_slots.values())
        for spec, slot in slots:
            if not slot.closed:
                self._register_tasks(spec, slot, scheduler)

    # ----------------------------------------------------------------- lifecycle

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

    def stop(self) -> CloseReport:
        """Stop phases reversed; safe to call once whatever start reached.

        The fleet stops first (every printer), then the root scope closes,
        newest mount first: services (unless the sink owns them), camera
        protocols, discovery entries, registry entries. Discovery stops last.
        Stored configs are kept. Returns the root scope's close report.
        """
        self.app.stop()
        with self._lock:
            self._stopping = True
        report = self.root.close("host stopped")
        self.stop_discovery()
        return report

    def start_discovery(self) -> None:
        """Start the one self-supervising service composed with this host."""
        self.discovery.start()

    def stop_discovery(self) -> None:
        """Stop the owned service; it can be started again idempotently."""
        self.discovery.stop()

    def start_services(self, event_loop_provider=None) -> None:
        """Construct every mounted type's background service once (idempotent)
        and route it to the owning sink. Later mounts build theirs at mount."""
        with self._lock:
            if not self._services_started:
                self._service_provider = event_loop_provider
                self._services_started = True
            specs = list(self._mounts.values())
        for spec in specs:
            self._build_service(spec)

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
