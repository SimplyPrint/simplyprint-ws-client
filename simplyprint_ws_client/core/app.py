__all__ = ["ClientApp", "ClientStartFailure"]

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional

from simplyprint_ws_client.core.client import (
    Client,
    ClientConfigChangedEvent,
    ClientStateChangeEvent,
)
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.client_context import (
    BackgroundService,
    ClientContext,
)
from simplyprint_ws_client.core.config import ConfigManager
from simplyprint_ws_client.core.config.flusher import ConfigFlusher
from simplyprint_ws_client.core.manager import ClientList
from simplyprint_ws_client.core.protocol.connection import (
    TransportFactory,
    default_transport_factory,
)
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.common.asyncio.event_loop_runner import Runner
from simplyprint_ws_client.common.asyncio.offload import (
    Offload,
    install_default_executor,
)
from simplyprint_ws_client.common.hardware.physical_machine import (
    make_host_telemetry_reader,
)
from simplyprint_ws_client.integration.camera.pool import CameraPool
from simplyprint_ws_client.integration.spec import IntegrationSpec
from simplyprint_ws_client.core.api.sentry import Sentry
from simplyprint_ws_client.core.api.simplyprint_api import SimplyPrintApi
from simplyprint_ws_client.common.utils.stoppable import SyncStoppable

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.accounts import AccountProvider
    from simplyprint_ws_client.integration.discovery.service import DiscoveryService


@dataclass(frozen=True)
class ClientStartFailure:
    """Why one stored printer is not running.

    One printer's broken driver or bad config must never stop the others, so
    the app skips it, keeps its config on disk, and records it here for the UI
    and diagnostics to show.
    """

    unique_id: str
    integration_id: str
    #: ``"config"``: the stored entry could not be parsed (retried on restart).
    #: ``"build"``: persisting it or ``client_factory`` raised (retried on the
    #: next ``add`` of this printer, e.g. after a config edit, or on restart).
    stage: str
    error: str


class ClientApp(SyncStoppable):
    settings: ClientSettings
    client_list: ClientList
    scheduler: Scheduler
    config_manager: ConfigManager[PrinterConfig]
    config_managers: Dict[str, ConfigManager[PrinterConfig]]
    #: One coalesced flusher per config manager; the change-event listener
    #: triggers these instead of flushing inline on the loop.
    config_flushers: Dict[str, ConfigFlusher]
    integrations: Dict[str, IntegrationSpec]
    camera_pool: Optional[CameraPool] = None
    #: The app's bounded blocking-work lanes; the only sanctioned hop for a
    #: blocking call off the loop. Created unconditionally, shut down last.
    offload: Offload
    client_context: ClientContext
    #: Printers that could not be started, by unique id. Written under
    #: ``_app_lock``; read it through :meth:`failed_client_ids` or copy it.
    failed_clients: Dict[str, ClientStartFailure]
    _background_services: Mapping[str, BackgroundService]
    _account_providers: Mapping[str, "AccountProvider"]
    logger: logging.Logger

    _app_event_loop: asyncio.AbstractEventLoop
    _app_instance: Optional[threading.Thread] = None
    _app_lock: threading.Lock
    _lifecycle_lock: threading.Lock

    def __init__(
        self,
        settings: ClientSettings,
        logger: logging.Logger = logging.getLogger("app"),
        *,
        discovery_service: Optional["DiscoveryService"],
        account_providers: Mapping[str, "AccountProvider"],
        background_services: Optional[Mapping[str, BackgroundService]] = None,
        transport_factory: TransportFactory = default_transport_factory,
    ) -> None:
        super().__init__()

        integrations = settings.resolved_integrations()

        self._app_lock = threading.Lock()
        # Serialize start/stop ownership without holding ``_app_lock`` while
        # waiting for the runner. Startup config replay calls ``add()``, which
        # also needs ``_app_lock``.
        self._lifecycle_lock = threading.Lock()

        # Every loop-aware dependency below receives this loop/provider explicitly.
        # Do not install it as the thread's process-current loop: constructing an
        # app must not mutate an unrelated runner's asyncio state.
        self._app_event_loop = settings.event_loop_backend.new_event_loop()
        install_default_executor(
            self._app_event_loop,
            thread_name_prefix="sp-app-loop",
        )
        self.settings = settings
        self.client_list = ClientList()
        self.failed_clients = {}
        self.scheduler = Scheduler(
            self.client_list,
            self.settings,
            loop=self._app_event_loop,
            transport_factory=transport_factory,
        )
        self.integrations = {
            str(integration.id): integration for integration in integrations
        }
        self.config_managers = {
            str(integration.id): (
                integration.config_manager_t or settings.config_manager_t
            )(
                name=integration.storage_name(
                    settings.name, multiple=len(integrations) > 1
                ),
                config_t=integration.config_factory,
            )
            for integration in integrations
        }
        self.config_manager = next(iter(self.config_managers.values()))
        # The single owner of every blocking-call hop off the app loop.
        self.offload = Offload()
        # Coalesce the chatty config-change flush off the loop (registration in
        # add/remove still flushes directly -- see below).
        self.config_flushers = {
            key: ConfigFlusher(manager, self.offload, loop=self._app_event_loop)
            for key, manager in self.config_managers.items()
        }
        self.logger = logger
        self._background_services = (
            background_services if background_services is not None else {}
        )
        self._account_providers = account_providers
        self.simplyprint_api = SimplyPrintApi(settings.endpoints)
        self.host_telemetry = make_host_telemetry_reader()

        if settings.sentry_dsn is not None:
            Sentry.initialize_sentry(settings)

        if self.settings.camera_workers is not None:
            # The scheduler is the app's EventLoopProvider; INLINE/THREAD cameras
            # deliver frames onto its loop.
            self.camera_pool = CameraPool(
                event_loop_provider=self.scheduler,
                process_workers=self.settings.camera_workers,
            )
            self.camera_pool.protocols.extend(self.settings.camera_protocols or [])

        self.client_context = ClientContext(
            event_loop_provider=self.scheduler,
            camera_pool=self.camera_pool,
            offload=self.offload,
            discovery_service=discovery_service,
            simplyprint_api=self.simplyprint_api,
            host_telemetry=self.host_telemetry,
        )

    def context_for(self, integration_id: str) -> ClientContext:
        """Return this integration's explicitly scoped runtime dependencies."""
        return replace(
            self.client_context,
            background_service=self._background_services.get(integration_id),
            account_provider=self._account_providers.get(integration_id),
        )

    async def run(self):
        try:
            # On start, load all current configs. Each printer starts on its
            # own: a failure is logged and recorded by add() (or by the config
            # store, for an entry it could not read) and the rest carry on.
            for integration_id, manager in self.config_managers.items():
                self._record_unreadable_configs(integration_id, manager)
                for config in manager.get_all():
                    try:
                        self.add(config, integration_id=integration_id)
                    except Exception:
                        continue  # add() logged it once and recorded it

            await self.scheduler.block_until_stopped()
        except Exception as e:
            self.logger.exception("An error occurred in the main loop: %s", e)
            raise
        finally:
            await self._close_loop_resources()

    async def _close_loop_resources(self) -> None:
        # Drain pending config writes and transports before Runner closes their
        # owner loop. Every operation here is idempotent.
        for flusher in self.config_flushers.values():
            await flusher.aclose()
        await asyncio.gather(
            self.client_context.mqtt_pools.close(),
            self.client_context.websocket_pools.close(),
        )

    def _record_unreadable_configs(
        self, integration_id: str, manager: ConfigManager[PrinterConfig]
    ) -> None:
        # The store already logged each entry once on load; only record them.
        with self._app_lock:
            for entry in manager.unreadable:
                self.failed_clients[entry.label] = ClientStartFailure(
                    unique_id=entry.label,
                    integration_id=integration_id,
                    stage="config",
                    error=entry.error,
                )

    def failed_client_ids(self) -> List[str]:
        """Unique ids of the stored printers that are not running because they
        failed to load or build. Their configs are untouched on disk."""
        with self._app_lock:
            return sorted(self.failed_clients)

    def _close_unstarted_loop(self) -> None:
        """Close resources for an app whose owned loop never ran."""
        errors = []

        def close() -> None:
            try:
                with Runner(backend=self.settings.event_loop_backend) as runner:
                    runner.run(
                        self._close_loop_resources(),
                        loop_factory=lambda: self._app_event_loop,
                    )
            except BaseException as error:  # propagate the owner-thread failure
                errors.append(error)

        thread = threading.Thread(target=close, name="sp-app-close")
        thread.start()
        thread.join()
        if errors:
            raise errors[0]

    def run_blocking(self, debug=False, contexts: Optional[list] = None):
        contexts = contexts or []

        with Runner(debug, contexts, self.settings.event_loop_backend) as runner:
            runner.run(self.run(), loop_factory=lambda: self._app_event_loop)

    def is_running(self) -> bool:
        """Whether the detached runner thread is alive.

        The liveness seam supervisors poll to decide on a restart.
        """
        with self._app_lock:
            thread = self._app_instance
        return thread is not None and thread.is_alive()

    def run_detached(self, *args, **kwargs):
        with self._lifecycle_lock:
            with self._app_lock:
                if self.is_stopped() or self._app_event_loop.is_closed():
                    self.logger.warning("Cannot restart a stopped application.")
                    return
                if self._app_instance is not None:
                    self.logger.warning("Scheduler already running.")
                    return

                self._app_instance = threading.Thread(
                    target=self.run_blocking, args=args, kwargs=kwargs
                )
                self._app_instance.start()

                # Register atexit handler to prevent spamming of "Cannot schedule new futures after shutdown" errors.
                atexit.register(self.stop)

    def generate_connectivity_report(self, **kwargs):
        """Generate a connectivity report for the SimplyPrint backends.

        The app-level hook the generic CLI calls, so the leaf debug module
        never has to look up the SimplyPrint endpoints itself.
        """
        from simplyprint_ws_client.core.api.url_builder import (
            default_connectivity_report,
        )

        return default_connectivity_report(**kwargs)

    def _get_integration(self, integration_id: Optional[str] = None) -> IntegrationSpec:
        if integration_id is not None:
            return self.integrations[integration_id]
        if len(self.integrations) == 1:
            return next(iter(self.integrations.values()))
        raise ValueError("Integration id is required when multiple integrations exist.")

    def get_config_manager(
        self, integration_id: Optional[str] = None
    ) -> ConfigManager[PrinterConfig]:
        integration = self._get_integration(integration_id)
        return self.config_managers[str(integration.id)]

    def add(
        self, config: PrinterConfig, integration_id: Optional[str] = None
    ) -> Client:
        """Build, wire and schedule the client for ``config``.

        The config is persisted first and stays persisted even if building the
        client fails: a broken driver must not cost the user the printer's
        registration. On failure the error is logged once, recorded in
        :attr:`failed_clients`, and re-raised to this caller; nothing has been
        registered, so calling ``add`` again (after a fix) retries cleanly.
        """
        integration = self._get_integration(integration_id)
        resolved_id = str(integration.id)
        config_manager = self.config_managers[resolved_id]

        with self._app_lock:
            if config.unique_id in self.client_list:
                return self.client_list[config.unique_id]

            try:
                config_manager.persist(config)
                config_manager.flush(config)
                client = integration.client_factory(
                    config,
                    context=self.context_for(resolved_id),
                )
            except Exception as e:
                self.failed_clients[str(config.unique_id)] = ClientStartFailure(
                    unique_id=str(config.unique_id),
                    integration_id=resolved_id,
                    stage="build",
                    error="%s: %s" % (type(e).__name__, e),
                )
                self.logger.error(
                    "Printer %s (%s) failed to start; its config is kept and "
                    "the other printers are unaffected",
                    config.unique_id,
                    resolved_id,
                    exc_info=e,
                )
                raise

            self.failed_clients.pop(str(config.unique_id), None)

            # From here on nothing can fail per printer: listener registration
            # is plain bookkeeping, and submit() is the single commit point
            # that makes the client visible to the scheduler.

            # Coalesce + offload the chatty change flush; trigger() is a cheap,
            # thread-safe "mark dirty" (covers emits from device threads).
            flusher = self.config_flushers[resolved_id]
            client.event_bus.on(
                ClientConfigChangedEvent,
                lambda *args, **kwargs: flusher.trigger(),
            )

            client.event_bus.on(ClientStateChangeEvent, self.scheduler.signal)

            self.scheduler.submit(client)

            # The client factory may start background connections (e.g. MQTT) during __init__
            # that set active=True before the ClientStateChangeEvent listener is registered above.
            # If that happened, the signal was lost. Re-signal the scheduler to pick it up.
            if client.active:
                self.scheduler.signal()

        return client

    def remove(
        self, config: PrinterConfig, integration_id: Optional[str] = None
    ) -> None:
        config_manager = self.get_config_manager(integration_id)

        with self._app_lock:
            client = self.client_list.get(config.unique_id)

            if client is None:
                # A printer that failed to start has a stored config but no
                # client; removing it must still delete what the user asked
                # to delete. ``contains`` is identity-based, so this only
                # touches this exact stored config.
                self.failed_clients.pop(str(config.unique_id), None)
                if config_manager.contains(config):
                    config_manager.remove(config)
                    config_manager.flush(config)
                return

            # Do not persist further changes to the config.
            client.event_bus.clear(ClientConfigChangedEvent, ClientStateChangeEvent)

            # TODO: TECHNICALLY this should happen after the scheduler calls _delete.
            config_manager.remove(config)
            config_manager.flush(config)

            self.scheduler.remove(client)

    def stop(self):
        super().stop()

        with self._lifecycle_lock:
            # Only snapshot runner ownership under the app data lock. The
            # runner can be replaying stored configs through ``add()`` while
            # stop waits for it, so joining under this lock deadlocks.
            with self._app_lock:
                thread = self._app_instance

            if self.scheduler.event_loop_is_running():
                self.scheduler.event_loop.call_soon_threadsafe(self.scheduler.stop)
            else:
                self.scheduler.stop()

            if thread is not None:
                thread.join()
                with self._app_lock:
                    if self._app_instance is thread:
                        self._app_instance = None
            elif (
                not self._app_event_loop.is_running()
                and not self._app_event_loop.is_closed()
            ):
                self._close_unstarted_loop()

            # Loop is dead now; if run()'s finally never executed (crash / no run),
            # write any still-pending config change synchronously so it is not lost.
            for flusher in self.config_flushers.values():
                flusher.flush_now_if_dirty()

            if self.camera_pool is not None:
                self.camera_pool.stop()

            # Last: after the loop is dead and no camera reconcile can submit,
            # join the blocking-work lanes so teardown leaks no executor threads.
            self.offload.shutdown(wait=True)

            self.logger.info("Stopped.")
