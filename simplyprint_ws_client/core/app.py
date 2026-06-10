__all__ = ["ClientApp"]

import asyncio
import atexit
import logging
import threading
from typing import Dict, Optional, cast

from simplyprint_ws_client.cloud.client import (
    Client,
    ClientConfigChangedEvent,
    ClientStateChangeEvent,
)
from simplyprint_ws_client.cloud.config import PrinterConfig
from simplyprint_ws_client.core.config import ConfigManager
from simplyprint_ws_client.core.connection_manager import ClientList
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.core.settings import ClientSettings, ClientSpec
from simplyprint_ws_client.common.asyncio.event_loop_runner import Runner
from simplyprint_ws_client.device.camera.pool import CameraPool
from simplyprint_ws_client.cloud.api.sentry import Sentry
from simplyprint_ws_client.cloud.api.url_builder import SimplyPrintURL
from simplyprint_ws_client.common.utils.stoppable import SyncStoppable


class ClientApp(SyncStoppable):
    settings: ClientSettings
    client_list: ClientList
    scheduler: Scheduler
    config_manager: ConfigManager[PrinterConfig]
    config_managers: Dict[str, ConfigManager[PrinterConfig]]
    client_specs: Dict[str, ClientSpec]
    camera_pool: Optional[CameraPool] = None
    logger: logging.Logger

    _app_event_loop: asyncio.AbstractEventLoop
    _app_instance: Optional[threading.Thread] = None
    _app_lock: threading.Lock

    def __init__(
        self,
        settings: ClientSettings,
        logger: logging.Logger = logging.getLogger("app"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        specs = settings.resolved_client_specs()

        self._app_lock = threading.Lock()

        # For older python versions we want to set the event loop that loop mixins use
        # before we can initialize objects that require it.
        self._app_event_loop = settings.event_loop_backend.new_event_loop()
        asyncio.set_event_loop(self._app_event_loop)

        self.settings = settings
        self.client_list = ClientList()
        self.scheduler = Scheduler(
            self.client_list, self.settings, loop=self._app_event_loop
        )
        self.client_specs = {spec.key: spec for spec in specs}
        self.config_managers = {
            spec.key: (spec.config_manager_t or settings.config_manager_t)(
                name=spec.storage_name(settings.name, multiple=len(specs) > 1),
                config_t=spec.config_factory,
            )
            for spec in specs
        }
        self.config_manager = next(iter(self.config_managers.values()))
        self.logger = logger

        if settings.backend is not None:
            SimplyPrintURL.set_backend(settings.backend)

        if settings.sentry_dsn is not None:
            Sentry.initialize_sentry(settings)

        if self.settings.camera_workers is not None:
            # The scheduler is the app's EventLoopProvider; INLINE/THREAD cameras
            # deliver frames onto its loop.
            self.camera_pool = CameraPool(
                pool_size=self.settings.camera_workers,
                event_loop_provider=self.scheduler,
            )
            self.camera_pool.protocols.extend(self.settings.camera_protocols or [])

    async def run(self):
        try:
            # On start, load all current configs.
            for key, manager in self.config_managers.items():
                for config in manager.get_all():
                    self.add(config, client_key=key)

            await self.scheduler.block_until_stopped()
        except Exception as e:
            self.logger.exception("An error occurred in the main loop: %s", e)
            raise

    def run_blocking(self, debug=False, contexts: Optional[list] = None):
        contexts = contexts or []

        with Runner(debug, contexts, self.settings.event_loop_backend) as runner:
            runner.run(self.run(), loop_factory=lambda: self._app_event_loop)

    def run_detached(self, *args, **kwargs):
        with self._app_lock:
            if self._app_instance is not None:
                self.logger.warning("Scheduler already running.")
                return

            self._app_instance = threading.Thread(
                target=self.run_blocking, args=args, kwargs=kwargs
            )
            self._app_instance.start()

            # Register atexit handler to prevent spamming of "Cannot schedule new futures after shutdown" errors.
            atexit.register(self.stop)

    def _get_client_spec(
        self, config: Optional[PrinterConfig] = None, client_key: Optional[str] = None
    ) -> ClientSpec:
        if client_key is not None:
            return self.client_specs[client_key]

        if len(self.client_specs) == 1:
            return next(iter(self.client_specs.values()))

        if config is None:
            raise ValueError("Client key is required when multiple specs exist.")

        matches = []
        config_mro = config.__class__.mro()

        for spec in self.client_specs.values():
            if isinstance(spec.config_factory, type) and isinstance(
                config, spec.config_factory
            ):
                matches.append((config_mro.index(spec.config_factory), spec))

        if len(matches) == 1:
            return matches[0][1]

        if len(matches) > 1:
            best_distance = min(distance for distance, _ in matches)
            best_matches = [
                spec for distance, spec in matches if distance == best_distance
            ]

            if len(best_matches) == 1:
                return best_matches[0]

            raise ValueError(
                f"Config {config!r} matches multiple client specs. Pass client_key."
            )

        raise ValueError(f"No client spec found for config {config!r}.")

    def get_config_manager(
        self, client_key: Optional[str] = None, config: Optional[PrinterConfig] = None
    ) -> ConfigManager[PrinterConfig]:
        spec = self._get_client_spec(config, client_key)
        return self.config_managers[spec.key]

    def add(self, config: PrinterConfig, client_key: Optional[str] = None) -> Client:
        spec = self._get_client_spec(config, client_key)
        config_manager = self.config_managers[spec.key]

        with self._app_lock:
            if config.unique_id in self.client_list:
                return self.client_list[config.unique_id]

            config_manager.persist(config)
            config_manager.flush(config)

            client = spec.client_factory(
                config, event_loop_provider=self.scheduler, camera_pool=self.camera_pool
            )

            client.event_bus.on(
                ClientConfigChangedEvent,
                lambda *args, **kwargs: config_manager.flush(
                    cast(PrinterConfig, client.config)
                ),
            )

            client.event_bus.on(ClientStateChangeEvent, self.scheduler.signal)

            self.scheduler.submit(client)

            # The client factory may start background connections (e.g. MQTT) during __init__
            # that set active=True before the ClientStateChangeEvent listener is registered above.
            # If that happened, the signal was lost. Re-signal the scheduler to pick it up.
            if client.active:
                self.scheduler.signal()

        return client

    def remove(self, config: PrinterConfig, client_key: Optional[str] = None) -> None:
        config_manager = self.get_config_manager(client_key, config)

        with self._app_lock:
            client = self.client_list.get(config.unique_id)

            if client is None:
                return

            # Do not persist further changes to the config.
            client.event_bus.clear(ClientConfigChangedEvent, ClientStateChangeEvent)

            # TODO: TECHNICALLY this should happen after the scheduler calls _delete.
            config_manager.remove(config)
            config_manager.flush(config)

            self.scheduler.remove(client)

    def stop(self):
        super().stop()

        with self._app_lock:
            if self.scheduler.event_loop_is_running():
                self.scheduler.event_loop.call_soon_threadsafe(self.scheduler.stop)
            else:
                self.scheduler.stop()

            if self._app_instance is not None:
                self._app_instance.join()
                self._app_instance = None

            if self.camera_pool is not None:
                self.camera_pool.stop()

            self.logger.info("Stopped.")
