__all__ = ["ClientApp"]

import asyncio
import atexit
import functools
import logging
import threading
from typing import Optional, cast

from .client import Client, ClientConfigChangedEvent, ClientStateChangeEvent
from .config import ConfigManager, PrinterConfig
from .connection_manager import ClientList
from .scheduler import Scheduler
from .settings import ClientSettings
from ..shared.asyncio.event_loop_runner import Runner
from ..shared.camera.pool import CameraPool
from ..shared.debug import traceability
from ..shared.sp.sentry import Sentry
from ..shared.sp.url_builder import SimplyPrintURL
from ..shared.utils.stoppable import SyncStoppable


class ClientApp(SyncStoppable):
    settings: ClientSettings
    client_list: ClientList
    scheduler: Scheduler
    config_manager: ConfigManager[PrinterConfig]
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

        if settings.client_factory is None or settings.config_factory is None:
            raise ValueError("Both client and config factory must be set in settings.")

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
        self.config_manager = settings.config_manager_t(
            name=settings.name,
            config_t=settings.config_factory,
        )
        self.logger = logger

        if settings.backend is not None:
            SimplyPrintURL.set_backend(settings.backend)

        if settings.sentry_dsn is not None:
            Sentry.initialize_sentry(settings)

        if self.settings.camera_workers is not None:
            self.camera_pool = CameraPool(pool_size=self.settings.camera_workers)
            self.camera_pool.protocols.extend(self.settings.camera_protocols or [])

    async def run(self):
        try:
            # On start, load all current configs.
            for config in self.config_manager.get_all():
                self.add(config)

            await self.scheduler.block_until_stopped()
        except Exception as e:
            self.logger.exception("An error occurred in the main loop: %s", e)
            raise

    def run_blocking(self, debug=False, contexts: Optional[list] = None):
        contexts = contexts or []
        contexts.append(functools.partial(traceability.enable_traceable, debug))

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

    def add(self, config: PrinterConfig) -> Client:
        with self._app_lock:
            if config.unique_id in self.client_list:
                return self.client_list[config.unique_id]

            self.config_manager.persist(config)
            self.config_manager.flush(config)

            client = self.settings.client_factory(
                config, event_loop_provider=self.scheduler, camera_pool=self.camera_pool
            )

            client.event_bus.on(
                ClientConfigChangedEvent,
                lambda *args, **kwargs: self.config_manager.flush(
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

    def remove(self, config: PrinterConfig) -> None:
        with self._app_lock:
            client = self.client_list.get(config.unique_id)

            if client is None:
                return

            # Do not persist further changes to the config.
            client.event_bus.clear(ClientConfigChangedEvent, ClientStateChangeEvent)

            # TODO: TECHNICALLY this should happen after the scheduler calls _delete.
            self.config_manager.remove(config)
            self.config_manager.flush(config)

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
