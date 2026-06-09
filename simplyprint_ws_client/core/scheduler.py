__all__ = ["Scheduler"]

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Set

from .client import Client, ClientState
from .connection_manager import ClientConnectionManager, ClientList
from .settings import ClientSettings
from ..shared.asyncio.async_task_scope import AsyncTaskScope
from ..shared.asyncio.continuous_task import ContinuousTask
from ..shared.asyncio.event_loop_provider import EventLoopProvider
from ..shared.asyncio.utils import cond_notify_all, cond_wait
from ..shared.utils.stoppable import AsyncStoppable


class Scheduler(AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop]):
    """Client scheduler.

    Attributes:
        settings:
        client_list: ClientList
        manager: ClientConnectionManager
        logger: logging.Logger
        _cond: asyncio.Condition
        _tasks: Dict[str, ContinuousTask]
        _to_delete: Set[str]
        _schedule_task: ContinuousTask
    """

    settings: ClientSettings
    client_list: ClientList
    manager: ClientConnectionManager
    logger: logging.Logger
    _cond: asyncio.Condition
    _tasks: Dict[str, ContinuousTask]
    _last_ticked: Dict[str, datetime]
    _tick_rate_delta: timedelta
    _to_delete: Set[str]
    _schedule_task: ContinuousTask
    _pending_signals: Set[asyncio.Future]
    _signal_lock: threading.Lock

    def __init__(
        self,
        client_list: ClientList,
        settings: ClientSettings,
        logger: logging.Logger = logging.getLogger("Scheduler"),
        **kwargs,
    ):
        AsyncStoppable.__init__(self, **kwargs)
        EventLoopProvider.__init__(self, **kwargs)

        self.settings = settings
        self.client_list = client_list
        self.manager = ClientConnectionManager(
            self.settings.mode,
            self.client_list,
            max_clients_per_connection=self.settings.max_clients_per_connection,
            provider=self,
        )
        self.logger = logger
        self._cond = asyncio.Condition()
        self._tasks = {}
        self._last_ticked = {}
        self._tick_rate_delta = timedelta(seconds=self.settings.tick_rate)
        self._to_delete = set()
        self._schedule_task = ContinuousTask(self._schedule_loop, provider=self)
        self._pending_signals = set()
        self._signal_lock = threading.Lock()

    def submit(self, client: Client):
        if client.unique_id in self.client_list:
            return

        if client.unique_id in self._to_delete:
            self.logger.warning(
                "client %s is being submitted is also pending deletion.",
                client.unique_id,
            )
            self._to_delete.discard(client.unique_id)

        self._tasks.pop(client.unique_id, None)
        self.client_list.add(client)
        self.signal()

    def remove(self, client: Client):
        if client.unique_id not in self.client_list:
            return

        client.active = False
        self._to_delete.add(client.unique_id)
        self.signal()

    def _delete(self, client: Client):
        self.client_list.remove(client)
        self._tasks.pop(client.unique_id, None)
        self._to_delete.discard(client.unique_id)
        self.signal()

    def signal(self):
        with self._signal_lock:
            # Optimization: No one to wake.
            if len(self._cond._waiters) == 0:
                return

            # Optimization: No need to signal if there are pending signals.
            if len(self._pending_signals) > 0:
                return

            if not self.event_loop_is_running():
                return

            fut = asyncio.run_coroutine_threadsafe(
                cond_notify_all(self._cond), self.event_loop
            )
            fut.add_done_callback(self._pending_signals.discard)
            self._pending_signals.add(fut)

    def _should_schedule_client(self, client: Client, when: datetime):
        # Always schedule clients that have changes.
        if client.has_changes:
            return True

        # Always schedule clients that are pending a tick.
        if (
            when - self._last_ticked.get(client.unique_id, datetime.min)
            >= self._tick_rate_delta
        ):
            return True

        # Schedule if the client needs to change its connection state.
        is_active = client.active
        return (is_active and not client.is_added()) or (
            not is_active and not client.is_removed()
        )

    async def _schedule_client(self, client: Client):
        """Schedule single client."""
        try:
            was_allocated = self.manager.is_allocated(client)

            if not client.active:
                if not was_allocated:
                    return

                # Remove the connection from the multi printer.
                if not await client.ensure_removed(self.settings.mode):
                    return

                # Then we can deallocate the client from the connection.
                await self.manager.deallocate(client)
                await client.halt()
                return

            if not was_allocated:
                await self.manager.allocate(client)
                await client.init()

            # Progress inner client state until we reach CONNECTED state.
            # e.i. in multi printer mode until we receive the connected message.
            if not await client.ensure_added(
                self.settings.mode, self.settings.allow_setup
            ):
                return

            # Tick client.
            last_ticked = self._last_ticked.get(client.unique_id, datetime.min)
            now = datetime.now()
            delta_tick = now - last_ticked

            if delta_tick >= self._tick_rate_delta:
                self._last_ticked[client.unique_id] = now

                # TODO: Manage timeouts.
                async with asyncio.timeout(5):
                    await client.tick(delta_tick)

            if not client.has_changes:
                return

            msgs, v = client.consume()

            for msg in msgs:
                await client.send(msg, skip_dispatch=True)

        except Exception as e:
            client.logger.error("Error while scheduling client", exc_info=e)

    def _process_clients(self):
        """Schedule all clients for processing."""
        now = datetime.now()

        for unique_id, client in list(self.client_list.items()):
            # Optimization: Skip clients that do not need to be scheduled.
            if not self._should_schedule_client(client, now):
                continue

            if unique_id not in self._tasks:
                self._tasks[unique_id] = ContinuousTask(
                    self._schedule_client, provider=self
                )

            task = self._tasks[unique_id]

            if task.done():
                task.discard()

            task.schedule(client)

    def _process_to_delete(self):
        """Process to_delete set."""
        if not self._to_delete:
            return

        # First ensure the client is properly removed
        # from its connection, then remove it from the
        # scheduler.
        for client_id in list(self._to_delete):
            client = self.client_list.get(client_id)

            if not client:
                self._to_delete.discard(client_id)
                continue

            if client.active or client.state > ClientState.NOT_CONNECTED:
                continue

            self._delete(client)
            # SAFETY: The client will never be considered for this again
            # so this spawns a single task per added client.
            self.event_loop.create_task(client.teardown())

    async def _teardown(self):
        """Teardown all clients, then await all connections to stop."""
        self.manager.stop()

        for task in self._tasks.values():
            task.discard()

        await asyncio.gather(
            *(client.teardown() for client in self.client_list.values())
        )
        await asyncio.gather(
            *(
                task
                for connection in self.manager.connections
                if (task := connection.loop_task) is not None
            ),
            return_exceptions=True,  # stopped engines wind down via CancelledError
        )

    async def _schedule_loop(self):
        if self._schedule_task.task != asyncio.current_task():
            raise RuntimeError("Connection task already running.")

        self.logger.info("Scheduler started")

        last_scheduled = datetime.now()
        task_scope = AsyncTaskScope(provider=self)

        while not self.is_stopped():
            try:
                now = datetime.now()
                delta = now - last_scheduled
                last_scheduled = now

                if delta > timedelta(seconds=self.settings.tick_rate) * 2:
                    self.logger.warning(f"Scheduler is running behind, delta={delta}")

                self._process_clients()
                self._process_to_delete()
            except Exception as e:
                self.logger.error("Critical error in scheduler", exc_info=e)
            finally:
                # Cancel and GC non-finalized tasks.
                with task_scope:
                    # Wait until either a change is made to the state or a timeout occurs.
                    conditions = [
                        task_scope.create_task(self.wait(self.settings.tick_rate)),
                        task_scope.create_task(cond_wait(self._cond)),
                    ]

                    await asyncio.wait(conditions, return_when=asyncio.FIRST_COMPLETED)

        await self._teardown()
        self.logger.info("Scheduler stopped")

    @property
    def running(self):
        return self._schedule_task.task is not None and not self._schedule_task.done()

    async def block_until_stopped(self):
        while not self.is_stopped():
            if self._schedule_task.done():
                self._schedule_task.discard()

            await self._schedule_task.schedule()

    def start(self):
        if self._schedule_task.done():
            self._schedule_task.discard()

        self._schedule_task.schedule()
