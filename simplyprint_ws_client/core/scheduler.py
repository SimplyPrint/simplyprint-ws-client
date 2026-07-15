__all__ = ["Scheduler"]

import asyncio
import logging
import sys
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from simplyprint_ws_client.core.client import Client, ClientState
from simplyprint_ws_client.core.manager import (
    ClientConnectionManager,
    ClientList,
)
from simplyprint_ws_client.core.protocol.connection import (
    TransportFactory,
    default_transport_factory,
)
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.common.asyncio.async_task_scope import AsyncTaskScope
from simplyprint_ws_client.common.asyncio.continuous_task import ContinuousTask
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable

if sys.version_info >= (3, 11):
    from asyncio import timeout as tick_timeout
else:
    from async_timeout import timeout as tick_timeout

#: Per-client budget for one ``tick``; a slow client is cut off so it cannot
#: stall the shared scheduling loop.
TICK_TIMEOUT_SECONDS = 5


class Scheduler(AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop]):
    """Client scheduler.

    Attributes:
        settings:
        client_list: ClientList
        manager: ClientConnectionManager
        logger: logging.Logger
        _wake_event: asyncio.Event
        _tasks: Dict[str, ContinuousTask]
        _to_delete: Set[str]
        _schedule_task: ContinuousTask
    """

    settings: ClientSettings
    client_list: ClientList
    manager: ClientConnectionManager
    logger: logging.Logger
    _wake_event: asyncio.Event
    _tasks: Dict[str, ContinuousTask]
    _last_ticked: Dict[str, datetime]
    _tick_timeout_reported: Set[str]
    _tick_rate_delta: timedelta
    _to_delete: Set[str]
    _schedule_task: ContinuousTask
    _signal_lock: threading.Lock
    _signal_scheduled: bool

    def __init__(
        self,
        client_list: ClientList,
        settings: ClientSettings,
        logger: logging.Logger = logging.getLogger("Scheduler"),
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        transport_factory: TransportFactory = default_transport_factory,
    ) -> None:
        AsyncStoppable.__init__(self)
        EventLoopProvider.__init__(self, loop=loop)

        self.settings = settings
        self.client_list = client_list
        self.manager = ClientConnectionManager(
            self.settings.mode,
            self.client_list,
            self.settings.endpoints.websocket_url,
            max_clients_per_connection=self.settings.max_clients_per_connection,
            provider=self,
            transport_factory=transport_factory,
        )
        self.logger = logger
        self._wake_event = asyncio.Event()
        self._tasks = {}
        self._last_ticked = {}
        self._tick_timeout_reported = set()
        self._tick_rate_delta = timedelta(seconds=self.settings.tick_rate)
        self._to_delete = set()
        self._schedule_task = ContinuousTask(self._schedule_loop, provider=self)
        self._signal_lock = threading.Lock()
        self._signal_scheduled = False
        #: Keeps in-flight client teardown tasks alive (asyncio holds tasks
        #: weakly) and reports their failures.
        self._teardown_tasks = set()

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
        self._last_ticked.pop(client.unique_id, None)
        self._tick_timeout_reported.discard(client.unique_id)
        self._to_delete.discard(client.unique_id)
        self.signal()

    def signal(self):
        """Latch one scheduling wake-up onto the app loop.

        An ``asyncio.Condition`` notification is edge-triggered and disappears
        when no coroutine is waiting. Client submission can land in the narrow
        gap between a scheduling pass and waiter registration, so the old
        waiter-count fast path delayed that client until the periodic tick.
        ``asyncio.Event`` retains the wake-up until the loop consumes it.
        """
        with self._signal_lock:
            if self._signal_scheduled:
                return
            self._signal_scheduled = True

        try:
            # Queuing on a not-yet-running loop is intentional: app.add() may
            # race run_detached() startup, and this callback then becomes the
            # retained first wake-up as soon as the loop begins.
            self.event_loop.call_soon_threadsafe(self._deliver_signal)
        except RuntimeError:
            with self._signal_lock:
                self._signal_scheduled = False

    def _deliver_signal(self) -> None:
        with self._signal_lock:
            self._signal_scheduled = False
        self._wake_event.set()

    async def _wait_for_signal(self) -> None:
        await self._wake_event.wait()
        self._wake_event.clear()

    def _should_schedule_client(self, client: Client, when: datetime):
        # Always schedule clients that still need their once-per-lifetime init.
        if not client.initialized:
            return True

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
        """Schedule single client.

        The client's own lifecycle (``init`` once at scheduler entry, ``tick``
        at the tick rate) runs for EVERY scheduled client, whether or not it is
        allocated to SimplyPrint -- the device edges produced by that lifecycle
        are what drive ``active``, so they cannot be gated on it. Allocation to
        a SimplyPrint connection is a separate state machine keyed on
        ``client.active``: an active client is allocated and added, an inactive
        one is removed, deallocated and parked via ``halt``.
        """
        try:
            if not client.initialized:
                # init is once per lifetime, never retried -- recovery paths
                # belong in tick (e.g. the device-driver ensure_started sweep).
                client.initialized = True
                await client.init()

            # Tick client.
            last_ticked = self._last_ticked.get(client.unique_id, datetime.min)
            now = datetime.now()
            delta_tick = now - last_ticked

            if delta_tick >= self._tick_rate_delta:
                self._last_ticked[client.unique_id] = now

                try:
                    async with tick_timeout(TICK_TIMEOUT_SECONDS):
                        await client.tick(delta_tick)
                    self._tick_timeout_reported.discard(client.unique_id)
                except asyncio.TimeoutError as e:
                    # A stalled device or ping can time out on every scheduler
                    # slice. Report one compact warning per failure streak;
                    # a successful tick rearms it. Repeated tracebacks for the
                    # same transient outage obscure the actual reconnect edge.
                    if client.unique_id not in self._tick_timeout_reported:
                        client.logger.warning("Client tick timed out: %s", e)
                        self._tick_timeout_reported.add(client.unique_id)
                except Exception as e:
                    # A slow or failing tick must not stall the allocation
                    # state machine below: tick runs first (device side), but
                    # its failures are its own -- it gets retried next pass
                    # either way, while allocate/ensure_added/ensure_removed
                    # still progress this pass.
                    client.logger.error("Error while ticking client", exc_info=e)

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

            # Progress inner client state until we reach CONNECTED state.
            # e.i. in multi printer mode until we receive the connected message.
            if not await client.ensure_added(
                self.settings.mode, self.settings.allow_setup
            ):
                return

            if not client.has_changes:
                return

            msgs = client.consume()

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
            task = self.event_loop.create_task(client.teardown())
            self._teardown_tasks.add(task)
            task.add_done_callback(self._on_teardown_task_done)

    def _on_teardown_task_done(self, task) -> None:
        self._teardown_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.logger.warning(
                "client teardown failed",
                exc_info=(type(error), error, error.__traceback__),
            )

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
                        task_scope.create_task(self._wait_for_signal()),
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
