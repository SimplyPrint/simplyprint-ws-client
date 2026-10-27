__all__ = ["Scheduler"]

import asyncio
import contextlib
import logging
import sys
import threading
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, Optional, Set

from simplyprint_ws_client._compat import AsyncEvent
from simplyprint_ws_client.core.client import Client, ClientState
from simplyprint_ws_client.core.manager import (
    ClientConnectionManager,
    ClientList,
)
from simplyprint_ws_client.core.protocol.connection import (
    TransportFactory,
    default_transport_factory,
)
from simplyprint_ws_client.core.protocol.models import WriteOutcome
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

#: Outcomes that mean the link failed under a write: the send stops there.
#: An edge that may have left counts as sent, but the link is just as dead.
_LINK_FAILED = (WriteOutcome.NOT_SENT, WriteOutcome.MAYBE_SENT)


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
    _dispatch_error_reported: Set[str]
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
        self._wake_event = AsyncEvent()
        self._tasks = {}
        self._last_ticked = {}
        self._tick_timeout_reported = set()
        #: Clients whose scheduling dispatch raised; logged once per streak.
        self._dispatch_error_reported = set()
        self._tick_rate_delta = timedelta(seconds=self.settings.tick_rate)
        self._to_delete = set()
        self._schedule_task = ContinuousTask(self._schedule_loop, provider=self)
        self._signal_lock = threading.Lock()
        self._signal_scheduled = False
        #: Keeps in-flight client teardown tasks alive (asyncio holds tasks
        #: weakly) and reports their failures.
        self._teardown_tasks = set()
        #: The in-flight removal teardown per client id, for until_removed.
        self._teardowns: Dict[str, asyncio.Task] = {}
        #: Woken whenever a removal teardown finishes (loop-only).
        self._removal_waiters: Set[asyncio.Future] = set()
        #: Set once the stop teardown has run for every client.
        self._torn_down = False

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
        client.hold_removal(0)  # a printer being deleted leaves at once
        self._to_delete.add(client.unique_id)
        self.signal()

    @contextlib.contextmanager
    def replaying(self) -> Iterator[None]:
        """Within: stored printers are being submitted in bulk. None is placed
        on a hub until the last replay ends, so the hub count is taken from
        the whole list once (:meth:`ClientConnectionManager.replaying`); the
        scheduler is woken then to place them."""
        try:
            with self.manager.replaying():
                yield
        finally:
            self.signal()

    def forget(self, client: Client) -> None:
        """Drop a client that was submitted but never scheduled.

        Only for while the loop is not running (nothing has run ``init`` yet, so
        there is nothing to tear down); a running scheduler removes clients
        through :meth:`remove`.
        """
        if client.unique_id in self.client_list:
            self._delete(client)

    async def until_removed(self, unique_ids: Iterable[str]) -> None:
        """On the loop: wait until each client is deleted and torn down.

        Returns at once for ids that are not scheduled, and once the stop
        teardown has run, which tears every client down.
        """
        ids = set(unique_ids)
        loop = asyncio.get_running_loop()
        while not self._torn_down:
            if not any(
                uid in self.client_list or uid in self._teardowns for uid in ids
            ):
                return
            waiter = loop.create_future()
            self._removal_waiters.add(waiter)
            try:
                # The tick-rate timeout is only a backstop for a missed wake.
                await asyncio.wait_for(
                    asyncio.shield(waiter), timeout=self.settings.tick_rate
                )
            except asyncio.TimeoutError:
                pass
            finally:
                self._removal_waiters.discard(waiter)

    def _wake_removal_waiters(self) -> None:
        waiters, self._removal_waiters = self._removal_waiters, set()
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    def _teardown_finished(self, client_id: str, task: asyncio.Task) -> None:
        self._teardown_tasks.discard(task)
        if self._teardowns.get(client_id) is task:
            del self._teardowns[client_id]
        self._wake_removal_waiters()

    def _delete(self, client: Client):
        self.client_list.remove(client)
        self._tasks.pop(client.unique_id, None)
        self._last_ticked.pop(client.unique_id, None)
        self._tick_timeout_reported.discard(client.unique_id)
        self._dispatch_error_reported.discard(client.unique_id)
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
            not is_active and not client.is_removed() and not client.removal_held()
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
                except (asyncio.TimeoutError, TimeoutError) as e:
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

                if client.removal_held():
                    # Its device dropped moments ago: it stays added (and on
                    # its hub) until the hold ends, which wakes this pass.
                    return

                # Remove the connection from the multi printer.
                if not await client.ensure_removed(self.settings.mode):
                    return

                # Then we can deallocate the client from the connection.
                await self.manager.deallocate(client)
                await client.halt()
                return

            if was_allocated and self.manager.is_misplaced(client):
                # The hub count changed and this printer's slot moved. Re-add
                # it on its new hub; the server's newest-wins supersession
                # retires the old child. The loss arms a full-state resend.
                client.logger.info(
                    "Moving to hub %d", self.manager.hub_index(client.unique_id)
                )
                await self.manager.deallocate(client)
                was_allocated = False

            if not was_allocated:
                if self.manager.is_replaying:
                    # Placed now, it would be hashed on a partial list and
                    # moved again once the rest arrived; the pass after the
                    # replay places it (see Scheduler.replaying).
                    return
                await self.manager.allocate(client)

            # Progress inner client state until we reach CONNECTED state.
            # e.i. in multi printer mode until we receive the connected message.
            if not await client.ensure_added(
                self.settings.mode, self.settings.allow_setup
            ):
                return

            # Deferred changes (rate limited, or a frame that never left) have
            # no new change to announce them; this pass is their retry.
            if not (client.has_changes or client.has_deferred):
                return

            # Each message's change markers are reset only once its frame is
            # written (Client.write -> settle), so a dead socket loses nothing.
            with client.settling_on_write():
                msgs = client.consume()

            for i, msg in enumerate(msgs):
                await client.write(msg)
                if msg.write_outcome in _LINK_FAILED:
                    # The link failed under this frame. Writing on could put
                    # the rest out ahead of it on a link that came back
                    # meanwhile (a status before a kept `finished` starts the
                    # server's job-failure clock); they go with the next send,
                    # in its order (edges first, job info before status).
                    client.unwritten(msgs[i + 1 :])
                    break

        except Exception as e:
            client.logger.error("Error while scheduling client", exc_info=e)

    def _process_clients(self):
        """Schedule all clients for processing."""
        now = datetime.now()

        for unique_id, client in list(self.client_list.items()):
            # Per client: _should_schedule_client reads client properties a
            # subclass may override. If one raised here unguarded, the loop
            # would abort and every client after it would never be scheduled.
            try:
                self._dispatch_client(unique_id, client, now)
                self._dispatch_error_reported.discard(unique_id)
            except Exception as e:
                if unique_id not in self._dispatch_error_reported:
                    self._dispatch_error_reported.add(unique_id)
                    client.logger.error(
                        "Error while dispatching client %s", unique_id, exc_info=e
                    )

    def _dispatch_client(self, unique_id: str, client: Client, now: datetime):
        # Optimization: Skip clients that do not need to be scheduled.
        if not self._should_schedule_client(client, now):
            return

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

            try:
                if client.active or client.state > ClientState.NOT_CONNECTED:
                    continue
            except Exception as e:
                # A client whose state cannot be read is deleted anyway: it
                # is leaving, and must not block the rest of this pass.
                client.logger.error(
                    "Error reading state of client %s pending deletion",
                    client_id,
                    exc_info=e,
                )

            self._delete(client)
            # SAFETY: The client will never be considered for this again
            # so this spawns a single task per added client.
            task = self.event_loop.create_task(self._teardown_client(client))
            self._teardown_tasks.add(task)
            self._teardowns[client_id] = task
            task.add_done_callback(
                lambda done, cid=client_id: self._teardown_finished(cid, done)
            )

    @staticmethod
    async def _teardown_client(client: Client) -> None:
        """Run one client's teardown, containing its failure.

        A raising teardown is that printer's problem: it is logged against the
        printer, and never aborts the shutdown of the others.
        """
        try:
            await client.teardown()
        except Exception as e:
            client.logger.warning(
                "Teardown of client %s failed", client.unique_id, exc_info=e
            )

    async def _teardown(self):
        """Teardown all clients, then await all connections to stop."""
        self.manager.stop()

        for task in self._tasks.values():
            task.discard()

        # Each teardown contains its own failure (see _teardown_client), so
        # one raising client can neither abort this gather nor leave the other
        # teardowns running unawaited on a loop that is about to close.
        await asyncio.gather(
            *(self._teardown_client(client) for client in self.client_list.values())
        )
        await asyncio.gather(
            *(
                task
                for connection in self.manager.connections
                if (task := connection.loop_task) is not None
            ),
            return_exceptions=True,  # stopped engines wind down via CancelledError
        )
        self._torn_down = True
        self._wake_removal_waiters()

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
