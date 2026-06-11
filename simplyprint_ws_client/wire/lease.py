"""The public lease a printer client holds for one shared transport."""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Callable,
    Coroutine,
    Generic,
    Hashable,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.asyncio.utils import submit_coro_threadsafe

from simplyprint_ws_client.wire.events import (
    Connected,
    WireEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.messages import (
    MqttMessage,
    WsMessage,
    as_ws_message,
)
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
    MqttTransport,
    NotConnected,
    Transport,
    WsTransport,
    topic_matches,
)

if TYPE_CHECKING:
    from simplyprint_ws_client.wire.keepalive import Keepalive
    from simplyprint_ws_client.wire.pool import Pool

__all__ = ["Lease", "MqttLease", "WsLease"]

T = TypeVar("T", bound=Transport)


class Lease(Generic[T]):
    """One caller's lease on a shared transport.

    Subscribe handlers on :attr:`event_bus`, ``await`` :meth:`ready`, ``send``, and
    :meth:`close` when done. The pool routes transport events into this lease's
    owned courier, so slow handlers cannot stall the transport recv loop.
    """

    def __init__(
        self,
        pool: "Pool[T]",
        transport: T,
        url: yarl.URL,
        endpoint_key: Hashable,
        *,
        provider: Optional[EventLoopProvider] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.pool = pool
        self.transport = transport
        self.url = url
        self.endpoint_key = endpoint_key
        self.provider = provider or EventLoopProvider.default()
        self.logger = logger or logging.getLogger("wire.lease")
        #: This lease's own filtered bus; the pool emits routed events here.
        self.event_bus: EventBus[WireEvent] = EventBus()
        self._courier = Courier(
            sink=self._emit_event,
            is_async_sink=True,
            provider=self.provider,
            policy=OverflowPolicy.DROP_OLDEST,
            maxsize=1024,
            lossless=self._lossless_event,
            logger=self.logger,
        )
        #: Topics this lease cares about; used to route messages to it.
        self.topics: Set[str] = set()
        self._close_callbacks: Set[Callable[[], None]] = set()
        self._tasks: Set[asyncio.Task] = set()
        #: Unresolved ``ready()`` futures; settled ``False`` when the lease
        #: closes so no waiter outlives its event source.
        self._ready_waiters: Set[asyncio.Future] = set()
        self.closed = False

    @property
    def generation(self) -> int:
        return self.transport.generation

    @property
    def state(self) -> ConnectionState:
        return self.transport.state

    @property
    def connected(self) -> bool:
        return self.transport.connected

    def wants(self, route: Hashable) -> bool:
        """Whether a message routed to ``route`` belongs to this lease.

        A lease with no tracked topics (a 1:1 wire) wants everything; otherwise it
        wants a topic covered by one of its subscriptions.
        """
        if not self.topics:
            return True
        return any(topic_matches(topic, route) for topic in self.topics)

    async def ready(self, timeout: Optional[float] = None) -> bool:
        """Resolve ``True`` on the next :class:`Connected`, ``False`` on give-up or
        timeout.

        Returns immediately if already connected. A permanent give-up (the
        transport reaching ``DISCONNECTED`` with no live wire after exhausting its
        retry policy) resolves ``False``.
        """
        if self.connected:
            return True

        loop = self.provider.event_loop
        result: asyncio.Future = loop.create_future()

        def on_connected(_: Connected) -> None:
            if not result.done():
                result.set_result(True)

        def on_disconnected(_: Disconnected) -> None:
            # Only a *terminal* disconnect (the supervisor gave up) ends the wait;
            # an ordinary drop is followed by another connect attempt.
            if not result.done() and not self.transport.supervising():
                result.set_result(False)

        self.event_bus.on(Connected, on_connected)
        self.event_bus.on(Disconnected, on_disconnected)
        self._ready_waiters.add(result)
        try:
            if self.closed and not result.done():
                result.set_result(False)
            elif self.connected and not result.done():
                result.set_result(True)
            elif not self.transport.supervising() and not result.done():
                result.set_result(False)
            if timeout is None:
                return await result
            return await asyncio.wait_for(result, timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._ready_waiters.discard(result)
            self.event_bus.off(Connected, on_connected)
            self.event_bus.off(Disconnected, on_disconnected)

    async def ready_transport(self) -> T:
        """Await a live connection and return the shared transport."""
        if not await self.ready():
            raise NotConnected("transport did not become connected")
        return self.transport

    async def send(self, message: object) -> None:
        """Send ``message`` over the shared transport (raises if the link is down)."""
        await self.transport.send(message)

    def deliver(self, event: WireEvent) -> None:
        """Queue a routed event for this lease without blocking the transport recv loop."""
        if not self.closed:
            self._courier.post(event)

    def _schedule(
        self, coro: Coroutine[object, object, object]
    ) -> Tuple[bool, Optional[asyncio.Task]]:
        """Schedule ``coro`` on this lease's loop from any thread.

        Returns ``(accepted, task)`` -- the task is only available when already
        on the lease's loop; a cross-thread submission creates it on the loop
        later. A rejected coroutine (lease closed, no running loop) is closed so
        it never leaks a 'never awaited' warning.
        """
        if self.closed:
            coro.close()
            return False, None
        try:
            loop = self.provider.event_loop
        except RuntimeError:
            coro.close()
            return False, None
        # The lease retains its child tasks via the create_task seam.
        return submit_coro_threadsafe(loop, coro, create_task=self._create_task_on_loop)

    def create_task(
        self, coro: Coroutine[object, object, object]
    ) -> Optional[asyncio.Task]:
        """Create a child task owned by this lease."""
        return self._schedule(coro)[1]

    async def cancel_tasks(self) -> None:
        """Cancel every child task owned by this lease."""
        current = asyncio.current_task()
        tasks = tuple(task for task in self._tasks if task is not current)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            self._tasks.discard(task)

    def submit(self, coro: Coroutine[object, object, object]) -> bool:
        """Schedule a coroutine on this lease's loop from sync code."""
        return self._schedule(coro)[0]

    def send_soon(self, message: object) -> bool:
        """Schedule ``send(message)`` if the lease is connected."""
        if not self.connected:
            return False
        return self.submit(self.send(message))

    def close_soon(self) -> bool:
        """Schedule ``close()`` from sync code."""
        return self.submit(self.close())

    def on_close(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Run ``callback`` when this lease closes."""
        self._close_callbacks.add(callback)
        return callback

    def off_close(self, callback: Callable[[], None]) -> None:
        """Remove a close callback registered with :meth:`on_close`."""
        self._close_callbacks.discard(callback)

    def keepalive(self, policy: "Keepalive"):
        """Attach an application-level keepalive policy to this lease."""
        from simplyprint_ws_client.wire.keepalive import (
            ConnectionKeepalive,
        )

        return ConnectionKeepalive(self, policy, logger=self.logger).start()

    async def close(self) -> None:
        """Release this lease back to the pool (refcount--).

        The last lease on the endpoint stops and drops the transport.
        """
        if self.closed:
            return
        self.closed = True
        for callback in tuple(self._close_callbacks):
            try:
                callback()
            except Exception:  # noqa: BLE001 -- close must continue cleanup
                self.logger.warning("connection close callback failed", exc_info=True)
        self._close_callbacks.clear()
        # Settle pending ready() waiters before their event source disappears.
        for waiter in tuple(self._ready_waiters):
            if not waiter.done():
                waiter.set_result(False)
        self._ready_waiters.clear()
        self.event_bus.clear_all()
        self._courier.close(drain=False)
        await self.cancel_tasks()
        transport = self.pool.release(self)
        if transport is not None:
            await transport.stop()

    def _create_task_on_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        coro: Coroutine[object, object, object],
    ) -> Optional[asyncio.Task]:
        if self.closed:
            coro.close()
            return None
        task = loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            self.logger.warning(
                "connection child task failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _emit_event(self, event: WireEvent) -> None:
        if not self.closed:
            await self.event_bus.emit(event)

    @staticmethod
    def _lossless_event(event: WireEvent) -> bool:
        return not isinstance(event, MessageReceived) or event.lossless


class MqttLease(Lease[MqttTransport]):
    """A lease on an MQTT broker transport -- subscribe to topics on one socket."""

    async def send(self, message: Union[MqttMessage, str, bytes]) -> None:
        """Send an ``MqttMessage`` (or raw bytes/str on the URL's default topic).

        The front door wraps bytes/str into an ``MqttMessage`` on the first
        ``?topic=`` of the URL before delegating here.
        """
        if isinstance(message, (bytes, str)):
            topics = list(self.url.query.getall("topic", []))
            if not topics:
                raise ValueError(
                    "mqtt: send(bytes/str) needs a default ?topic= on the connect URL"
                )
            payload = message.encode() if isinstance(message, str) else message
            message = MqttMessage(topics[0], payload)
        await self.transport.send(message)

    async def subscribe(self, topic: str) -> None:
        """Track ``topic`` on this lease (for routing) and assert it on the shared
        socket (refcounted across leases by the broker transport)."""
        self.topics.add(topic)
        self.pool.add_route(self, topic)
        await self.transport.subscribe(topic)

    def subscribe_soon(self, topic: str) -> None:
        """Subscribe from sync code: interest is recorded on the lease NOW (so
        routing is correct the instant the link comes up, and re-asserted on every
        (re)connect), and the wire subscribe runs as a task on the lease's loop."""
        self.topics.add(topic)
        self.pool.add_route(self, topic)
        self.create_task(self.transport.subscribe(topic))

    async def unsubscribe(self, topic: str) -> None:
        """Stop tracking ``topic`` and drop the subscription on the shared socket."""
        self.topics.discard(topic)
        self.pool.remove_route(self, topic)
        await self.transport.unsubscribe(topic)

    async def close(self) -> None:
        if self.closed:
            return
        for topic in tuple(self.topics):
            try:
                await self.unsubscribe(topic)
            except Exception:  # noqa: BLE001 -- a broken link must not abort close
                # Local route bookkeeping is already dropped (unsubscribe does
                # it before the broker call); transport teardown handles the
                # broker side.
                self.logger.debug(
                    "unsubscribe(%s) failed during close", topic, exc_info=True
                )
        await super().close()


class WsLease(Lease[WsTransport]):
    """A lease on a 1:1 WebSocket transport -- every frame is this lease's."""

    async def send(self, message: Union[str, bytes, WsMessage]) -> None:
        """Send a ``WsMessage`` (or raw str -> text frame, bytes -> binary frame).

        The front door wraps a bare str/bytes into the matching frame type before
        delegating here.
        """
        await self.transport.send(as_ws_message(message).payload)
