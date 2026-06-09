"""The public lease a printer client holds for one shared transport."""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Generic,
    Hashable,
    Optional,
    Set,
    TypeVar,
)

import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    ConnectionEvent,
    Disconnected,
)
from simplyprint_ws_client.contrib.connection.messages import (
    MqttMessage,
    as_ws_message,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    MqttTransport,
    NotConnected,
    Transport,
    topic_matches,
)

if TYPE_CHECKING:
    from simplyprint_ws_client.contrib.connection.pool import Pool

__all__ = ["Connection", "MqttConnection", "WsConnection"]

T = TypeVar("T", bound=Transport)


class Connection(Generic[T]):
    """One caller's lease on a shared transport.

    Subscribe handlers on :attr:`event_bus`, ``await`` :meth:`ready`, ``send``, and
    :meth:`close` when done. The pool routes transport events to this lease's
    :class:`EventBus`; the lease itself does not own a delivery queue.
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
        self.backend = transport
        self.url = url
        self.endpoint_key = endpoint_key
        self.provider = provider or EventLoopProvider.default()
        self.logger = logger or logging.getLogger("conn.connection")
        #: This lease's own filtered bus; the pool emits routed events here.
        self.event_bus: EventBus[ConnectionEvent] = EventBus()
        #: Topics this lease cares about; used to route messages to it.
        self.topics: Set[str] = set()
        self.closed = False

    @property
    def generation(self) -> int:
        return self.backend.generation

    @property
    def state(self) -> ConnectionState:
        return self.backend.state

    @property
    def connected(self) -> bool:
        return self.backend.connected

    def wants(self, route: Hashable) -> bool:
        """Whether a message routed to ``route`` belongs to this lease.

        A lease with no tracked topics (a 1:1 wire) wants everything; otherwise it
        wants a topic covered by one of its subscriptions.
        """
        if not self.topics:
            return True
        return isinstance(route, str) and any(
            topic_matches(topic, route) for topic in self.topics
        )

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
            if not result.done() and not self.backend.supervising():
                result.set_result(False)

        self.event_bus.on(Connected, on_connected)
        self.event_bus.on(Disconnected, on_disconnected)
        try:
            if self.connected and not result.done():
                result.set_result(True)
            elif not self.backend.supervising() and not result.done():
                result.set_result(False)
            if timeout is None:
                return await result
            return await asyncio.wait_for(result, timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self.event_bus.off(Connected, on_connected)
            self.event_bus.off(Disconnected, on_disconnected)

    async def transport(self) -> T:
        """Await a live connection and return the backend transport."""
        if not await self.ready():
            raise NotConnected("transport did not become connected")
        return self.backend

    async def send(self, message: object) -> None:
        """Send ``message`` over the shared transport (raises if the link is down)."""
        await self.backend.send(message)

    async def close(self) -> None:
        """Release this lease back to the pool (refcount--).

        The last lease on the endpoint stops and drops the transport.
        """
        if self.closed:
            return
        self.closed = True
        self.event_bus.clear(*tuple(self.event_bus.listeners.keys()))
        transport = self.pool.release(self)
        if transport is not None:
            await transport.stop()


class MqttConnection(Connection[T]):
    """A lease on an MQTT broker transport -- subscribe to topics on one socket."""

    async def send(self, message: object) -> None:
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
        await self.backend.send(message)

    async def subscribe(self, topic: str) -> None:
        """Track ``topic`` on this lease (for routing) and assert it on the shared
        socket (refcounted across leases by the broker transport)."""
        self.topics.add(topic)
        if isinstance(self.backend, MqttTransport):
            await self.backend.subscribe(topic)

    async def unsubscribe(self, topic: str) -> None:
        """Stop tracking ``topic`` and drop the subscription on the shared socket."""
        self.topics.discard(topic)
        if isinstance(self.backend, MqttTransport):
            await self.backend.unsubscribe(topic)

    async def close(self) -> None:
        if self.closed:
            return
        for topic in tuple(self.topics):
            await self.unsubscribe(topic)
        await super().close()


class WsConnection(Connection[T]):
    """A lease on a 1:1 WebSocket transport -- every frame is this lease's."""

    async def send(self, message: object) -> None:
        """Send a ``WsMessage`` (or raw str -> text frame, bytes -> binary frame).

        The front door wraps a bare str/bytes into the matching frame type before
        delegating here.
        """
        await self.backend.send(as_ws_message(message).payload)
