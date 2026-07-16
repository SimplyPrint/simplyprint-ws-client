"""The wire contract the pool drives.

A :class:`Transport` is a supervised link to ONE endpoint that the
:class:`~simplyprint_ws_client.wire.pool.Pool` shares across leases. It
owns the whole reliability story -- connect, reconnect, state, generation -- and
publishes :class:`~simplyprint_ws_client.wire.events.WireEvent` s on
its :attr:`~Transport.events` bus so a consumer drives it by events, never by a
thread. Where the work runs (a paho network thread, an asyncio task) is the
transport's private business and never leaks across this seam.

:class:`MqttTransport` adds topic subscription; MQTT routing belongs to the pool
front door because it is a fan-out concern, not a wire lifecycle concern.
:class:`WsTransport` has no topics and broadcasts every message to every lease.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Hashable

import yarl

from simplyprint_ws_client.events import EventBus

if TYPE_CHECKING:
    from simplyprint_ws_client.common.asyncio.event_loop_provider import (
        EventLoopProvider,
    )

from simplyprint_ws_client.wire.events import WireEvent
from simplyprint_ws_client.wire.errors import (
    AuthenticationError,
    FatalError,
    NotConnected,
    TransientError,
    TransportError,
)
from simplyprint_ws_client.wire.state import ConnectionState

__all__ = [
    "TransportError",
    "NotConnected",
    "TransientError",
    "FatalError",
    "AuthenticationError",
    "Transport",
    "MqttTransport",
    "WsTransport",
    "topic_matches",
    "is_wildcard_filter",
]


class Transport(ABC):
    """A supervised link to one endpoint, driven by events.

    A concrete transport owns its wire and its reliability loop and publishes
    lifecycle/message events on :attr:`events`. ``state`` and ``generation`` are
    read-only to consumers; ``generation`` advances once per established
    connection so a consumer can tell one live link from the next.
    """

    #: The endpoint this transport is bound to.
    url: yarl.URL
    #: Last-known lifecycle state (also published via the events).
    state: ConnectionState
    #: Monotonic connection epoch; advances once per established connection.
    generation: int
    #: Where consumers subscribe -- keyed by event type.
    events: EventBus[WireEvent]
    #: The loop this transport delivers its events on (every concrete family
    #: sets one; shutdown helpers schedule ``stop`` through it).
    provider: "EventLoopProvider"

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the wire currently holds a live, usable connection."""

    @abstractmethod
    def start(self) -> None:
        """Begin keeping the link up (connect, then auto-reconnect).

        Idempotent and fire-and-forget: it returns immediately and readiness is
        reported through :class:`~simplyprint_ws_client.wire.events.Connected`.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Tear the link down permanently and release resources. Idempotent."""

    @abstractmethod
    async def send(self, message: object) -> None:
        """Send ``message`` over the wire.

        Raises :class:`NotConnected` if there is no live link. For a message whose
        QoS requires an acknowledgement the call awaits that ack.
        """

    @abstractmethod
    def trip(self, generation: int, reason: Exception) -> None:
        """End one live generation so supervision replaces its wire."""

    def supervising(self) -> bool:
        """Whether the transport is still trying to keep the link up.

        ``True`` by default (a self-healing wire never stops on its own). A
        A stopped transport returns ``False``.
        """
        return True


class MqttTransport(Transport):
    """A broker transport: many topics multiplexed over one shared socket.

    Subscriptions are refcounted across leases by the transport. The pool decides
    how inbound MQTT messages route to leases by using its configured route
    function.
    """

    @abstractmethod
    def subscribe(self, topic: str) -> None:
        """Assert a subscription for ``topic`` on the shared socket."""

    @abstractmethod
    def unsubscribe(self, topic: str) -> None:
        """Drop a subscription for ``topic`` from the shared socket."""

class WsTransport(Transport):
    """A 1:1 WebSocket transport: no topics, every message is the lease's.

    The pool has no route function for WebSockets, so every inbound frame is
    broadcast to every lease on the link.
    """


def topic_matches(subscription: str, topic: Hashable) -> bool:
    """Whether an incoming ``topic`` is covered by an MQTT ``subscription``.

    Implements MQTT filter matching: ``+`` matches exactly one level and a
    trailing ``#`` matches every remaining level (including none, so ``foo/#``
    covers ``foo`` itself). An exact string match always wins.
    """
    if not isinstance(topic, str):
        return False

    if subscription == topic:
        return True

    levels = subscription.split("/")
    topic_levels = topic.split("/")

    for i, pattern in enumerate(levels):
        if pattern == "#":
            return i == len(levels) - 1
        if i >= len(topic_levels):
            return False
        if pattern != "+" and pattern != topic_levels[i]:
            return False

    return len(topic_levels) == len(levels)


def is_wildcard_filter(subscription: Hashable) -> bool:
    """Whether an MQTT subscription filter contains a ``+`` or ``#`` wildcard."""
    if not isinstance(subscription, str):
        return False
    return any(level in ("+", "#") for level in subscription.split("/"))
