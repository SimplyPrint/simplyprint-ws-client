"""The wire contract the pool drives.

A :class:`Transport` is a supervised link to ONE endpoint that the
:class:`~simplyprint_ws_client.contrib.connection.pool.Pool` shares across leases. It
owns the whole reliability story -- connect, reconnect, state, generation -- and
publishes :class:`~simplyprint_ws_client.contrib.connection.events.ConnectionEvent` s on
its :attr:`~Transport.events` bus so a consumer drives it by events, never by a
thread. Where the work runs (a paho network thread, an asyncio task) is the
transport's private business and never leaks across this seam.

:class:`MqttTransport` adds topic subscription; MQTT routing belongs to the pool
front door because it is a fan-out concern, not a wire lifecycle concern.
:class:`WsTransport` has no topics and broadcasts every message to every lease.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable

import yarl

from simplyprint_ws_client.common.events import EventBus

from simplyprint_ws_client.contrib.connection.events import ConnectionEvent
from simplyprint_ws_client.contrib.connection.errors import (
    FatalError,
    NotConnected,
    TransientError,
    TransportError,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState

__all__ = [
    "TransportError",
    "NotConnected",
    "TransientError",
    "FatalError",
    "Transport",
    "MqttTransport",
    "WsTransport",
    "topic_matches",
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
    events: EventBus[ConnectionEvent]

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the wire currently holds a live, usable connection."""

    @abstractmethod
    def start(self) -> None:
        """Begin keeping the link up (connect, then auto-reconnect).

        Idempotent and fire-and-forget: it returns immediately and readiness is
        reported through :class:`~simplyprint_ws_client.contrib.connection.events.Connected`.
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

    def supervising(self) -> bool:
        """Whether the transport is still trying to keep the link up.

        ``True`` by default (a self-healing wire never stops on its own). A
        :class:`~simplyprint_ws_client.contrib.connection.reconnect.Reconnecting`
        transport returns ``False`` once its retry policy is exhausted and it has
        permanently given up -- the signal a waiter uses to resolve "gave up".
        """
        return True


class MqttTransport(Transport):
    """A broker transport: many topics multiplexed over one shared socket.

    Subscriptions are refcounted across leases by the transport. The pool decides
    how inbound MQTT messages route to leases by using its configured route
    function.
    """

    @abstractmethod
    async def subscribe(self, topic: str) -> None:
        """Assert a subscription for ``topic`` on the shared socket."""

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Drop a subscription for ``topic`` from the shared socket."""


class WsTransport(Transport):
    """A 1:1 WebSocket transport: no topics, every message is the lease's.

    The pool has no route function for WebSockets, so every inbound frame is
    broadcast to every lease on the link.
    """


def topic_matches(subscription: str, topic: Hashable) -> bool:
    """Whether an incoming ``topic`` is covered by an MQTT ``subscription``.

    Supports the trailing multi-level ``#`` wildcard: ``foo/#`` matches ``foo``
    itself and anything beneath it (``foo/bar``, ``foo/bar/baz``). An exact
    string match always wins.
    """
    if not isinstance(topic, str):
        return False

    if subscription == topic:
        return True

    if not subscription.endswith("/#"):
        return False

    prefix = subscription[:-2]
    return topic == prefix or topic.startswith(f"{prefix}/")
