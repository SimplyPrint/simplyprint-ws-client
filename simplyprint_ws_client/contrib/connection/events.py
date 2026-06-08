"""The transport event vocabulary -- the single language every wire speaks.

A transport (sync or async, MQTT or WebSocket) publishes these on its event bus;
a consumer subscribes and never learns which wire produced them. That uniformity
is what lets a brand sit on a threaded paho link or an async WebSocket link
unchanged. They are frozen dataclasses keyed by type, so a consumer subscribes
with ``bus.on(Connected, handler)`` and the handler receives the typed instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from simplyprint_ws_client.contrib.connection.state import ConnectionState

__all__ = [
    "TransportEvent",
    "Connected",
    "Disconnected",
    "MessageReceived",
    "ConnectionSuspect",
    "StateChanged",
]


@dataclass(frozen=True)
class TransportEvent:
    """Base of every event a transport publishes on its ``events`` bus."""


@dataclass(frozen=True)
class Connected(TransportEvent):
    """The link is up (first connect or a recovery)."""


@dataclass(frozen=True)
class Disconnected(TransportEvent):
    """The link went down.

    ``transient`` marks a drop the transport is already recovering from on its
    own -- a consumer can tolerate it until the failures pile up, rather than
    treating every blip as a hard disconnect.
    """

    reason: str = ""
    transient: bool = False


@dataclass(frozen=True)
class MessageReceived(TransportEvent):
    """An inbound message.

    ``payload`` is wire-shaped -- text for a WebSocket, the broker message for
    MQTT. The transport does not parse brand protocol; routing a pooled
    endpoint's message to the right client is the pool's job.
    """

    payload: Any


@dataclass(frozen=True)
class ConnectionSuspect(TransportEvent):
    """Repeated connect failures: the endpoint may be unreachable. Advisory --
    the transport keeps retrying regardless."""

    error: Optional[BaseException] = None


@dataclass(frozen=True)
class StateChanged(TransportEvent):
    """The transport's :class:`ConnectionState` changed."""

    state: ConnectionState
