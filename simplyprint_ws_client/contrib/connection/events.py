"""The connection event vocabulary -- the single language every wire speaks.

A transport publishes these on an :class:`~simplyprint_ws_client.events.EventBus`,
keyed by type, so a consumer subscribes with ``bus.on(Connected, handler)`` and
the handler receives the typed instance. The same four events flow whether the
wire underneath is MQTT or WebSocket, sync or async -- a consumer that only
listens never learns which it got.

Every event carries the :attr:`~ConnectionEvent.generation` it belongs to: a
monotonic epoch that the transport bumps once per established attempt. A consumer
compares it to discard work queued against a link that has since dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simplyprint_ws_client.events import Event


@dataclass(frozen=True, eq=False)
class ConnectionEvent(Event):
    """Base of every event a transport publishes."""

    generation: int


@dataclass(frozen=True, eq=False)
class Connecting(ConnectionEvent):
    """The transport began reaching for the endpoint (first try or a recovery)."""


@dataclass(frozen=True, eq=False)
class Connected(ConnectionEvent):
    """A live wire is up and able to carry messages.

    Its :attr:`generation` is the epoch the wire just entered -- the value
    subsequent :class:`MessageReceived` events for this link will carry.
    """


@dataclass(frozen=True, eq=False)
class Disconnected(ConnectionEvent):
    """The live wire went down.

    ``code`` is the library-specific reason the attempt ended -- typically the
    exception raised from the wire (a
    :class:`~simplyprint_ws_client.contrib.connection.transport.TransientError` or
    :class:`~simplyprint_ws_client.contrib.connection.transport.FatalError`), or ``None``
    for a clean teardown. It is informational only: the reconnect loop keeps retrying
    regardless of what ended the attempt.
    """

    code: Optional[object] = None


@dataclass(frozen=True, eq=False)
class MessageReceived(ConnectionEvent):
    """An inbound wire message, tagged with the generation it arrived on.

    ``message`` is wire-shaped (a decoded MQTT message, a WebSocket frame); the
    transport does not parse any brand protocol.
    """

    message: object
