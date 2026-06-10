"""The neutral self-healing wire engine both sides ride.

A :class:`Transport` is a supervised link to one endpoint; :class:`Reconnecting`
is the one supervision loop concrete wires fill four hooks on. The cloud protocol
composes a wire for the SimplyPrint socket exactly like the device front doors do
for printers -- nothing in here knows which side it is carrying, which is why this
lives in ``common`` and not on either side.
"""

from __future__ import annotations

from simplyprint_ws_client.common.wire.errors import ErrorCode, TransportError
from simplyprint_ws_client.common.wire.events import (
    Connected,
    Connecting,
    Disconnected,
    MessageReceived,
    WireEvent,
)
from simplyprint_ws_client.common.wire.messages import (
    MqttMessage,
    QoS,
    WsKind,
    WsMessage,
)
from simplyprint_ws_client.common.wire.policy import RetryPolicy
from simplyprint_ws_client.common.wire.reconnect import Reconnecting
from simplyprint_ws_client.common.wire.state import ConnectionState
from simplyprint_ws_client.common.wire.transport import (
    FatalError,
    MqttTransport,
    NotConnected,
    TransientError,
    Transport,
    WsTransport,
    topic_matches,
)

__all__ = [
    "Connected",
    "Connecting",
    "ConnectionState",
    "Disconnected",
    "ErrorCode",
    "FatalError",
    "MessageReceived",
    "MqttMessage",
    "MqttTransport",
    "NotConnected",
    "QoS",
    "Reconnecting",
    "RetryPolicy",
    "TransientError",
    "Transport",
    "TransportError",
    "WireEvent",
    "WsKind",
    "WsMessage",
    "WsTransport",
    "topic_matches",
]
