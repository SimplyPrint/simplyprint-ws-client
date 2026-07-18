"""The connection subsystem: self-healing wires, shared pools, per-caller leases.

One semantic home for everything connection: the supervised :class:`Transport`
engine (:class:`Reconnecting` + the concrete backends), the wire events both
sides ride, the :class:`Pool` that shares one socket per endpoint, the
per-caller :class:`Lease` (with app-level keepalive), and the ``mqtt`` /
``websocket`` connect front doors. The SimplyPrint socket and a printer link
compose the same machinery -- nothing in here knows which side it carries.
"""

from __future__ import annotations

from simplyprint_ws_client.wire import mqtt, websocket
from simplyprint_ws_client.wire.errors import ErrorCode, TransportError
from simplyprint_ws_client.wire.events import (
    ActivityTimeout,
    Connected,
    Connecting,
    Disconnected,
    MessageReceived,
    WireEvent,
)
from simplyprint_ws_client.wire.keepalive import (
    ConnectionKeepalive,
    Keepalive,
    KeepaliveTimeout,
)
from simplyprint_ws_client.wire.lease import Lease, MqttLease, WsLease
from simplyprint_ws_client.wire.messages import MqttMessage, QoS, WsKind, WsMessage
from simplyprint_ws_client.wire.mqtt_probe import (
    MqttProbeOutcome,
    MqttProbeResult,
    probe_mqtt,
)
from simplyprint_ws_client.wire.options import ConnectionOptions, WireKeepalive
from simplyprint_ws_client.wire.paho import Paho
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.pool import Endpoint, Pool
from simplyprint_ws_client.wire.pools import PoolRegistry
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
    AuthenticationError,
    FatalError,
    MqttTransport,
    NotConnected,
    TransientError,
    Transport,
    WsTransport,
    topic_matches,
)
from simplyprint_ws_client.wire.websockets import Websockets

ws = websocket

__all__ = [
    "mqtt",
    "websocket",
    "ws",
    "ActivityTimeout",
    "Connected",
    "AuthenticationError",
    "Connecting",
    "ConnectionKeepalive",
    "ConnectionOptions",
    "ConnectionState",
    "Disconnected",
    "Endpoint",
    "ErrorCode",
    "FatalError",
    "Keepalive",
    "KeepaliveTimeout",
    "Lease",
    "MessageReceived",
    "MqttLease",
    "MqttMessage",
    "MqttProbeOutcome",
    "MqttProbeResult",
    "MqttTransport",
    "NotConnected",
    "Paho",
    "Pool",
    "PoolRegistry",
    "QoS",
    "Reconnecting",
    "RetryPolicy",
    "TransientError",
    "Transport",
    "TransportError",
    "Websockets",
    "WireEvent",
    "WireKeepalive",
    "WsKind",
    "WsLease",
    "WsMessage",
    "WsTransport",
    "topic_matches",
    "probe_mqtt",
]
