"""The device-facing connection front: pooled, self-healing leases on shared wires.

``mqtt.connect(url)`` / ``websocket.connect(url)`` hand back a :class:`Lease` on a
shared :class:`~simplyprint_ws_client.common.wire.transport.Transport`. The wire
engine itself (transports, the reconnect loop, wire events) is side-neutral and
lives in :mod:`simplyprint_ws_client.common.wire`; this package owns what is
device-specific about it -- sharing one socket across many printers (``Pool``),
the per-caller :class:`Lease`, app-level keepalive, and the connect front doors.

The wire surface is re-exported here for convenience, so a device-side consumer
imports one package.
"""

from __future__ import annotations

from simplyprint_ws_client.common.wire import (
    Connected,
    Connecting,
    ConnectionState,
    Disconnected,
    ErrorCode,
    FatalError,
    MessageReceived,
    MqttMessage,
    MqttTransport,
    NotConnected,
    QoS,
    Reconnecting,
    RetryPolicy,
    TransientError,
    Transport,
    TransportError,
    WireEvent,
    WsKind,
    WsMessage,
    WsTransport,
    topic_matches,
)
from simplyprint_ws_client.common.wire.aiohttp import Aiohttp
from simplyprint_ws_client.common.wire.aiomqtt import AioMqtt
from simplyprint_ws_client.common.wire.paho import Paho
from simplyprint_ws_client.common.wire.websockets import Websockets

from simplyprint_ws_client.device.connection import mqtt, websocket
from simplyprint_ws_client.device.connection.keepalive import (
    ConnectionKeepalive,
    Keepalive,
    KeepaliveTimeout,
)
from simplyprint_ws_client.device.connection.lease import Lease, MqttLease, WsLease
from simplyprint_ws_client.device.connection.options import (
    ConnectionOptions,
    WireKeepalive,
)
from simplyprint_ws_client.device.connection.pool import Endpoint, Pool

ws = websocket

__all__ = [
    "mqtt",
    "websocket",
    "ws",
    "AioMqtt",
    "Aiohttp",
    "Connected",
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
    "MqttTransport",
    "NotConnected",
    "Paho",
    "Pool",
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
]
