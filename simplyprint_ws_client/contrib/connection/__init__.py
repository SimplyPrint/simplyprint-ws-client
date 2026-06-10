"""Pooled, self-healing connection transports driven by events."""

from __future__ import annotations

from simplyprint_ws_client.contrib.connection import events, mqtt, websocket
from simplyprint_ws_client.contrib.connection.aiohttp import Aiohttp
from simplyprint_ws_client.contrib.connection.aiomqtt import AioMqtt
from simplyprint_ws_client.contrib.connection.connection import Connection, MqttConnection, WsConnection
from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    Connecting,
    ConnectionEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.contrib.connection.keepalive import ConnectionKeepalive, Keepalive, KeepaliveTimeout
from simplyprint_ws_client.contrib.connection.messages import (
    MqttMessage,
    QoS,
    WsKind,
    WsMessage,
)
from simplyprint_ws_client.contrib.connection.options import ConnectionOptions, WireKeepalive
from simplyprint_ws_client.contrib.connection.paho import Paho
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.pool import Endpoint, Pool
from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    FatalError,
    MqttTransport,
    NotConnected,
    TransientError,
    Transport,
    WsTransport,
    topic_matches,
)
from simplyprint_ws_client.contrib.connection.errors import ErrorCode, TransportError
from simplyprint_ws_client.contrib.connection.websockets import Websockets

ws = websocket

__all__ = [
    "events",
    "mqtt",
    "websocket",
    "ws",
    "AioMqtt",
    "Aiohttp",
    "Connected",
    "Connecting",
    "Connection",
    "ConnectionEvent",
    "ConnectionKeepalive",
    "ConnectionOptions",
    "ConnectionState",
    "Disconnected",
    "Endpoint",
    "ErrorCode",
    "FatalError",
    "Keepalive",
    "KeepaliveTimeout",
    "MessageReceived",
    "MqttConnection",
    "MqttMessage",
    "MqttTransport",
    "NotConnected",
    "Paho",
    "Pool",
    "QoS",
    "Reconnecting",
    "RetryPolicy",
    "TransientError",
    "TransportError",
    "Transport",
    "Websockets",
    "WireKeepalive",
    "WsConnection",
    "WsKind",
    "WsMessage",
    "WsTransport",
    "topic_matches",
]
