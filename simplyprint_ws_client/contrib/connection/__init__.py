"""Pooled, self-healing connection transports driven by events."""

from __future__ import annotations

from . import events, mqtt, websocket
from .aiohttp import Aiohttp
from .aiomqtt import AioMqtt
from .connection import Connection, MqttConnection, WsConnection
from .events import (
    Connected,
    Connecting,
    ConnectionEvent,
    Disconnected,
    MessageReceived,
)
from .keepalive import ConnectionKeepalive, Keepalive, KeepaliveTimeout
from .messages import (
    MqttMessage,
    QoS,
    WsKind,
    WsMessage,
)
from .options import ConnectionOptions, WireKeepalive
from .paho import Paho
from .policy import RetryPolicy
from .pool import Endpoint, Pool
from .reconnect import Reconnecting
from .state import ConnectionState
from .transport import (
    FatalError,
    MqttTransport,
    NotConnected,
    TransientError,
    Transport,
    WsTransport,
    topic_matches,
)
from .errors import ErrorCode, TransportError
from .websockets import Websockets

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
