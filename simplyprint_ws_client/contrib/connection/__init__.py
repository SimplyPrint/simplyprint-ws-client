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
from .messages import (
    MqttMessage,
    QoS,
    WsBytesMessage,
    WsKind,
    WsMessage,
    WsTextMessage,
)
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
    "ConnectionState",
    "Disconnected",
    "Endpoint",
    "FatalError",
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
    "Transport",
    "Websockets",
    "WsBytesMessage",
    "WsConnection",
    "WsKind",
    "WsMessage",
    "WsTextMessage",
    "WsTransport",
    "topic_matches",
]
