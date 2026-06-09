"""Message contracts shared by the connection front doors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Union


class QoS(Enum):
    """How hard the transport works to deliver one message.

    ``AT_MOST_ONCE`` is fire-and-forget: under buffer overflow such a message may
    be dropped, and on send it is not acknowledged. ``AT_LEAST_ONCE`` is never
    dropped under overflow and, for wires that support acks (MQTT qos >= 1), a
    send awaits the broker's acknowledgement before returning.
    """

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1


@dataclass(frozen=True)
class MqttMessage:
    """One MQTT message on the wire."""

    topic: str
    payload: bytes
    qos: QoS = QoS.AT_MOST_ONCE
    retain: bool = False


class MqttInboundMessage(Protocol):
    """The MQTT message surface needed for routing inbound broker messages."""

    topic: str
    payload: bytes
    qos: int
    retain: bool


class WsKind(Enum):
    """Which of the two WebSocket data frames a message rides in."""

    TEXT = "text"
    BINARY = "binary"


@dataclass(frozen=True)
class WsMessage:
    """One WebSocket data frame, in or out."""

    kind: WsKind
    payload: Union[str, bytes]
    qos: QoS = field(default=QoS.AT_LEAST_ONCE)

    @classmethod
    def text(cls, text: str, *, qos: QoS = QoS.AT_LEAST_ONCE) -> "WsMessage":
        return cls(WsKind.TEXT, text, qos)

    @classmethod
    def binary(cls, data: bytes, *, qos: QoS = QoS.AT_LEAST_ONCE) -> "WsMessage":
        return cls(WsKind.BINARY, data, qos)


def as_ws_message(message: Union[str, bytes, WsMessage]) -> WsMessage:
    if isinstance(message, WsMessage):
        return message
    if isinstance(message, str):
        return WsMessage.text(message)
    if isinstance(message, (bytes, bytearray)):
        return WsMessage.binary(bytes(message))
    raise TypeError(f"cannot send {type(message).__name__} over a WebSocket")


def ws_message_for_payload(payload: Union[str, bytes]) -> WsMessage:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return WsMessage.binary(bytes(payload))
    return WsMessage.text(payload)


def mqtt_message_from_inbound(message: MqttInboundMessage) -> MqttMessage:
    qos = QoS.AT_LEAST_ONCE if int(message.qos) > 0 else QoS.AT_MOST_ONCE
    return MqttMessage(str(message.topic), bytes(message.payload), qos, message.retain)


def message_qos(message: object) -> QoS:
    if isinstance(message, (MqttMessage, WsMessage)):
        return message.qos
    return QoS.AT_LEAST_ONCE
