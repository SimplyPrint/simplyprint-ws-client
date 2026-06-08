"""Shared MQTT identity + topic primitives (no MQTT client library imported).

These are what the sync (paho) and async (aiomqtt) families genuinely share: the
broker-endpoint key both pool by, the subscription wildcard match, and the topic
extractor each pool uses to route a message to the right lease.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class MqttParams(NamedTuple):
    """Hashable identity of an MQTT broker connection -- the pool key."""

    host: str
    port: int
    username: str = ""
    password: str = ""

    def __str__(self) -> str:
        return f"mqtts://{self.username}:<redacted>@{self.host}:{self.port}"


def mqtt_topic_matches(registered_topic: str, incoming_topic: str) -> bool:
    """Return whether an incoming topic is covered by an MQTT subscription."""
    if registered_topic == incoming_topic:
        return True

    if not registered_topic.endswith("/#"):
        return False

    prefix = registered_topic[:-2]
    return incoming_topic == prefix or incoming_topic.startswith(f"{prefix}/")


def topic_of(payload: Any) -> str:
    """A pool's topic extractor: an MQTT message always carries ``.topic``."""
    return str(payload.topic)
