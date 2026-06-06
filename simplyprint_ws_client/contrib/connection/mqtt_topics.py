"""MQTT topic matching primitives that do not import the MQTT client library."""

from __future__ import annotations


def mqtt_topic_matches(registered_topic: str, incoming_topic: str) -> bool:
    """Return whether an incoming topic is covered by an MQTT subscription."""
    if registered_topic == incoming_topic:
        return True

    if not registered_topic.endswith("/#"):
        return False

    prefix = registered_topic[:-2]
    return incoming_topic == prefix or incoming_topic.startswith(f"{prefix}/")
