"""The MQTT connection family.

Two transports over one shared identity (:class:`MqttParams`): :mod:`.sync` (paho,
a network thread wrapped as a :class:`~..transport.Transport`) and :mod:`.aio`
(aiomqtt, on the loop as an :class:`~..transport.AsyncTransport`). Each pools by
broker endpoint and routes by topic. Importing this package drags neither client
library -- both build their client lazily, so a base install imports cleanly.
"""

from simplyprint_ws_client.contrib.connection.mqtt.aio import (
    AsyncMqttPool,
    AsyncMqttTransport,
)
from simplyprint_ws_client.contrib.connection.mqtt.common import (
    MqttParams,
    mqtt_topic_matches,
)
from simplyprint_ws_client.contrib.connection.mqtt.sync import (
    MqttConnectionManager,
    MqttPool,
    MqttTransport,
)

__all__ = [
    "MqttParams",
    "mqtt_topic_matches",
    "MqttTransport",
    "MqttPool",
    "MqttConnectionManager",
    "AsyncMqttTransport",
    "AsyncMqttPool",
]
