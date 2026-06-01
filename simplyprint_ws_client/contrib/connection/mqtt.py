"""MQTT transport for the connection pool (paho-mqtt).

Multiple brands speak MQTT over TLS with the same pooling needs; this module
holds everything they share. A brand provides:

* a ``params_from_config`` callable producing :class:`MqttConnectionParams`,
* its connected / disconnected / message event types,
* a keepalive command (via :meth:`MqttConnectionManager._send_keepalive`),

and -- only if needed -- overrides for auth handling (e.g. a re-handshake on
rejection) or wildcard topic matching.
"""

from __future__ import annotations

import ssl
from typing import Hashable, NamedTuple, Type, Union

import paho.mqtt.client as mqtt
from paho.mqtt.reasoncodes import ReasonCode

from simplyprint_ws_client.contrib.connection.pool import (
    KEEPALIVE_TIMEOUT_MS,
    ClientBucket,
    ConnectionManager,
    PooledConnection,
    TClient,
)

#: A paho CONNACK / DISCONNECT reason. paho-mqtt v2 hands us a ``ReasonCode``,
#: but the legacy callbacks (and some error paths) still surface a bare int, and
#: it can be absent entirely -- so the neutral union covers all three.
PahoReasonCode = Union[ReasonCode, int, None]


class MqttConnectionParams(NamedTuple):
    """Hashable identity of an MQTT broker connection."""

    username: str
    password: str
    host: str
    port: int

    def __str__(self) -> str:
        return f"mqtts://{self.username}:<redacted>@{self.host}:{self.port}"


#: Disconnect reason codes we treat as transient (tolerate, let paho reconnect).
TRANSIENT_DISCONNECT_CODES = (
    mqtt.MQTT_ERR_KEEPALIVE,
    mqtt.MQTT_ERR_CONN_LOST,
    mqtt.MQTT_ERR_AGAIN,
)


class MqttConnection(PooledConnection[MqttConnectionParams]):
    """A single physical paho-mqtt connection shared by matching clients."""

    def __init__(
        self,
        bucket: ClientBucket,
        params: MqttConnectionParams,
        *,
        connected_event: Hashable,
        disconnected_event: Hashable,
        message_event: Hashable,
        **kwargs: object,
    ) -> None:
        super().__init__(
            bucket,
            params,
            connected_event=connected_event,
            disconnected_event=disconnected_event,
            **kwargs,
        )

        self.message_event = message_event

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,  # noqa
            reconnect_on_failure=True,
        )
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS, cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=5)

        self.client.on_connect = self._on_connect
        self.client.on_connect_fail = self._on_connect_fail
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.client.username_pw_set(params.username, params.password)
        self.client.connect_async(
            params.host, params.port, keepalive=KEEPALIVE_TIMEOUT_MS // 1000
        )

        if self.client.loop_start() != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("Failed to start MQTT loop.")

        self.logger.info("Connecting to %s", params)

    @property
    def connected(self) -> bool:
        return self.client.is_connected()

    # -- brand hooks --------------------------------------------------------

    def _connect_succeeded(self, reason_code: PahoReasonCode) -> bool:
        """Whether the CONNACK indicates success. Override for stricter checks."""
        return True

    def _on_auth_failure(self, reason_code: PahoReasonCode) -> None:
        """Handle a rejected connection. Default: treat as a connect failure."""
        self.handle_connect_failed()

    # -- paho callbacks (run on the paho network loop thread) ---------------

    def _on_connect(self, _client, _userdata, _flags, reason_code, *_a, **_kw):  # noqa
        if self._connect_succeeded(reason_code):
            self.handle_connected()
        else:
            self.logger.warning(
                "Connection rejected by %s: %s", self.params, reason_code
            )
            self._on_auth_failure(reason_code)

    def _on_connect_fail(self, _client, _userdata, *_a, **_kw):  # noqa
        self.handle_connect_failed()

    def _on_message(self, _client, _userdata, message: mqtt.MQTTMessage, *_a, **_kw):  # noqa
        client = self.bucket.get_from_topic(message.topic)
        if not client:
            self.logger.warning(
                "Message on un-linked topic %s; perhaps orphaned connection.",
                message.topic,
            )
            return
        client.event_bus_worker.emit_sync(self.message_event, message)

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, *_a, **_kw):  # noqa
        transient = reason_code in TRANSIENT_DISCONNECT_CODES
        self.handle_disconnected(
            transient=transient, reason="Disconnected from MQTT broker"
        )

    def stop(self) -> None:
        self.logger.info("Stopping MQTT connection to %s", self.params)
        super().stop()
        self.client.disconnect()
        self.client.loop_stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


class MqttConnectionManager(ConnectionManager[TClient, MqttConnectionParams]):
    """Connection manager wired for the MQTT transport.

    Concrete brand managers set the event types + ``params_factory`` (as class
    attributes) and implement :meth:`_send_keepalive`. Override
    :attr:`connection_class` to use a brand-specific :class:`MqttConnection`.
    """

    connection_class: Type[MqttConnection] = MqttConnection

    def _create_connection(self, params: MqttConnectionParams) -> MqttConnection:
        return self.connection_class(
            bucket=self.bucket,
            params=params,
            connected_event=self.connected_event,
            disconnected_event=self.disconnected_event,
            message_event=self.message_event,
            parent_stoppable=self,
        )

    def _refresh_subscription(
        self, client: TClient, connection: MqttConnection
    ) -> None:
        connection.client.subscribe(client.report_topic)

    def _unsubscribe(self, client: TClient, connection: MqttConnection) -> None:
        connection.client.unsubscribe(client.report_topic)
