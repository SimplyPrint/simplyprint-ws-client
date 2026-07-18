"""Paho MQTT transport with one client and worker per endpoint lifetime."""

from __future__ import annotations

import logging
import os
import ssl
import tempfile
import threading
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Union

import yarl

from simplyprint_ws_client.common.asyncio.concurrent import run_in_thread
from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.wire.errors import (
    AuthenticationError,
    ErrorCode,
    TransportError,
)
from simplyprint_ws_client.wire.events import (
    Connected,
    Connecting,
    Disconnected,
    MessageReceived,
    WireEvent,
)
from simplyprint_ws_client.wire.messages import MqttMessage, mqtt_message_from_inbound
from simplyprint_ws_client.wire.options import TlsClientAuth
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
    MqttTransport,
    NotConnected,
    TransientError,
)

if TYPE_CHECKING:
    from paho.mqtt.client import MQTTMessage
    from paho.mqtt.reasoncodes import ReasonCode

__all__ = [
    "Paho",
    "PahoClientFactory",
    "client_cert_ssl_context",
    "default_paho_client",
]

PahoReasonCode = Optional[Union[int, "ReasonCode"]]
PUBLISH_ACK_TIMEOUT = 30.0
WORKER_RESTART_DELAY = 1.0
WORKER_SHUTDOWN_TIMEOUT = 5.0
EVENT_CAPACITY = 1024
RECONNECT_MIN_DELAY = 1
RECONNECT_MAX_DELAY = 5
AUTHENTICATION_FAILURE_CODES = frozenset((4, 5, 134, 135))


class PahoPublishInfo(Protocol):
    rc: int

    def wait_for_publish(self, timeout: Optional[float] = None) -> None: ...

    def is_published(self) -> bool: ...


class PahoClient(Protocol):
    on_pre_connect: Optional[Callable[..., None]]
    on_connect: Optional[Callable[..., None]]
    on_connect_fail: Optional[Callable[..., None]]
    on_message: Optional[Callable[..., None]]
    on_disconnect: Optional[Callable[..., None]]

    def username_pw_set(
        self, username: Optional[str], password: Optional[str]
    ) -> None: ...

    def connect_async(self, host: str, port: int, keepalive: int) -> None: ...

    def loop_forever(
        self, timeout: float = 1.0, retry_first_connection: bool = False
    ) -> int: ...

    def disconnect(self) -> object: ...

    def is_connected(self) -> bool: ...

    def subscribe(self, topic: str) -> tuple[int, int]: ...

    def unsubscribe(self, topic: str) -> tuple[int, int]: ...

    def publish(
        self,
        topic: str,
        payload: bytes,
        *,
        qos: int,
        retain: bool,
    ) -> PahoPublishInfo: ...


PahoClientFactory = Callable[[yarl.URL, logging.Logger], PahoClient]


def client_cert_ssl_context(auth: TlsClientAuth) -> ssl.SSLContext:
    """Build the mutual-TLS context used by certificate-authenticated printers."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=auth.ca_pem)
    directory = tempfile.mkdtemp()
    cert_path = os.path.join(directory, "cert.pem")
    key_path = os.path.join(directory, "key.pem")
    try:
        for path, data in ((cert_path, auth.cert_pem), (key_path, auth.key_pem)):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(descriptor, data.encode())
            finally:
                os.close(descriptor)
        context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    finally:
        for path in (cert_path, key_path):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(directory)
        except OSError:
            pass
    return context


def default_paho_client(
    url: yarl.URL,
    logger: logging.Logger,
    *,
    verify_tls: bool = False,
    tls_client_auth: Optional[TlsClientAuth] = None,
) -> PahoClient:
    """Build the one Paho client retained for this endpoint's lifetime."""
    import paho.mqtt.client as paho

    client = paho.Client(
        callback_api_version=paho.CallbackAPIVersion.VERSION2,
        protocol=paho.MQTTv311,
        clean_session=True,
        reconnect_on_failure=True,
    )
    client.reconnect_delay_set(RECONNECT_MIN_DELAY, RECONNECT_MAX_DELAY)
    client.enable_logger(logger)
    if url.scheme == "mqtts":
        if tls_client_auth is not None:
            client.tls_set_context(client_cert_ssl_context(tls_client_auth))
            client.tls_insecure_set(True)
        elif verify_tls:
            client.tls_set(tls_version=ssl.PROTOCOL_TLS)
        else:
            client.tls_set(tls_version=ssl.PROTOCOL_TLS, cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
    return client


class Paho(MqttTransport):
    """Adapt one long-lived Paho client to the shared transport contract."""

    def __init__(
        self,
        url: yarl.URL,
        provider: Optional[EventLoopProvider] = None,
        *,
        client_factory: PahoClientFactory = default_paho_client,
        keepalive: int = 60,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if url.host is None:
            raise ValueError("MQTT URL has no host")
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.provider = provider or EventLoopProvider.default()
        self.events: EventBus[WireEvent] = EventBus(self.provider)
        self.client_factory = client_factory
        self.keepalive = keepalive
        self.logger = logger or logging.getLogger("wire.mqtt.paho")
        self.client: Optional[PahoClient] = None
        self.subscriptions: set[str] = set()
        self.lock = threading.Lock()
        self.started = False
        self.worker: Optional[threading.Thread] = None
        self.worker_stop: Optional[threading.Event] = None
        self.courier: Optional[Courier[WireEvent]] = None

    @property
    def connected(self) -> bool:
        with self.lock:
            client = self.client
            ready = self.started and self.state is ConnectionState.CONNECTED
        return bool(ready and client is not None and client.is_connected())

    def supervising(self) -> bool:
        with self.lock:
            worker = self.worker
            return self.started and worker is not None and worker.is_alive()

    def start(self) -> None:
        with self.lock:
            if self.started:
                return
            client = self.client_factory(self.url, self.logger)
            client.on_pre_connect = self.on_pre_connect
            client.on_connect = self.on_connect
            client.on_connect_fail = self.on_connect_fail
            client.on_message = self.on_message
            client.on_disconnect = self.on_disconnect
            if self.url.user or self.url.password:
                client.username_pw_set(self.url.user or None, self.url.password or None)

            stop = threading.Event()
            worker = threading.Thread(
                target=self._run,
                args=(client, stop),
                name=f"mqtt-{self.url.host}",
                daemon=True,
            )
            self.client = client
            self.worker_stop = stop
            self.worker = worker
            self.courier = Courier(
                sink=self.events.emit,
                is_async_sink=True,
                provider=self.provider,
                policy=OverflowPolicy.DROP_OLDEST,
                maxsize=EVENT_CAPACITY,
                lossless=self._lossless_event,
                logger=self.logger,
            )
            self.started = True
        try:
            worker.start()
        except Exception:
            with self.lock:
                courier = self.courier
                self.client = None
                self.worker_stop = None
                self.worker = None
                self.courier = None
                self.started = False
            if courier is not None:
                courier.close(drain=False)
            raise

    async def stop(self) -> None:
        with self.lock:
            if not self.started and self.worker is None:
                return
            client = self.client
            stop = self.worker_stop
            worker = self.worker
            courier = self.courier
            self.started = False
            self.client = None
            self.worker_stop = None
            self.worker = None
            self.courier = None
            self.state = ConnectionState.DISCONNECTED
        if stop is not None:
            stop.set()
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                self.logger.debug(
                    "paho %s disconnect failed", self.url.host, exc_info=True
                )
        if worker is not None and worker is not threading.current_thread():
            await run_in_thread(
                worker.join,
                WORKER_SHUTDOWN_TIMEOUT,
                thread_name="paho-stop",
            )
            if worker.is_alive():
                self.logger.error("paho %s worker did not stop", self.url.host)
        if courier is not None:
            courier.close(drain=False)

    async def send(self, message: object) -> None:
        if not isinstance(message, MqttMessage):
            raise TypeError(f"cannot send {type(message).__name__} over MQTT")
        with self.lock:
            client = self.client
            generation = self.generation
            ready = self.started and self.state is ConnectionState.CONNECTED
        if client is None or not ready or not client.is_connected():
            raise NotConnected("paho transport not connected")
        info = client.publish(
            message.topic,
            message.payload,
            qos=message.qos.value,
            retain=message.retain,
        )
        if info.rc != 0:
            error = NotConnected(f"paho publish rejected (rc={info.rc})", code=info.rc)
            self.trip(generation, error)
            raise error
        if message.qos.value == 0:
            return
        await run_in_thread(
            info.wait_for_publish,
            PUBLISH_ACK_TIMEOUT,
            thread_name="paho-publish",
        )
        if not info.is_published():
            error = TransientError(
                f"paho publish unacked after {PUBLISH_ACK_TIMEOUT:g}s"
            )
            self.trip(generation, error)
            raise error

    def subscribe(self, topic: str) -> None:
        with self.lock:
            if topic in self.subscriptions:
                return
            self.subscriptions.add(topic)
            client = self.client
            active = self.started
        if active and client is not None and client.is_connected():
            self._subscribe(client, topic)

    def unsubscribe(self, topic: str) -> None:
        with self.lock:
            if topic not in self.subscriptions:
                return
            self.subscriptions.remove(topic)
            client = self.client
            active = self.started
        if not active or client is None or not client.is_connected():
            return
        try:
            result, _mid = client.unsubscribe(topic)
        except Exception as error:
            self.trip(self.generation, TransientError.wrap(error))
            return
        if result != 0:
            self.trip(
                self.generation,
                TransientError(
                    f"paho unsubscribe failed for {topic!r} (rc={result})",
                    code=result,
                ),
            )

    def trip(self, generation: int, reason: Exception) -> None:
        error = (
            reason
            if isinstance(reason, TransportError)
            else TransientError.wrap(reason)
        )
        with self.lock:
            if not self.started or generation != self.generation:
                return
            client = self.client
        if client is None:
            return
        self._connection_failed(client, error)
        try:
            client.disconnect()
        except Exception:
            self.logger.debug("paho %s reset failed", self.url.host, exc_info=True)

    def on_pre_connect(self, client: PahoClient, userdata: object = None) -> None:
        if self._active(client):
            self._mark_connecting()

    def on_connect(
        self,
        client: PahoClient,
        userdata: object = None,
        flags: object = None,
        reason_code: PahoReasonCode = None,
        properties: object = None,
    ) -> None:
        if not self._active(client):
            return
        if connect_rejected(reason_code):
            code = paho_reason_code(reason_code)
            error_type = (
                AuthenticationError
                if code in AUTHENTICATION_FAILURE_CODES
                else TransientError
            )
            self._connection_failed(
                client,
                error_type(f"paho connection rejected: {reason_code}", code=code),
            )
            return

        with self.lock:
            if self.client is not client or not self.started:
                return
            topics = tuple(self.subscriptions)
        for topic in topics:
            if not self._subscribe(client, topic):
                return
        with self.lock:
            if (
                self.client is not client
                or not self.started
                or self.state is ConnectionState.CONNECTED
            ):
                return
            self.generation += 1
            self.state = ConnectionState.CONNECTED
            event = Connected(self.generation)
        self._post(event)

    def on_connect_fail(self, client: PahoClient, userdata: object = None) -> None:
        self._connection_failed(client, TransientError("paho connection failed"))

    def on_message(
        self,
        client: PahoClient,
        userdata: object,
        message: "MQTTMessage",
    ) -> None:
        with self.lock:
            if (
                self.client is not client
                or not self.started
                or self.state is not ConnectionState.CONNECTED
            ):
                return
            generation = self.generation
        try:
            parsed = mqtt_message_from_inbound(message)
        except Exception:
            self.logger.warning(
                "paho %s delivered an invalid message", self.url.host, exc_info=True
            )
            return
        self._post(MessageReceived(generation, parsed, parsed.qos))

    def on_disconnect(
        self,
        client: PahoClient,
        userdata: object = None,
        flags: object = None,
        reason_code: PahoReasonCode = None,
        properties: object = None,
    ) -> None:
        self._connection_failed(
            client,
            TransientError(
                f"paho disconnected: {reason_code}",
                code=paho_reason_code(reason_code),
            ),
        )

    def _run(self, client: PahoClient, stop: threading.Event) -> None:
        while not stop.is_set():
            self._mark_connecting()
            try:
                client.connect_async(
                    self.url.host,
                    self.port(),
                    keepalive=self.keepalive,
                )
                result = client.loop_forever(retry_first_connection=True)
                if stop.is_set():
                    return
                error = TransientError(f"paho network loop stopped (rc={result})")
            except Exception as exception:
                if stop.is_set():
                    return
                self.logger.exception("paho %s network loop crashed", self.url.host)
                error = TransientError.wrap(exception, "paho network loop crashed")
            self._connection_failed(client, error)
            stop.wait(WORKER_RESTART_DELAY)

    def _subscribe(self, client: PahoClient, topic: str) -> bool:
        try:
            result, _mid = client.subscribe(topic)
        except Exception as error:
            self.trip(self.generation, TransientError.wrap(error))
            return False
        if result == 0:
            return True
        self.trip(
            self.generation,
            TransientError(
                f"paho subscribe failed for {topic!r} (rc={result})",
                code=result,
            ),
        )
        return False

    def _active(self, client: PahoClient) -> bool:
        with self.lock:
            return self.client is client and self.started

    def _connection_failed(self, client: PahoClient, error: TransportError) -> None:
        with self.lock:
            if self.client is not client or not self.started:
                return
            if self.state is ConnectionState.DISCONNECTED:
                return
            self.state = ConnectionState.DISCONNECTED
            event = Disconnected(self.generation, code=error)
        self._post(event)

    def _mark_connecting(self) -> None:
        with self.lock:
            if not self.started or self.state is ConnectionState.CONNECTING:
                return
            self.state = ConnectionState.CONNECTING
            event = Connecting(self.generation)
        self._post(event)

    def _post(self, event: WireEvent) -> None:
        with self.lock:
            courier = self.courier
        if courier is not None:
            courier.post(event)

    @staticmethod
    def _lossless_event(event: WireEvent) -> bool:
        return not isinstance(event, MessageReceived) or event.lossless

    def port(self) -> int:
        if self.url.port is not None:
            return self.url.port
        return 8883 if self.url.scheme == "mqtts" else 1883


def connect_rejected(reason_code: PahoReasonCode) -> bool:
    if reason_code is None:
        return False
    if isinstance(reason_code, int):
        return reason_code != 0
    return reason_code.is_failure


def paho_reason_code(reason_code: PahoReasonCode) -> Optional[ErrorCode]:
    if reason_code is None:
        return None
    if isinstance(reason_code, int):
        return reason_code
    return reason_code.value
