"""The synchronous MQTT wire: a paho-mqtt broker transport on the Courier hop.

paho runs its own network thread and its own reconnect loop, so a paho link is
already self-healing -- it never needs the
:class:`~simplyprint_ws_client.contrib.connection.reconnect.Reconnecting` supervisor.
:class:`Paho` therefore *adapts* paho rather than driving it: :meth:`start` builds
the client, wires its ``on_connect`` / ``on_message`` / ``on_disconnect`` callbacks
to the connection event vocabulary, and lets paho keep the socket alive on its own.

Those callbacks fire on paho's network thread, but every
:class:`~simplyprint_ws_client.contrib.connection.transport.Transport` consumer (the pool
fan-out, every lease) lives on one asyncio loop. So :class:`Paho` owns a single
:class:`~simplyprint_ws_client.shared.asyncio.courier.Courier`: a callback posts a
ready-to-emit event from the paho thread, the courier coalesces the wakeup, and the
event is emitted on the loop. The pool and the leases never see a foreign thread.

Subscriptions are refcounted on the transport and re-asserted on every reconnect,
since paho drops them when the socket drops. The paho client is injectable through
``client_factory`` so the whole adapter is testable with a fake -- no broker, and
paho need not even be installed. The default factory imports paho lazily, so
importing this module never drags the dependency in.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    Connecting,
    ConnectionEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.contrib.connection.messages import MqttMessage
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    FatalError,
    MqttTransport,
    NotConnected,
    TransientError,
)

__all__ = ["Paho", "PahoClientFactory", "default_paho_client"]

#: Builds (but does not connect) a paho client for a URL. Injectable for tests so
#: the adapter can be exercised without a broker or even paho installed.
PahoClientFactory = Callable[[yarl.URL, logging.Logger], Any]


def default_paho_client(url: yarl.URL, logger: logging.Logger) -> Any:
    """Build a real ``paho.mqtt.client.Client``, TLS-enabled for ``mqtts://``.

    Imported lazily so merely importing this module never requires paho. paho
    keeps the socket alive itself (``reconnect_on_failure``), which is why
    :class:`Paho` adapts it instead of supervising it.
    """
    import ssl

    import paho.mqtt.client as paho  # lazy: importing this module must not need paho

    client = paho.Client(
        callback_api_version=paho.CallbackAPIVersion.VERSION2,
        reconnect_on_failure=True,
    )
    if url.scheme == "mqtts":
        client.tls_set(tls_version=ssl.PROTOCOL_TLS, cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    return client


class Paho(MqttTransport):
    """A broker transport backed by paho-mqtt, bridged onto one asyncio loop.

    paho owns connection, reconnection, and its network thread; :class:`Paho`
    translates paho's callbacks into :class:`Connected` / :class:`Disconnected` /
    :class:`MessageReceived` events and couriers them from the paho thread onto the
    provider loop, where the pool and leases consume them. ``generation`` advances
    once per successful ``on_connect`` so a consumer can tell one live link from the
    next; subscriptions are refcounted and re-asserted on each reconnect.
    """

    def __init__(
        self,
        url: yarl.URL,
        *,
        provider: Optional[EventLoopProvider] = None,
        client_factory: PahoClientFactory = default_paho_client,
        keepalive: int = 60,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[ConnectionEvent] = EventBus()
        self.provider = provider or EventLoopProvider.default()
        self.client_factory = client_factory
        self.keepalive = keepalive
        self.logger = logger or logging.getLogger("conn.paho")
        #: The live paho client, or ``None`` while stopped.
        self.client: Any = None
        #: topic -> lease refcount; re-asserted on every (re)connect.
        self.subscriptions: Dict[str, int] = {}
        #: Guards the client handle and the subscription table against the paho
        #: network thread racing the loop thread.
        self.lock = threading.Lock()
        self.started = False
        #: Carries paho-thread events onto the loop; created on :meth:`start`.
        self.courier: Optional[Courier[ConnectionEvent]] = None

    @property
    def connected(self) -> bool:
        client = self.client
        return bool(client is not None and client.is_connected())

    def start(self) -> None:
        """Build the paho client, wire its callbacks, and begin connecting.

        Idempotent and fire-and-forget: it returns at once and reports readiness
        through the :class:`Connected` event. paho keeps the link up from here.
        """
        if self.started:
            return
        self.started = True
        self.courier = Courier(
            sink=self.emit,
            is_async_sink=True,
            provider=self.provider,
            policy=OverflowPolicy.UNBOUNDED,
        )

        client = self.client_factory(self.url, self.logger)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.on_disconnect = self.on_disconnect
        if self.url.user or self.url.password:
            client.username_pw_set(self.url.user or None, self.url.password or None)
        with self.lock:
            self.client = client

        self.state = ConnectionState.CONNECTING
        self.post(Connecting(self.generation))
        client.connect_async(self.url.host, self.port(), keepalive=self.keepalive)
        client.loop_start()

    async def stop(self) -> None:
        """Tear the paho client down and close the courier. Idempotent."""
        self.started = False
        with self.lock:
            client = self.client
            self.client = None
        if client is not None:
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001 -- stop must never raise
                self.logger.debug("paho %s disconnect failed", self.url, exc_info=True)
            try:
                client.loop_stop()
            except Exception:  # noqa: BLE001
                self.logger.debug("paho %s loop_stop failed", self.url, exc_info=True)
        if self.courier is not None:
            self.courier.close(drain=False)
            self.courier = None
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: Any) -> None:
        """Publish ``message`` on the live socket (raises if the link is down).

        ``message`` is wire-shaped (a topic + payload, with a ``qos`` enum and a
        ``retain`` flag): the MQTT front door's ``MqttMessage``. For a message whose
        QoS requires an acknowledgement, paho's publish info is awaited until the
        broker acks before returning.
        """
        client = self.client
        if client is None or not client.is_connected():
            raise NotConnected("paho transport not connected")
        if not isinstance(message, MqttMessage):
            raise TypeError("paho transport sends MqttMessage instances")
        info = client.publish(
            message.topic,
            message.payload,
            qos=message.qos.value,
            retain=message.retain,
        )
        if info.rc != 0:
            raise NotConnected(f"paho publish rejected (rc={info.rc})")
        if message.qos.value > 0:
            await self.provider.event_loop.run_in_executor(None, info.wait_for_publish)

    async def subscribe(self, topic: str) -> None:
        """Assert a subscription for ``topic``, refcounted across leases.

        Only the first lease for a topic hits the broker; the count is re-asserted
        on every reconnect (paho forgets subscriptions when the socket drops).
        """
        with self.lock:
            already = self.subscriptions.get(topic, 0)
            self.subscriptions[topic] = already + 1
            client = self.client
        if already == 0 and client is not None and client.is_connected():
            client.subscribe(topic)

    async def unsubscribe(self, topic: str) -> None:
        """Drop one lease's hold on ``topic``; unsubscribe at the broker on the last."""
        with self.lock:
            remaining = self.subscriptions.get(topic, 0) - 1
            if remaining > 0:
                self.subscriptions[topic] = remaining
                return
            self.subscriptions.pop(topic, None)
            client = self.client
        if client is not None and client.is_connected():
            client.unsubscribe(topic)

    def route(self, message: Any) -> Optional[str]:
        """The inbound message's topic -- the key the pool routes leases by."""
        return str(message.topic)

    def on_connect(
        self,
        client: Any,
        userdata: Any = None,
        flags: Any = None,
        reason_code: Any = None,
        properties: Any = None,
    ) -> None:
        """paho established (or re-established) the link: re-assert subscriptions,
        bump the generation, and announce :class:`Connected`."""
        if connect_rejected(reason_code):
            self.logger.warning("paho %s rejected: %s", self.url, reason_code)
            self.state = ConnectionState.DISCONNECTED
            self.post(Disconnected(self.generation, code=FatalError(str(reason_code))))
            return
        with self.lock:
            topics = list(self.subscriptions)
        for topic in topics:
            client.subscribe(topic)
        self.generation += 1
        self.state = ConnectionState.CONNECTED
        self.post(Connected(self.generation))

    def on_message(self, client: Any, userdata: Any, message: Any, *extra: Any) -> None:
        """An inbound broker message -- forwarded wire-shaped (paho ``MQTTMessage``)."""
        self.post(MessageReceived(self.generation, message))

    def on_disconnect(
        self,
        client: Any,
        userdata: Any = None,
        flags: Any = None,
        reason_code: Any = None,
        properties: Any = None,
    ) -> None:
        """The socket dropped. paho will reconnect on its own; we surface it as a
        transient :class:`Disconnected` and wait for the next ``on_connect``."""
        self.state = ConnectionState.DISCONNECTED
        self.post(Disconnected(self.generation, code=TransientError(str(reason_code))))

    def post(self, event: ConnectionEvent) -> None:
        """Hand an event to the courier for delivery on the loop.

        Called from paho's network thread (and, for the opening ``Connecting``,
        from the caller's thread). If the courier is gone (stopped) the event is
        dropped -- there is no one left to deliver to.
        """
        courier = self.courier
        if courier is not None:
            courier.post(event)

    async def emit(self, event: ConnectionEvent) -> None:
        """The courier sink: re-publish one event on the loop's event bus."""
        await self.events.emit(event)

    def port(self) -> int:
        """The broker port from the URL, defaulting by scheme (8883 TLS / 1883)."""
        if self.url.port is not None:
            return self.url.port
        return 8883 if self.url.scheme == "mqtts" else 1883


def connect_rejected(reason_code: Any) -> bool:
    """Whether a CONNACK reason code means the broker refused the connection.

    paho's VERSION2 reason code exposes ``is_failure``; older shapes are an int
    where non-zero is a refusal. ``None`` (a clean success) is never a rejection.
    """
    if reason_code is None:
        return False
    try:
        return bool(reason_code.is_failure)
    except AttributeError:
        pass
    try:
        return int(reason_code) != 0
    except (TypeError, ValueError):
        return False
