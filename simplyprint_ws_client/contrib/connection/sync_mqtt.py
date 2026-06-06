"""Synchronous MQTT transport + pool (paho-mqtt) -- the sync family on the Courier.

paho drives its own network thread, so it is the *synchronous* :class:`Transport`
family (vs. the aiomqtt :class:`AsyncTransport` in :mod:`.async_mqtt`). Both speak
the same :class:`TransportEvent` surface; the difference is where the events are
born. aiomqtt emits on the consumer loop (zero hops). paho emits on its network
thread -- so :class:`MqttPool` wires a :class:`Courier` between the transport's
event bus and the per-lease router: each event is posted to the courier on the
paho thread (a deque append + at most one ``call_soon_threadsafe``) and the router
fans it to leases *on the consumer loop*.

That replaces the old path -- paho thread -> ``queue.Queue`` -> a dedicated
dispatch thread -> ``run_coroutine_threadsafe`` (a ``Future`` per event) -> loop
-- with one coalesced hop, no extra thread, and no per-event future.

The paho client is injectable, so the mechanics are testable without a broker (or
even paho installed); the default factory builds the real client lazily, so
importing this module never drags paho in.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Hashable, Optional

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.async_mqtt import MqttParams
from simplyprint_ws_client.contrib.connection.manager import PooledConnectionManager
from simplyprint_ws_client.contrib.connection.mqtt_topics import mqtt_topic_matches
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    Connected,
    Connection,
    ConnectSuspect,
    ConsumerLoop,
    Disconnected,
    MessageReceived,
    Pool,
    StateChanged,
    Transport,
    TransportEvent,
    TransportRouter,
    _SyncLease,
)

#: Builds (but does not connect) a paho client for ``params``. Injectable for tests.
PahoClientFactory = Callable[[MqttParams, logging.Logger], Any]


def _mqtt_topic_of(payload: Any) -> Optional[str]:
    """The pool's topic extractor: a paho ``MQTTMessage`` carries ``.topic``."""
    topic = getattr(payload, "topic", None)
    return str(topic) if topic is not None else None


def _default_client_factory(params: MqttParams, _logger: logging.Logger) -> Any:
    """Build a real ``paho.mqtt.client.Client`` (imported lazily, TLS like before)."""
    import ssl

    import paho.mqtt.client as mqtt  # lazy: importing this module must not need paho

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        reconnect_on_failure=True,
    )
    client.tls_set(tls_version=ssl.PROTOCOL_TLS, cert_reqs=ssl.CERT_NONE)
    client.tls_insecure_set(True)
    client.reconnect_delay_set(min_delay=1, max_delay=5)
    return client


def _connect_rejected(reason_code: Any) -> bool:
    """Whether a CONNACK reason code indicates rejection (auth/refused)."""
    if reason_code is None:
        return False
    is_failure = getattr(reason_code, "is_failure", None)
    if is_failure is not None:
        return bool(is_failure)
    try:
        return int(reason_code) != 0
    except (TypeError, ValueError):
        return False


class PahoMqttTransport(Transport[MqttParams]):
    """A supervised paho link. paho owns the reconnect loop and the network
    thread; we translate its callbacks into :class:`TransportEvent` s.

    The caller only ``send`` s, ``subscribe`` s, and listens on :attr:`events`.
    The events are emitted on paho's thread -- crossing onto the consumer loop is
    the pool's courier's job, not the transport's.
    """

    def __init__(
        self,
        params: MqttParams,
        *,
        logger: Optional[logging.Logger] = None,
        client_factory: PahoClientFactory = _default_client_factory,
        keepalive: int = 60,
    ) -> None:
        self.params = params
        self.events: EventBus[TransportEvent] = EventBus()
        self.state = ConnectionState.OFFLINE
        self._logger = logger or logging.getLogger("paho_mqtt")
        self._client_factory = client_factory
        self._keepalive = keepalive
        self._client: Any = None
        self._topics: Dict[str, int] = {}
        self._topics_lock = threading.Lock()
        self._started = False

    @property
    def connected(self) -> bool:
        client = self._client
        return bool(client is not None and client.is_connected())

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        client = self._client_factory(self.params, self._logger)
        self._client = client
        client.on_connect = self._on_connect
        client.on_connect_fail = self._on_connect_fail
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect
        client.username_pw_set(
            self.params.username or None, self.params.password or None
        )
        client.connect_async(
            self.params.host, self.params.port, keepalive=self._keepalive
        )
        client.loop_start()

    def stop(self) -> None:
        self._started = False
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            try:
                client.loop_stop()
            except Exception:  # noqa: BLE001
                pass
        self._set_state(ConnectionState.OFFLINE)

    def subscribe(self, topic: str) -> None:
        with self._topics_lock:
            refs = self._topics.get(topic, 0)
            self._topics[topic] = refs + 1
        if refs > 0:
            return
        client = self._client
        if client is not None and client.is_connected():
            client.subscribe(topic)

    def unsubscribe(self, topic: str) -> None:
        with self._topics_lock:
            refs = self._topics.get(topic, 0)
            if refs <= 0:
                return
            if refs == 1:
                self._topics.pop(topic, None)
            else:
                self._topics[topic] = refs - 1
                return
        client = self._client
        unsubscribe = (
            getattr(client, "unsubscribe", None) if client is not None else None
        )
        if unsubscribe is not None and client.is_connected():
            unsubscribe(topic)

    def send(self, payload: Any) -> bool:
        client = self._client
        if client is None or not client.is_connected():
            return False
        topic, data = payload
        info = client.publish(topic, data)
        return getattr(info, "rc", 0) == 0

    def _set_state(self, state: ConnectionState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    def _on_connect(
        self, client, userdata, flags=None, reason_code=None, *args, **kwargs
    ):  # noqa: ANN001
        if _connect_rejected(reason_code):
            self._logger.warning(
                "Connection rejected by %s: %s", self.params, reason_code
            )
            self._emit(ConnectSuspect())
            return
        live = self._client
        if live is not None:
            with self._topics_lock:
                topics = list(self._topics.keys())
            for topic in topics:
                live.subscribe(topic)  # (re)assert subscriptions on (re)connect
        self._set_state(ConnectionState.ONLINE)
        self._emit(Connected())

    def _on_connect_fail(self, client, userdata, *args, **kwargs):  # noqa: ANN001
        self._emit(ConnectSuspect())

    def _on_message(self, client, userdata, message, *args, **kwargs):  # noqa: ANN001
        self._emit(MessageReceived(message))

    def _on_disconnect(self, client, userdata, *args, **kwargs):  # noqa: ANN001
        # paho auto-reconnects (reconnect_on_failure); surface as transient.
        self._set_state(ConnectionState.OFFLINE)
        self._emit(Disconnected(reason="MQTT disconnect", transient=True))


#: Builds a :class:`PahoMqttTransport` for ``params``. Injectable for tests.
MqttTransportFactory = Callable[[MqttParams], PahoMqttTransport]


class MqttPool(Pool[MqttParams, PahoMqttTransport]):
    """Pooling for the sync paho transport -- the sync sibling of ``AsyncMqttPool``.

    One transport per broker endpoint, ref-counted, plus the courier that carries
    its paho-thread events onto the consumer loop. :meth:`connect` hands back a
    :class:`Connection` lease scoped to a topic.
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[MqttTransportFactory] = None,
        event_loop_provider: Optional[EventLoopProvider] = None,
        message_overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        message_maxsize: int = 1024,
        lifecycle_overflow: OverflowPolicy = OverflowPolicy.UNBOUNDED,
        lifecycle_maxsize: int = 1024,
    ) -> None:
        self._logger = logger or logging.getLogger("mqtt_pool")
        self._transport_factory = transport_factory or self._build_transport
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._consumer = ConsumerLoop(
            self._provider, logger=self._logger.getChild("consumer")
        )
        self._message_overflow = message_overflow
        self._message_maxsize = message_maxsize
        self._lifecycle_overflow = lifecycle_overflow
        self._lifecycle_maxsize = lifecycle_maxsize
        self._transports: Dict[MqttParams, PahoMqttTransport] = {}
        self._refs: Dict[MqttParams, int] = {}
        self._routers: Dict[MqttParams, TransportRouter] = {}
        self._lock = threading.Lock()

    def _build_transport(self, params: MqttParams) -> PahoMqttTransport:
        return PahoMqttTransport(params, logger=self._logger.getChild(str(params)))

    def submit_to_consumer(
        self,
        coro_factory: Callable[[], Any],
        *,
        coalesce_key: Optional[Hashable] = None,
    ) -> None:
        """Run async work on the consumer loop from a transport thread."""
        self._consumer.submit(coro_factory, coalesce_key=coalesce_key)

    def call_to_consumer(self, fn: Callable[[], None]) -> None:
        self._consumer.call(fn)

    def connect(
        self, params: MqttParams, *, route: Optional[Hashable] = None
    ) -> Connection:
        transport = self.acquire(params)
        with self._lock:
            router = self._routers.get(params)
            if router is None:
                router = TransportRouter(
                    transport, topic_of=_mqtt_topic_of, matcher=mqtt_topic_matches
                )
                message_courier: Courier = Courier(
                    sink=router.dispatch,
                    provider=self._provider,
                    policy=self._message_overflow,
                    maxsize=self._message_maxsize,
                )
                lifecycle_courier: Courier = Courier(
                    sink=router.dispatch,
                    provider=self._provider,
                    policy=self._lifecycle_overflow,
                    maxsize=self._lifecycle_maxsize,
                )
                router.attach(
                    message_courier=message_courier,
                    lifecycle_courier=lifecycle_courier,
                )
                self._routers[params] = router
        lease = _SyncLease(
            pool=self,
            params=params,
            transport=transport,
            router=router,
            consumer=self._consumer,
            route=route,
        )
        router.add(lease)
        if route is not None:
            lease.subscribe(str(route))
        return lease

    def acquire(self, params: MqttParams) -> PahoMqttTransport:
        with self._lock:
            transport = self._transports.get(params)
            if transport is None:
                transport = self._transport_factory(params)
                self._transports[params] = transport
                self._refs[params] = 0
                transport.start()
            self._refs[params] += 1
            return transport

    def release(self, params: MqttParams) -> None:
        transport = None
        router = None
        with self._lock:
            if params not in self._refs:
                return
            self._refs[params] -= 1
            if self._refs[params] <= 0:
                self._refs.pop(params, None)
                transport = self._transports.pop(params, None)
                router = self._routers.pop(params, None)
        if router is not None:
            router.detach()
        if transport is not None:
            transport.stop()

    def stop(self) -> None:
        with self._lock:
            transports = list(self._transports.values())
            routers = list(self._routers.values())
            self._transports.clear()
            self._refs.clear()
            self._routers.clear()
        for router in routers:
            router.detach()
        for transport in transports:
            transport.stop()


class PooledMqttConnectionManager(PooledConnectionManager):
    """A :class:`PooledConnectionManager` backed by the sync paho :class:`MqttPool`.

    The base for brand MQTT managers: a subclass sets the event types,
    ``params_factory`` and the keepalive command; the wire + dispatch (paho ->
    Courier -> loop) and the topic-routed lease come from :class:`MqttPool`.
    """

    def __init__(
        self,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        logger: Optional[logging.Logger] = None,
        message_overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        message_maxsize: int = 1024,
        lifecycle_overflow: OverflowPolicy = OverflowPolicy.UNBOUNDED,
        lifecycle_maxsize: int = 1024,
        **kwargs: object,
    ) -> None:
        # Set before super().__init__ -> _make_pool() reads it.
        self._event_loop_provider = event_loop_provider
        self._pool_logger = logger
        self._message_overflow = message_overflow
        self._message_maxsize = message_maxsize
        self._lifecycle_overflow = lifecycle_overflow
        self._lifecycle_maxsize = lifecycle_maxsize
        super().__init__(**kwargs)

    def _make_pool(self) -> MqttPool:
        return MqttPool(
            event_loop_provider=self._event_loop_provider,
            logger=self._pool_logger,
            message_overflow=self._message_overflow,
            message_maxsize=self._message_maxsize,
            lifecycle_overflow=self._lifecycle_overflow,
            lifecycle_maxsize=self._lifecycle_maxsize,
        )
