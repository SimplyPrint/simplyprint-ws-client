"""Async-native MQTT transport + pool (aiomqtt) -- the proof that the transport
contract spans more than paho's threads.

paho (see :mod:`.mqtt`) is the *synchronous* family: it runs its own network
thread and we wrap it as a :class:`~.transport.Transport`. aiomqtt is the
*asynchronous* family: it lives on the event loop, so we wrap it as an
:class:`~.transport.AsyncTransport`. Both speak the exact same
:class:`~.transport.TransportEvent` surface, so a consumer that only listens
cannot tell which it got -- which is the whole point of the contract.

:class:`AsyncMqttPool` then shows pooling-as-a-capability for the async family:
one transport per endpoint, shared by every client that resolves to it,
ref-counted, torn down when the last leaves. A dedicated link is a pool of one.

Both the aiomqtt client and the transport are injectable, so the mechanics are
testable without a broker (or even aiomqtt installed); the default factories
build the real thing lazily so importing this module never drags aiomqtt in.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Dict,
    Hashable,
    NamedTuple,
    Optional,
)

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.utils.backoff import Backoff, ConstantBackoff

from simplyprint_ws_client.contrib.connection.mqtt_topics import mqtt_topic_matches
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    AsyncConnection,
    AsyncPool,
    AsyncTransport,
    ConnectSuspect,
    Connected,
    ConsumerLoop,
    Disconnected,
    MessageReceived,
    StateChanged,
    TransportEvent,
    TransportRouter,
    _AsyncLease,
)


def _mqtt_topic_of(payload: Any) -> Optional[str]:
    """The pool's topic extractor: an aiomqtt message carries ``.topic``."""
    topic = getattr(payload, "topic", None)
    return str(topic) if topic is not None else None


class MqttParams(NamedTuple):
    """Hashable identity of an MQTT broker connection -- the pool key.

    Kept independent of the paho module's params so the async family never drags
    paho into its import graph.
    """

    host: str
    port: int
    username: str = ""
    password: str = ""

    def __str__(self) -> str:
        return f"mqtts://{self.username}:<redacted>@{self.host}:{self.port}"


#: Builds the aiomqtt client (an async context manager) for one connect attempt.
#: Injectable so a test can hand in a fake; the default builds ``aiomqtt.Client``.
MqttClientFactory = Callable[[MqttParams, logging.Logger], AsyncContextManager[Any]]


def _default_client_factory(
    params: MqttParams, _logger: logging.Logger
) -> AsyncContextManager[Any]:
    """Build a real ``aiomqtt.Client`` (imported lazily, TLS to match paho)."""
    import ssl

    import aiomqtt  # lazy: importing this module must not require aiomqtt

    tls = aiomqtt.TLSParameters(cert_reqs=ssl.CERT_NONE)
    return aiomqtt.Client(
        hostname=params.host,
        port=params.port,
        username=params.username or None,
        password=params.password or None,
        tls_params=tls,
        tls_insecure=True,
    )


class AsyncMqttTransport(AsyncTransport[MqttParams]):
    """A supervised aiomqtt link, driven on the harness event loop.

    Owns the reliability story ONCE: a single supervised task connects,
    subscribes, streams inbound messages as :class:`MessageReceived`, and -- on
    any error -- backs off and reconnects, all without the caller ever seeing a
    task or a thread. The caller only ``subscribe`` s, ``await send`` s, and
    listens on :attr:`events`.
    """

    def __init__(
        self,
        params: MqttParams,
        *,
        logger: Optional[logging.Logger] = None,
        client_factory: MqttClientFactory = _default_client_factory,
        backoff: Optional[Backoff] = None,
    ) -> None:
        self.params = params
        self.events: EventBus[TransportEvent] = EventBus()
        self.state = ConnectionState.OFFLINE
        self._logger = logger or logging.getLogger("async_mqtt")
        self._client_factory = client_factory
        self._backoff = backoff or ConstantBackoff()
        self._topics: Dict[str, int] = {}
        self._client: Any = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def connected(self) -> bool:
        return self.state is ConnectionState.ONLINE

    def start(self) -> None:
        """Schedule the supervised connect/reconnect loop on the harness loop."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def subscribe(self, topic: str) -> None:
        """Register interest in ``topic`` (applied on connect / live if up)."""
        refs = self._topics.get(topic, 0)
        self._topics[topic] = refs + 1
        if refs > 0:
            return
        client = self._client
        if client is not None:
            try:
                asyncio.get_running_loop().create_task(client.subscribe(topic))
            except RuntimeError:
                pass  # no running loop; (re)subscribed on the next connect

    def unsubscribe(self, topic: str) -> None:
        refs = self._topics.get(topic, 0)
        if refs <= 0:
            return
        if refs == 1:
            self._topics.pop(topic, None)
            client = self._client
            unsubscribe = (
                getattr(client, "unsubscribe", None) if client is not None else None
            )
            if unsubscribe is not None:
                try:
                    asyncio.get_running_loop().create_task(unsubscribe(topic))
                except RuntimeError:
                    pass
            return
        self._topics[topic] = refs - 1

    async def send(self, payload: Any) -> None:
        """Publish ``(topic, data)`` to the broker; raises if the link is down."""
        client = self._client
        if client is None:
            raise ConnectionError("MQTT transport not connected")
        topic, data = payload
        await client.publish(topic, data)

    def _set_state(self, state: ConnectionState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                async with self._client_factory(self.params, self._logger) as client:
                    self._client = client
                    for topic in list(self._topics.keys()):
                        await client.subscribe(topic)
                    self._set_state(ConnectionState.ONLINE)
                    self._emit(Connected())
                    self._backoff.reset()
                    async for message in client.messages:
                        self._emit(MessageReceived(message))
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 -- supervised: any error reconnects
                self._logger.warning("MQTT link to %s dropped: %s", self.params, e)
                self._emit(ConnectSuspect(error=e))
            finally:
                self._client = None
                self._set_state(ConnectionState.OFFLINE)
                self._emit(Disconnected(reason="link down", transient=True))

            if self._stop.is_set():
                break
            await asyncio.sleep(self._backoff.delay())


#: Builds an :class:`AsyncMqttTransport` for ``params``. Injectable for tests.
MqttTransportFactory = Callable[[MqttParams], AsyncMqttTransport]


class AsyncMqttPool(AsyncPool[MqttParams, AsyncMqttTransport]):
    """Pooling-as-a-capability for the async MQTT transport.

    Hands out ONE :class:`AsyncMqttTransport` per endpoint, shared by every
    client that resolves to the same :class:`MqttParams`, ref-counted so the
    transport is started on the first acquire and stopped after the last
    release. A dedicated link is just a pool entry with a single holder.
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        transport_factory: Optional[MqttTransportFactory] = None,
        event_loop_provider: Optional[
            EventLoopProvider[asyncio.AbstractEventLoop]
        ] = None,
    ) -> None:
        self._logger = logger or logging.getLogger("async_mqtt_pool")
        self._transport_factory = transport_factory or self._build_transport
        self._provider = event_loop_provider or EventLoopProvider.default()
        self._transports: Dict[MqttParams, AsyncMqttTransport] = {}
        self._refs: Dict[MqttParams, int] = {}
        self._routers: Dict[MqttParams, TransportRouter] = {}
        self._consumer = ConsumerLoop(
            self._provider, logger=self._logger.getChild("consumer")
        )
        self._lock = asyncio.Lock()

    def _build_transport(self, params: MqttParams) -> AsyncMqttTransport:
        return AsyncMqttTransport(params, logger=self._logger.getChild(str(params)))

    async def connect(
        self, params: MqttParams, *, route: Optional[Hashable] = None
    ) -> AsyncConnection:
        """Front door: lease the shared broker socket as an :class:`AsyncConnection`.

        ``route`` is the MQTT topic this lease wants -- it both subscribes the
        shared transport and scopes which inbound messages reach this lease, so
        N printers on one broker each see only their own topic.
        """
        transport = await self.acquire(params)
        async with self._lock:
            router = self._routers.get(params)
            if router is None:
                router = TransportRouter(
                    transport, topic_of=_mqtt_topic_of, matcher=mqtt_topic_matches
                )
                router.attach()
                self._routers[params] = router
        lease: _AsyncLease = _AsyncLease(
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

    async def acquire(self, params: MqttParams) -> AsyncMqttTransport:
        async with self._lock:
            transport = self._transports.get(params)
            if transport is None:
                transport = self._transport_factory(params)
                self._transports[params] = transport
                self._refs[params] = 0
                transport.start()
            self._refs[params] += 1
            return transport

    async def release(self, params: MqttParams) -> None:
        async with self._lock:
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

    async def stop(self) -> None:
        """Tear down every pooled transport (process shutdown)."""
        async with self._lock:
            transports = list(self._transports.values())
            routers = list(self._routers.values())
            self._transports.clear()
            self._refs.clear()
            self._routers.clear()
        for router in routers:
            router.detach()
        for transport in transports:
            transport.stop()
