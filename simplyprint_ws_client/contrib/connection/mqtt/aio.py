"""Async-native MQTT transport + pool (aiomqtt) -- the proof that the reconnection
engine + transport contract span more than paho's threads.

paho (see :mod:`.sync`) is the *synchronous* family: it runs its own network thread
and self-heals. aiomqtt is the *asynchronous* family: one ``async with`` client is a
single connection, so it is wrapped as a :class:`~..reconnect.Link` and made
self-healing by the shared :class:`~..reconnect.ReconnectingTransport` wrapper. Both speak the
exact same :class:`TransportEvent` surface, so a consumer that only listens cannot
tell which it got -- the whole point of the contract.

:class:`AsyncMqttPool` then shows pooling-as-a-capability for the async family. The
aiomqtt client is injectable and imported lazily, so importing this module needs no
broker (or even aiomqtt installed).
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
    Optional,
)

from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.utils.backoff import Backoff

from simplyprint_ws_client.contrib.connection.mqtt.common import (
    MqttParams,
    mqtt_topic_matches,
    topic_of,
)
from simplyprint_ws_client.contrib.connection.pool import AsyncTransportPool
from simplyprint_ws_client.contrib.connection.reconnect import (
    Link,
    ReconnectingTransport,
)
from simplyprint_ws_client.contrib.connection.transport import AsyncLease


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


class _MqttLink(Link):
    """Adapts one aiomqtt client (an ``async with`` session + ``.messages`` stream)
    to the neutral :class:`Link` -- one connection attempt's worth of broker link."""

    def __init__(
        self, client_cm: AsyncContextManager[Any], topics: Dict[str, int]
    ) -> None:
        self._cm = client_cm
        self._topics = topics  # shared with the transport (by reference)
        self._client: Any = None
        self._stream: Any = None

    async def open(self) -> None:
        self._client = await self._cm.__aenter__()

    async def on_connected(self) -> None:
        for topic in list(self._topics.keys()):  # (re)assert subscriptions on connect
            await self._client.subscribe(topic)

    async def recv(self) -> Optional[Any]:
        if self._stream is None:
            self._stream = self._client.messages.__aiter__()
        return await self._stream.__anext__()  # raises when the stream ends -> drop

    async def send(self, payload: Any) -> None:
        topic, data = payload
        await self._client.publish(topic, data)

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._stream = None
        if client is not None:
            await self._cm.__aexit__(None, None, None)

    @property
    def is_open(self) -> bool:
        return self._client is not None

    def subscribe_live(self, topic: str) -> None:
        client = self._client
        if client is not None:
            try:
                asyncio.get_running_loop().create_task(client.subscribe(topic))
            except RuntimeError:
                pass  # no running loop; (re)subscribed on the next connect

    def unsubscribe_live(self, topic: str) -> None:
        client = self._client
        if client is not None:
            try:
                asyncio.get_running_loop().create_task(client.unsubscribe(topic))
            except RuntimeError:
                pass


class AsyncMqttTransport(ReconnectingTransport[MqttParams]):
    """A self-healing aiomqtt transport: a :class:`ReconnectingTransport` driving a fresh
    aiomqtt :class:`_MqttLink` per attempt. Topic interest is ref-counted and
    re-asserted on every (re)connect; a live ``subscribe`` is applied immediately."""

    def __init__(
        self,
        params: MqttParams,
        *,
        logger: Optional[logging.Logger] = None,
        client_factory: MqttClientFactory = _default_client_factory,
        backoff: Optional[Backoff] = None,
    ) -> None:
        self._topics: Dict[str, int] = {}
        log = logger or logging.getLogger("mqtt.aio")

        def link_factory() -> Link:
            return _MqttLink(client_factory(params, log), self._topics)

        super().__init__(params, link_factory, logger=log, backoff=backoff)

    def subscribe(self, topic: str) -> None:
        """Register interest in ``topic`` (applied live if up, else on next connect)."""
        refs = self._topics.get(topic, 0)
        self._topics[topic] = refs + 1
        if refs > 0:
            return
        link = self._link
        if link is not None:
            link.subscribe_live(topic)

    def unsubscribe(self, topic: str) -> None:
        refs = self._topics.get(topic, 0)
        if refs <= 0:
            return
        if refs == 1:
            self._topics.pop(topic, None)
            link = self._link
            if link is not None:
                link.unsubscribe_live(topic)
            return
        self._topics[topic] = refs - 1


#: Builds an :class:`AsyncMqttTransport` for ``params``. Injectable for tests.
MqttTransportFactory = Callable[[MqttParams], AsyncMqttTransport]


class AsyncMqttPool(AsyncTransportPool[MqttParams, AsyncMqttTransport]):
    """Pooling-as-a-capability for the async MQTT transport.

    Hands out ONE :class:`AsyncMqttTransport` per endpoint, shared by every client
    that resolves to the same :class:`MqttParams`, ref-counted so the transport is
    started on the first lease and stopped after the last closes. A dedicated link
    is just a pool entry with a single holder.
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
        super().__init__(
            logger=logger or logging.getLogger("mqtt.aio.pool"),
            transport_factory=transport_factory,
            event_loop_provider=event_loop_provider,
        )

    def _build_transport(self, params: MqttParams) -> AsyncMqttTransport:
        return AsyncMqttTransport(params, logger=self._logger.getChild(str(params)))

    def _lease_filter(self):
        # A pooled broker is topic-routed: each lease keeps only its own topic.
        return (topic_of, mqtt_topic_matches)

    def _after_connect(self, lease: AsyncLease, route: Optional[Hashable]) -> None:
        """Subscribe the shared broker socket to the lease's route -- it both
        subscribes the transport and scopes which inbound messages reach this lease,
        so N printers on one broker each see only their own topic."""
        if route is not None:
            lease.subscribe(str(route))
