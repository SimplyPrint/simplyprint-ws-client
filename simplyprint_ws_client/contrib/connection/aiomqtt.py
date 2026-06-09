"""The async MQTT wire: an aiomqtt client driven by the reconnect loop.

aiomqtt is the *asynchronous* MQTT family -- one ``async with aiomqtt.Client(...)``
is exactly one connection, so it has no self-healing of its own. :class:`AioMqtt`
gives it the whole reliability story by being a
:class:`~simplyprint_ws_client.contrib.connection.reconnect.Reconnecting` transport: it
fills the four wire hooks on itself and the shared supervision loop owns connect,
reconnect, state, generation, and the lifecycle events. It is also an
:class:`~simplyprint_ws_client.contrib.connection.transport.MqttTransport`, so it carries
ref-counted topic subscriptions and routes an inbound message by its topic.

There is no wrapper around the wire: the live aiomqtt client is a plain attribute
(:attr:`AioMqtt.client`) and the four hooks call it directly. The client is built
by an injectable factory and imported lazily, so building or importing this module
needs no broker -- and not even aiomqtt installed. A test hands in a fake
async-context client exposing ``.messages``, ``.subscribe``, ``.unsubscribe`` and
``.publish``; the default factory builds a real ``aiomqtt.Client`` from the URL.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Dict,
    Optional,
)

import yarl

from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.contrib.connection.messages import MqttMessage
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
from simplyprint_ws_client.contrib.connection.transport import (
    FatalError,
    MqttTransport,
    TransientError,
)

__all__ = ["AioMqtt", "MqttClientFactory", "default_aiomqtt_client"]

#: An aiomqtt client is an async context manager (``async with`` is one
#: connection) exposing ``.messages``, ``.subscribe``/``.unsubscribe`` and
#: ``.publish``. The factory builds a fresh one per connect attempt; injectable so
#: a test hands in a fake with neither a broker nor aiomqtt installed.
MqttClientFactory = Callable[[yarl.URL, logging.Logger], AsyncContextManager[Any]]


def default_aiomqtt_client(
    url: yarl.URL, logger: logging.Logger
) -> AsyncContextManager[Any]:
    """Build a real ``aiomqtt.Client`` from ``url`` (aiomqtt imported lazily).

    A ``mqtts://`` scheme enables TLS (matching the broker the printer fleet uses);
    credentials and port come from the URL. Importing this module never triggers
    this code, so aiomqtt stays an optional dependency.
    """
    import aiomqtt  # lazy: importing this module must not require aiomqtt

    tls = url.scheme == "mqtts"
    tls_params = aiomqtt.TLSParameters(cert_reqs=ssl.CERT_NONE) if tls else None
    return aiomqtt.Client(
        hostname=url.host,
        port=url.port or (8883 if tls else 1883),
        username=url.user or None,
        password=url.password or None,
        tls_params=tls_params,
        tls_insecure=tls or None,
    )


class AioMqtt(MqttTransport, Reconnecting):
    """A self-healing aiomqtt transport -- one broker link, kept alive by the reconnect loop.

    Each connect attempt builds a fresh aiomqtt client and enters its async
    context (held as the plain attribute :attr:`client`); :meth:`recv` streams
    ``client.messages`` as inbound messages until the stream ends (which ends the
    attempt and triggers a retry). Topic interest is ref-counted: a subscription is
    asserted on the live client immediately when connected and re-asserted on every
    reconnect, and dropped only when its last holder unsubscribes. An inbound
    message routes by its topic, so the pool scopes it to the leases that
    subscribed.
    """

    def __init__(
        self,
        url: yarl.URL,
        policy: Optional[RetryPolicy] = None,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        client_factory: MqttClientFactory = default_aiomqtt_client,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            url, policy, provider, logger=logger or logging.getLogger("conn.aiomqtt")
        )
        self.client_factory = client_factory
        #: topic -> refcount across leases; re-asserted on every (re)connect.
        self.subscriptions: Dict[str, int] = {}
        #: The live aiomqtt client, or ``None`` while disconnected.
        self.client: Optional[Any] = None
        self.context: Optional[AsyncContextManager[Any]] = None
        self.stream: Optional[AsyncIterator[Any]] = None

    async def open(self) -> None:
        """Enter a fresh aiomqtt client's context and (re)assert tracked topics.

        A refused connection or bad credentials surface as the aiomqtt error the
        context raises; it is tagged onto ``Disconnected.code`` as either a
        :class:`~simplyprint_ws_client.contrib.connection.transport.FatalError` (auth) or
        a :class:`~simplyprint_ws_client.contrib.connection.transport.TransientError`.
        """
        context = self.client_factory(self.url, self.logger)
        try:
            client = await context.__aenter__()
        except Exception as error:  # noqa: BLE001 -- classify, then re-raise to retry
            raise self.classify(error) from error
        self.context = context
        self.client = client
        self.stream = None
        for topic in list(self.subscriptions):
            await client.subscribe(topic)

    async def recv(self) -> Optional[Any]:
        """Pull the next inbound message from ``client.messages``.

        Raises ``StopAsyncIteration`` (re-raised as a transient drop) when the
        stream ends -- the signal aiomqtt gives that the link went away.
        """
        if self.client is None:
            raise TransientError("no live client")
        if self.stream is None:
            self.stream = self.client.messages.__aiter__()
        try:
            return await self.stream.__anext__()
        except StopAsyncIteration as end:
            raise TransientError("message stream ended") from end

    async def write(self, message: Any) -> None:
        """Publish one outbound message (topic + payload) on the live client."""
        if self.client is None:
            raise TransientError("no live client")
        if not isinstance(message, MqttMessage):
            raise TypeError("aiomqtt transport sends MqttMessage instances")
        await self.client.publish(
            message.topic,
            message.payload,
            qos=message.qos.value,
            retain=message.retain,
        )

    async def aclose(self) -> None:
        """Exit the aiomqtt client's context. Idempotent; swallows teardown errors."""
        context = self.context
        self.context = None
        self.client = None
        self.stream = None
        if context is not None:
            try:
                await context.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 -- aclose must never break supervision
                self.logger.debug("aiomqtt %s aclose failed", self.url, exc_info=True)

    async def subscribe(self, topic: str) -> None:
        """Assert interest in ``topic`` (refcounted); apply live if connected.

        The first holder of a topic subscribes it on the live client now (and the
        topic is re-asserted on every reconnect); later holders only bump its
        refcount.
        """
        refs = self.subscriptions.get(topic, 0)
        self.subscriptions[topic] = refs + 1
        if refs == 0 and self.client is not None:
            await self.client.subscribe(topic)

    async def unsubscribe(self, topic: str) -> None:
        """Drop one holder of ``topic``; unsubscribe on the live client at the last.

        Only the final holder leaving drops the subscription from the shared socket;
        an unknown topic is a no-op.
        """
        refs = self.subscriptions.get(topic, 0)
        if refs == 0:
            return
        if refs > 1:
            self.subscriptions[topic] = refs - 1
            return
        self.subscriptions.pop(topic, None)
        if self.client is not None:
            await self.client.unsubscribe(topic)

    def route(self, message: Any) -> Optional[str]:
        """The inbound message's topic -- the key the pool routes leases by."""
        return str(message.topic)

    @staticmethod
    def classify(error: Exception) -> Exception:
        """Tag a connect failure as fatal (auth) or transient (everything else).

        The reconnect loop retries regardless; the tag only rides through to
        ``Disconnected.code`` so a consumer can tell "the broker said no" from "the
        broker is unreachable".
        """
        text = str(error).lower()
        if "auth" in text or "not authorized" in text or "password" in text:
            return FatalError(str(error))
        return TransientError(str(error))
