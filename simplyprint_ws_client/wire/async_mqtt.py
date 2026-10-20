"""An MQTT 3.1.1 broker wire that lives entirely on the event loop.

:class:`AsyncMqtt` fills the four :class:`~simplyprint_ws_client.wire.reconnect.Reconnecting`
hooks with an :class:`~simplyprint_ws_client.wire.mqtt_stream.MqttStream`
(TCP, optional TLS) and the packets from
:mod:`~simplyprint_ws_client.wire.mqtt_codec`. Every read, write, ping and
acknowledgement happens on the provider's loop, so:

- no thread ever touches the socket, and a publish can never race the reader
  (the TLS state is one ``SSLObject``, used on one thread);
- a broker costs a task and a socket, not an OS thread, and buffers only what
  it carries, which is what lets one host run hundreds of printers;
- every outgoing packet is one ``write`` call, so packets never interleave.

The base class owns connect/consume/drop/backoff, the generation and the
lifecycle events. This class adds MQTT itself: the session handshake, the
desired-subscription set (re-asserted on every connection), QoS 1/2 delivery
and acknowledgement, and keepalive pings.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import ssl
import string
import time
from typing import Callable, Dict, Optional, Set, Tuple

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ExponentialBackoff
from simplyprint_ws_client.wire import mqtt_codec as codec
from simplyprint_ws_client.wire.errors import AuthenticationError, TransportError
from simplyprint_ws_client.wire.events import MessageReceived
from simplyprint_ws_client.wire.messages import MqttMessage, QoS
from simplyprint_ws_client.wire.mqtt_stream import MqttStream
from simplyprint_ws_client.wire.options import TlsClientAuth
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.transport import MqttTransport, TransientError
from simplyprint_ws_client.wire.tls import client_cert_ssl_context

__all__ = ["AsyncMqtt"]

#: How long a QoS 1 publish waits for the broker's PUBACK.
PUBLISH_ACK_TIMEOUT = 30.0
#: Bound on TCP connect + TLS handshake + CONNACK. Printers on a LAN answer in
#: milliseconds; an unreachable one should be retried soon, not in a minute.
OPEN_TIMEOUT = 10.0
#: How long a graceful close waits for the socket to finish closing.
CLOSE_TIMEOUT = 2.0
#: CONNACK return codes meaning "these credentials are wrong", never transient.
AUTHENTICATION_FAILURES = frozenset((4, 5))
IDENTIFIER_REJECTED = 2

def mqtt_retry_policy() -> RetryPolicy:
    """Reconnect in about 1, 2, 4, then 5 seconds (a printer rebooting is back fast)."""
    return RetryPolicy(ExponentialBackoff(base=1.0, maximum=5.0))


def mqtt_ssl_context(
    *, verify_tls: bool, tls_client_auth: Optional[TlsClientAuth]
) -> ssl.SSLContext:
    if tls_client_auth is not None:
        return client_cert_ssl_context(tls_client_auth)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if verify_tls:
        context.load_default_certs()
    else:
        # Printers present self-signed certificates; verification is opt-in.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def random_client_id() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(22))


class AsyncMqtt(MqttTransport, Reconnecting):
    """A self-healing MQTT 3.1.1 client for one broker endpoint.

    Messages are delivered to :attr:`events` listeners one at a time, in wire
    order, and reading waits for each listener (that is the back-pressure). A
    listener on the transport itself should therefore not block for long; the
    pool's leases already hand messages on through their own bounded queues.
    """

    def __init__(
        self,
        url: yarl.URL,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        keepalive: int = 60,
        verify_tls: bool = False,
        tls_client_auth: Optional[TlsClientAuth] = None,
        policy: Optional[RetryPolicy] = None,
        open_timeout: Optional[float] = OPEN_TIMEOUT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if url.host is None:
            raise ValueError("MQTT URL has no host")
        super().__init__(
            url,
            policy or mqtt_retry_policy(),
            provider,
            open_timeout=open_timeout,
            logger=logger or logging.getLogger("wire.mqtt"),
        )
        self.keepalive = keepalive
        self.verify_tls = verify_tls
        self.tls_client_auth = tls_client_auth
        #: Empty, as for any MQTT 3.1.1 clean session: the broker assigns one.
        #: A broker that refuses an empty id gets a random one from then on.
        self.client_id = ""
        #: Topics this transport keeps subscribed, across reconnects.
        self.subscriptions: Set[str] = set()

        # The live wire. All of it belongs to the provider's loop.
        self.connection: Optional[MqttStream] = None
        self._next_packet_id = 0
        self._acks: Dict[int, "asyncio.Future[None]"] = {}
        self._subscribing: Dict[int, Tuple[str, ...]] = {}
        self._unsubscribing: Set[int] = set()
        self._connack: Optional["asyncio.Future[codec.Packet]"] = None
        self._qos2_received: Set[int] = set()
        self._pinger: Optional["asyncio.Task[None]"] = None
        self._last_in = self._last_out = 0.0
        self._ping_sent: Optional[float] = None

    # -- the wire hooks --------------------------------------------------------

    async def open(self) -> None:
        secure = self.url.scheme == "mqtts"
        context = (
            mqtt_ssl_context(
                verify_tls=self.verify_tls, tls_client_auth=self.tls_client_auth
            )
            if secure
            else None
        )
        # A broker that refuses an empty client id gets a random one, within
        # this same attempt (a one-shot probe must not see it as a failure).
        for _ in range(2):
            code = await self._handshake(context, secure)
            if code == IDENTIFIER_REJECTED and not self.client_id:
                self.client_id = random_client_id()
                self._drop_stream()
                continue
            break
        if code in AUTHENTICATION_FAILURES:
            raise AuthenticationError(f"MQTT connection refused (code {code})", code=code)
        if code:
            raise TransientError(f"MQTT connection refused (code {code})", code=code)
        if self.subscriptions:
            self._subscribe_now(tuple(sorted(self.subscriptions)))
        self._pinger = asyncio.ensure_future(self._keep_alive())

    async def _handshake(self, context: Optional[ssl.SSLContext], secure: bool) -> int:
        """Connect, send CONNECT, return the CONNACK code."""
        connack: "asyncio.Future[codec.Packet]" = (
            asyncio.get_running_loop().create_future()
        )
        self._connack = connack
        try:
            self.connection = await MqttStream.open(
                self.url.host,
                self.port(),
                ssl_context=context,
                server_hostname=self.url.host if secure else None,
                on_control=self._on_control,
            )
        except (OSError, ssl.SSLError, ssl.CertificateError) as error:
            raise TransientError.wrap(error)
        self._last_in = self._last_out = time.monotonic()
        self._ping_sent = None
        self._put(
            codec.encode_connect(
                self.client_id,
                keepalive=self.keepalive,
                username=self.url.user,
                password=self.url.password,
            )
        )
        await self._drain()
        # CONNACK arrives as a control packet; a dead stream ends the read.
        reading = asyncio.ensure_future(self._read())
        try:
            await asyncio.wait({connack, reading}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            reading.cancel()
            await asyncio.gather(reading, return_exceptions=True)
            self._connack = None
        if not connack.done():
            reading.result()  # re-raises how the stream died
            raise TransientError("PUBLISH before CONNACK")
        _session, code = codec.decode_connack(connack.result().body)
        return code

    def _drop_stream(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            connection.abort()

    async def recv(self) -> Optional[MqttMessage]:
        message, _ack = await self._receive()
        return message

    async def consume_one(self) -> None:
        # Acknowledge an inbound QoS 1 message only after it was delivered, as
        # the protocol intends (the broker re-sends it if we drop first).
        message, ack = await self._receive()
        if message is not None:
            await self.events.emit(
                MessageReceived(self.generation, message, message.qos)
            )
        if ack is not None:
            self._put(ack)

    async def write(self, message: object) -> None:
        if not isinstance(message, MqttMessage):
            raise TypeError(f"cannot send {type(message).__name__} over MQTT")
        if message.qos is QoS.AT_MOST_ONCE:
            self._put(
                codec.encode_publish(
                    message.topic, message.payload, retain=message.retain
                )
            )
            await self._drain()
            return
        packet_id = self._packet_id()
        acked: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
        self._acks[packet_id] = acked
        try:
            self._put(
                codec.encode_publish(
                    message.topic,
                    message.payload,
                    qos=1,
                    retain=message.retain,
                    packet_id=packet_id,
                )
            )
            await self._drain()
            try:
                await asyncio.wait_for(asyncio.shield(acked), PUBLISH_ACK_TIMEOUT)
            except asyncio.TimeoutError:
                raise TransientError(
                    f"no PUBACK within {PUBLISH_ACK_TIMEOUT:g}s"
                ) from None
        finally:
            self._acks.pop(packet_id, None)

    async def aclose(self) -> None:
        pinger, self._pinger = self._pinger, None
        if pinger is not None:
            pinger.cancel()
        connection, self.connection = self.connection, None
        for acked in self._acks.values():
            if not acked.done():
                acked.set_exception(TransientError("MQTT connection closed"))
        self._acks.clear()
        self._subscribing.clear()
        self._unsubscribing.clear()
        self._qos2_received.clear()
        if connection is None:
            return
        try:
            connection.write(codec.DISCONNECT_PACKET)  # a goodbye, if it still can
        except Exception:  # noqa: BLE001 -- the wire is already gone
            pass
        connection.close()
        try:
            await asyncio.wait_for(connection.wait_closed(), CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            connection.abort()
        except Exception:  # noqa: BLE001 -- aclose must never raise
            self.logger.debug("mqtt %s close failed", self.url.host, exc_info=True)

    # -- subscriptions -----------------------------------------------------------

    def subscribe(self, topic: str) -> None:
        self._on_loop(self._add_subscription, topic)

    def unsubscribe(self, topic: str) -> None:
        self._on_loop(self._drop_subscription, topic)

    def _add_subscription(self, topic: str) -> None:
        if topic in self.subscriptions:
            return
        self.subscriptions.add(topic)
        if self.connection is not None:
            self._subscribe_now((topic,))

    def _drop_subscription(self, topic: str) -> None:
        if topic not in self.subscriptions:
            return
        self.subscriptions.discard(topic)
        if self.connection is not None:
            packet_id = self._packet_id()
            self._unsubscribing.add(packet_id)  # reserved until UNSUBACK
            self._put(codec.encode_unsubscribe(packet_id, (topic,)))

    def _subscribe_now(self, topics: Tuple[str, ...]) -> None:
        packet_id = self._packet_id()
        self._subscribing[packet_id] = topics
        self._put(codec.encode_subscribe(packet_id, topics))

    def _on_loop(self, fn: Callable[[str], None], topic: str) -> None:
        """Run ``fn`` on the provider's loop: the wire state is single-threaded."""
        try:
            loop: Optional[asyncio.AbstractEventLoop] = self.provider.event_loop
        except RuntimeError:
            loop = None  # no loop yet: nothing can be running on it either
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop or not loop.is_running():
            fn(topic)
        else:
            loop.call_soon_threadsafe(fn, topic)

    # -- inbound -----------------------------------------------------------------

    async def _read(self) -> codec.Packet:
        connection = self.connection
        if connection is None:
            raise TransientError("MQTT not connected")
        try:
            packet = await connection.read_packet()
        except codec.MalformedPacket as error:
            raise TransientError(f"malformed MQTT packet: {error}") from error
        except (OSError, ssl.SSLError) as error:
            raise TransientError.wrap(error)
        self._last_in = time.monotonic()
        return packet

    async def _receive(self) -> Tuple[Optional[MqttMessage], Optional[bytes]]:
        """The next packet: a message to deliver (or ``None``) and the
        acknowledgement to send once it was delivered (or ``None``)."""
        packet = await self._read()
        if packet.type == codec.PUBREL:
            # In wire order with the publishes (a queued QoS 2 re-send must
            # still be recognised as one before its exchange ends).
            packet_id = codec.decode_packet_id(packet.body)
            self._qos2_received.discard(packet_id)
            return None, codec.encode_ack(codec.PUBCOMP, packet_id)
        if packet.type != codec.PUBLISH:  # controls never queue (see _on_control)
            raise TransientError(f"unexpected MQTT packet type {packet.type}")
        return self._on_publish(packet)

    def _on_control(self, packet: codec.Packet) -> None:
        """Every packet but PUBLISH, handled as it arrives: acknowledgements
        must not wait behind a message listener (which may itself be awaiting
        one). Raising ends the connection."""
        self._last_in = time.monotonic()
        kind = packet.type
        if kind == codec.CONNACK:
            connack = self._connack
            if connack is None or connack.done():
                raise TransientError("unexpected CONNACK")
            connack.set_result(packet)
        elif kind == codec.PUBACK:
            acked = self._acks.get(codec.decode_packet_id(packet.body))
            if acked is not None and not acked.done():
                acked.set_result(None)
        elif kind == codec.SUBACK:
            packet_id, codes = codec.decode_suback(packet.body)
            topics = self._subscribing.pop(packet_id, ())
            for topic, code in zip(topics, codes):
                if code == 0x80:
                    self.logger.warning(
                        "mqtt %s refused the subscription to %s", self.url.host, topic
                    )
        elif kind == codec.UNSUBACK:
            self._unsubscribing.discard(codec.decode_packet_id(packet.body))
        elif kind == codec.PINGRESP:
            self._ping_sent = None
        elif kind not in (codec.PUBREC, codec.PUBCOMP):
            raise TransientError(f"unexpected MQTT packet type {kind}")

    def _on_publish(
        self, packet: codec.Packet
    ) -> Tuple[Optional[MqttMessage], Optional[bytes]]:
        try:
            publish = codec.decode_publish(packet.flags, packet.body)
        except codec.MalformedPacket as error:
            raise TransientError(f"malformed MQTT PUBLISH: {error}") from error
        message = MqttMessage(
            publish.topic,
            publish.payload,
            QoS.AT_LEAST_ONCE if publish.qos else QoS.AT_MOST_ONCE,
            publish.retain,
        )
        if publish.qos == 0 or publish.packet_id is None:
            return message, None
        if publish.qos == 1:
            return message, codec.encode_ack(codec.PUBACK, publish.packet_id)
        # QoS 2: deliver once, however often the broker re-sends before PUBREL.
        first = publish.packet_id not in self._qos2_received
        self._qos2_received.add(publish.packet_id)
        return (message if first else None), codec.encode_ack(
            codec.PUBREC, publish.packet_id
        )

    # -- outbound ----------------------------------------------------------------

    def _put(self, packet: bytes) -> None:
        """Queue one whole packet on the socket (never interleaves)."""
        connection = self.connection
        if connection is None or connection.is_closing():
            raise TransientError("MQTT not connected")
        try:
            connection.write(packet)
        except (OSError, ssl.SSLError) as error:
            raise TransientError.wrap(error)
        self._last_out = time.monotonic()

    async def _drain(self) -> None:
        connection = self.connection
        if connection is None:
            raise TransientError("MQTT not connected")
        try:
            await connection.drain()
        except (OSError, ssl.SSLError) as error:
            raise TransientError.wrap(error)

    def _packet_id(self) -> int:
        for _ in range(0xFFFF):
            self._next_packet_id = self._next_packet_id % 0xFFFF + 1
            if (
                self._next_packet_id not in self._acks
                and self._next_packet_id not in self._subscribing
                and self._next_packet_id not in self._unsubscribing
            ):
                return self._next_packet_id
        raise TransientError("no free MQTT packet id")

    async def _keep_alive(self) -> None:
        """Ping when either direction was idle for ``keepalive`` seconds; a
        ping not answered within another ``keepalive`` ends the connection."""
        if self.keepalive <= 0:
            return
        while True:
            now = time.monotonic()
            if self._ping_sent is not None:
                deadline = self._ping_sent + self.keepalive
                if now >= deadline:
                    self.trip(
                        self.generation,
                        TransientError(f"no PINGRESP within {self.keepalive}s"),
                    )
                    return
            else:
                deadline = min(self._last_in, self._last_out) + self.keepalive
                if now >= deadline:
                    try:
                        self._put(codec.PINGREQ_PACKET)
                    except TransportError:
                        return
                    self._ping_sent = now
                    continue
            await asyncio.sleep(max(deadline - now, 0.05))

    def port(self) -> int:
        if self.url.port is not None:
            return self.url.port
        return 8883 if self.url.scheme == "mqtts" else 1883
