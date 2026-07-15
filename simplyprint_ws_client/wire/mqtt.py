"""The MQTT front door: an explicit pool -> a self-healing lease.

This is the protocol-specific face a printer client meets. It owns the wire
message shape (:class:`MqttMessage`), the broker-endpoint identity it pools by
(:class:`MqttBroker`), and :func:`connect`, which parses a broker URL and
shares one socket per endpoint through its caller-owned
:class:`~simplyprint_ws_client.wire.pool.Pool`,
and hands back a lease already carrying the URL's initial subscriptions.

The concrete broker wire is :class:`~simplyprint_ws_client.wire.paho.Paho`: its
network thread owns reconnects and events are couriered onto the client loop.
The dependency is imported lazily, so importing this module does not require
``paho-mqtt`` to be installed.

Everything beyond ``url``/``pool`` is carried by one
:class:`~simplyprint_ws_client.wire.options.ConnectionOptions`.
"""

from __future__ import annotations

import logging

from typing import List, NamedTuple, Optional, Union

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire.messages import MqttMessage
from simplyprint_ws_client.wire.paho import Paho, default_paho_client
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.transport import MqttTransport
from simplyprint_ws_client.wire.lease import MqttLease
from simplyprint_ws_client.wire.options import (
    ConnectionOptions,
    TlsClientAuth,
    WireKeepalive,
)
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.pools import PoolRegistry

__all__ = [
    "MqttMessage",
    "MqttBroker",
    "MqttLease",
    "connect",
    "pool_for",
]

#: yarl knows default ports for ws/wss but not mqtt/mqtts; supply them.
DEFAULT_PORTS = {"mqtt": 1883, "mqtts": 8883}


class MqttBroker(NamedTuple):
    """The hashable identity of a broker endpoint -- the key the pool shares by.

    Two URLs that resolve to the same host/port/credentials ride one socket. The
    string form redacts the password so it is safe to log.
    """

    host: str
    port: int
    username: str = ""
    password: str = ""

    @classmethod
    def from_url(cls, url: yarl.URL) -> "MqttBroker":
        """Parse a broker endpoint out of an ``mqtt(s)://`` URL.

        Raises :class:`ValueError` if the URL carries no host.
        """
        if not url.host:
            raise ValueError(f"mqtt.connect: no host in {url!s}")
        port = url.port or DEFAULT_PORTS.get(url.scheme, DEFAULT_PORTS["mqtt"])
        return cls(url.host, port, url.user or "", url.password or "")

    def __str__(self) -> str:
        return f"mqtt://{self.username}:<redacted>@{self.host}:{self.port}"

    # NamedTuple's auto-repr would print the password; redact it everywhere.
    __repr__ = __str__


class MqttConnectParams(NamedTuple):
    """What one ``connect`` call carries into the pool.

    The pool shares transports by ``broker`` alone; ``retry``, ``verify_tls``,
    ``tls_client_auth`` and ``logger`` configure the transport the *first* lease
    on a broker builds (later leases share that socket, so per-lease values
    cannot apply).
    """

    broker: MqttBroker
    #: ``None`` preserves paho's native 1..5 second reconnect window. An
    #: explicit policy is mapped onto paho's reconnect bounds by the factory.
    retry: Optional[RetryPolicy]
    verify_tls: bool = False
    tls_client_auth: Optional[TlsClientAuth] = None
    logger: Optional["logging.Logger"] = None


def initial_topics(url: yarl.URL) -> List[str]:
    """The ``?topic=`` subscriptions a URL asks for, in order (may be empty)."""
    return list(url.query.getall("topic", []))


def pool_for(
    registry: PoolRegistry[MqttTransport],
    provider: Optional[EventLoopProvider] = None,
    wire_keepalive: Optional[WireKeepalive] = None,
) -> Pool[MqttTransport]:
    """Return this owner's Paho pool for a loop/keepalive identity.

    The pool shares one wire per :class:`MqttBroker` and hands out
    :class:`MqttLease` leases backed by the sync network-thread
    :class:`~simplyprint_ws_client.wire.paho.Paho`.
    Per-connect retry/TLS settings ride in the
    :class:`MqttConnectParams` each ``connect`` passes, never in this closure -
    the pool is cached, so a closure would freeze the first caller's options.
    """
    mqtt_keepalive = _mqtt_keepalive_seconds(wire_keepalive)

    def make_transport(url: yarl.URL, params: object) -> MqttTransport:
        if isinstance(params, MqttConnectParams):
            retry, verify_tls, logger = params.retry, params.verify_tls, params.logger
            tls_client_auth = params.tls_client_auth
        else:
            retry, verify_tls, logger = None, False, None
            tls_client_auth = None
        return Paho(
            url,
            provider=provider,
            keepalive=mqtt_keepalive or 60,
            connect_failure_limit=(
                retry.max_attempts if retry is not None and retry.max_attempts else 3
            ),
            client_factory=lambda u, logger: default_paho_client(
                u,
                logger,
                verify_tls=verify_tls,
                tls_client_auth=tls_client_auth,
                retry=retry,
            ),
            logger=logger,
        )

    def endpoint_key(url: yarl.URL, params: object) -> MqttBroker:
        if isinstance(params, MqttConnectParams):
            return params.broker
        if isinstance(params, MqttBroker):
            return params
        return MqttBroker.from_url(url)

    def make_pool() -> Pool[MqttTransport]:
        return Pool(
            build=make_transport,
            key=endpoint_key,
            route=mqtt_message_route,
            lease_class=MqttLease,
            provider=provider,
        )

    return registry.get(provider, wire_keepalive, make_pool)


def connect(
    url: Union[str, yarl.URL],
    *,
    pool: Pool[MqttTransport],
    options: Optional[ConnectionOptions] = None,
) -> MqttLease:
    """Lease a pooled, self-healing MQTT connection to ``url``.

    Synchronous and fire-and-forget: it parses the broker out of ``url``, leases
    the shared transport from the caller-owned :class:`Pool`, applies the URL's
    ``?topic=`` subscriptions, and returns at once. Readiness/failure arrive as
    events on the lease's ``event_bus``; ``await conn.ready()`` waits for the
    first connect.

    Retry policy, loop provider, and wire/app keepalive ride in ``options``.
    Raises :class:`ValueError` on a URL with no host.
    """
    url = yarl.URL(url) if isinstance(url, str) else url
    options = options or ConnectionOptions()
    broker = MqttBroker.from_url(url)
    lease = pool.connect(
        url,
        MqttConnectParams(
            broker=broker,
            retry=options.retry,
            verify_tls=options.verify_tls,
            tls_client_auth=options.tls_client_auth,
            logger=options.logger,
        ),
    )
    if not isinstance(lease, MqttLease):
        raise TypeError(
            f"mqtt.connect needs a pool handing out MqttLease, got {type(lease).__name__}"
        )

    # Apply the URL's initial subscriptions. ``subscribe_soon`` records the
    # interest on the lease synchronously first, so routing is correct the
    # instant the link comes up (and re-asserted on every (re)connect), then
    # applies the wire subscribe as a task on the transport's loop.
    for topic in initial_topics(url):
        lease.subscribe_soon(topic)

    if options.app_keepalive is not None:
        lease.keepalive(options.app_keepalive)

    return lease


def _mqtt_keepalive_seconds(wire_keepalive: Optional[WireKeepalive]) -> Optional[int]:
    if wire_keepalive is None or wire_keepalive.interval is None:
        return None
    return int(wire_keepalive.interval)


def mqtt_message_route(message: MqttMessage) -> str:
    """Route an inbound MQTT message by its topic."""
    return message.topic
