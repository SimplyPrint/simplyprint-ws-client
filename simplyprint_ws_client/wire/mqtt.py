"""The MQTT front door: ``mqtt.connect(url)`` -> a pooled, self-healing lease.

This is the protocol-specific face a printer client meets. It owns the wire
message shape (:class:`MqttMessage`), the broker-endpoint identity it pools by
(:class:`MqttBroker`), and the one-liner :func:`connect` that parses a broker URL,
shares one socket per endpoint through a :class:`~simplyprint_ws_client.wire.pool.Pool`,
and hands back a lease already carrying the URL's initial subscriptions.

The two concrete broker wires live in their own modules and are imported here, not
re-implemented: the sync :class:`~simplyprint_ws_client.wire.paho.Paho`
(its own network thread, self-healing, events couriered onto the loop -- the
default, and what production runs) and the async
:class:`~simplyprint_ws_client.wire.aiomqtt.AioMqtt` (one ``async with``
client per attempt, on the shared reconnect loop). Both speak the exact
:class:`~simplyprint_ws_client.wire.transport.MqttTransport` contract, so a
consumer that only listens cannot tell which it got, and both import their wire
library lazily so importing this module needs neither installed.

Everything beyond ``url``/``impl``/``pool`` is carried by one
:class:`~simplyprint_ws_client.wire.options.ConnectionOptions`.
"""

from __future__ import annotations

import logging

from typing import List, Literal, NamedTuple, Optional, Tuple, Union

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire.aiomqtt import (
    AioMqtt,
    default_aiomqtt_client,
)
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
from simplyprint_ws_client.wire.pools import DefaultPools

__all__ = [
    "MqttMessage",
    "MqttBroker",
    "MqttImpl",
    "MqttLease",
    "connect",
    "shutdown",
]

#: yarl knows default ports for ws/wss but not mqtt/mqtts; supply them.
DEFAULT_PORTS = {"mqtt": 1883, "mqtts": 8883}

#: The name of a shipped broker wire. ``paho`` is the default.
MqttImpl = Literal["paho", "aiomqtt"]

#: The two shipped broker wires, by name. ``paho`` is the default.
SUPPORTED_IMPLS: Tuple[MqttImpl, ...] = ("paho", "aiomqtt")


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
    retry: RetryPolicy
    verify_tls: bool = False
    tls_client_auth: Optional[TlsClientAuth] = None
    logger: Optional["logging.Logger"] = None
    #: Bound on one connect attempt for the supervised async impls
    #: (``None`` = the transport's own default; paho owns its own timeouts).
    open_timeout: Optional[float] = None


def initial_topics(url: yarl.URL) -> List[str]:
    """The ``?topic=`` subscriptions a URL asks for, in order (may be empty)."""
    return list(url.query.getall("topic", []))


def build_pool(
    impl: MqttImpl,
    pool: Optional[Pool[MqttTransport]],
    provider: Optional[EventLoopProvider] = None,
    wire_keepalive: Optional[WireKeepalive] = None,
) -> Pool[MqttTransport]:
    """The :class:`Pool` to lease from -- the caller's, or a default for ``impl``.

    A default pool shares one wire per :class:`MqttBroker` and hands out
    :class:`MqttLease` leases. ``paho`` builds the sync network-thread
    :class:`~simplyprint_ws_client.wire.paho.Paho`; ``aiomqtt`` the async
    reconnecting :class:`~simplyprint_ws_client.wire.aiomqtt.AioMqtt`.
    Per-connect retry/TLS settings ride in the
    :class:`MqttConnectParams` each ``connect`` passes, never in this closure -
    the pool is cached, so a closure would freeze the first caller's options.
    """
    if pool is not None:
        return pool

    mqtt_keepalive = _mqtt_keepalive_seconds(wire_keepalive)

    def make_transport(url: yarl.URL, params: object) -> MqttTransport:
        if isinstance(params, MqttConnectParams):
            retry, verify_tls, logger = params.retry, params.verify_tls, params.logger
            tls_client_auth = params.tls_client_auth
            open_timeout = params.open_timeout
        else:
            retry, verify_tls, logger = RetryPolicy(), False, None
            tls_client_auth = None
            open_timeout = None
        if impl == "paho":
            return Paho(
                url,
                provider=provider,
                keepalive=mqtt_keepalive or 60,
                client_factory=lambda u, logger: default_paho_client(
                    u,
                    logger,
                    verify_tls=verify_tls,
                    tls_client_auth=tls_client_auth,
                    retry=retry,
                ),
                logger=logger,
            )
        if impl == "aiomqtt":
            if tls_client_auth is not None:
                raise ValueError(
                    "mqtt.connect: tls_client_auth requires impl='paho'; "
                    "aiomqtt does not support mutual-TLS client certificates"
                )
            # ``None`` means "the transport's own default", not "unbounded".
            timeout_kwargs = (
                {} if open_timeout is None else {"open_timeout": open_timeout}
            )
            return AioMqtt(
                url,
                retry,
                provider,
                client_factory=lambda u, logger: default_aiomqtt_client(
                    u, logger, keepalive=mqtt_keepalive, verify_tls=verify_tls
                ),
                logger=logger,
                **timeout_kwargs,
            )
        raise ValueError(f"mqtt.connect: unknown impl {impl!r} (use 'paho'/'aiomqtt')")

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

    return DEFAULT_POOLS.get(impl, provider, wire_keepalive, make_pool)


#: One default pool per ``impl``, created on first use and torn down by
#: :func:`shutdown`. A caller that passes its own ``pool`` never touches these.
DEFAULT_POOLS: DefaultPools[MqttTransport] = DefaultPools()


def connect(
    url: Union[str, yarl.URL],
    *,
    impl: MqttImpl = "paho",
    pool: Optional[Pool[MqttTransport]] = None,
    options: Optional[ConnectionOptions] = None,
) -> MqttLease:
    """Lease a pooled, self-healing MQTT connection to ``url``.

    Synchronous and fire-and-forget: it parses the broker out of ``url``, leases
    the shared transport (building a default per-``impl`` :class:`Pool` keyed by
    broker endpoint when ``pool`` is omitted), applies the URL's ``?topic=``
    subscriptions, and returns at once. Readiness/failure arrive as events on the
    lease's ``event_bus``; ``await conn.ready()`` waits for the first connect.

    ``impl`` selects the broker wire (``"paho"`` default, or ``"aiomqtt"``);
    everything else (retry policy, loop provider, wire/app keepalive) rides in
    ``options``. Raises :class:`ValueError` on a URL with no host or an unknown
    ``impl``.
    """
    url = yarl.URL(url) if isinstance(url, str) else url
    if impl not in SUPPORTED_IMPLS:
        raise ValueError(f"mqtt.connect: unknown impl {impl!r} (use 'paho'/'aiomqtt')")
    options = options or ConnectionOptions()
    broker = MqttBroker.from_url(url)
    pool = build_pool(impl, pool, options.provider, options.wire_keepalive)

    lease = pool.connect(
        url,
        MqttConnectParams(
            broker=broker,
            retry=options.retry or RetryPolicy(),
            verify_tls=options.verify_tls,
            tls_client_auth=options.tls_client_auth,
            logger=options.logger,
            open_timeout=options.open_timeout,
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


def shutdown() -> None:
    """Tear down every default pool of this front door. Idempotent."""
    DEFAULT_POOLS.shutdown()
