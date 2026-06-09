"""The MQTT front door: ``mqtt.connect(url)`` -> a pooled, self-healing lease.

This is the protocol-specific face a printer client meets. It owns the wire
message shape (:class:`MqttMessage`), the broker-endpoint identity it pools by
(:class:`MqttBroker`), and the one-liner :func:`connect` that parses a broker URL,
shares one socket per endpoint through a :class:`~simplyprint_ws_client.contrib.connection.pool.Pool`,
and hands back a lease already carrying the URL's initial subscriptions.

The two concrete broker wires live in their own modules and are imported here, not
re-implemented: the async :class:`~simplyprint_ws_client.contrib.connection.aiomqtt.AioMqtt`
(one ``async with`` client per attempt, on the shared reconnect loop) and the sync
:class:`~simplyprint_ws_client.contrib.connection.paho.Paho` (its own network thread,
self-healing, events couriered onto the loop). Both speak the exact
:class:`~simplyprint_ws_client.contrib.connection.transport.MqttTransport` contract, so a
consumer that only listens cannot tell which it got, and both import their wire
library lazily so importing this module needs neither installed.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Union

import yarl

from simplyprint_ws_client.contrib.connection.aiomqtt import AioMqtt
from simplyprint_ws_client.contrib.connection.connection import MqttConnection
from simplyprint_ws_client.contrib.connection.messages import MqttMessage
from simplyprint_ws_client.contrib.connection.paho import Paho
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.pool import Pool
from simplyprint_ws_client.contrib.connection.transport import MqttTransport

__all__ = [
    "MqttMessage",
    "MqttBroker",
    "MqttConnection",
    "connect",
    "shutdown",
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


def initial_topics(url: yarl.URL) -> List[str]:
    """The ``?topic=`` subscriptions a URL asks for, in order (may be empty)."""
    return list(url.query.getall("topic", []))


def default_topic(url: yarl.URL) -> Optional[str]:
    """The first ``?topic=`` of a URL -- the topic a bare ``send(bytes)`` uses."""
    topics = initial_topics(url)
    return topics[0] if topics else None


def build_pool(
    impl: str,
    retry: RetryPolicy,
    pool: Optional[Pool[MqttTransport]],
) -> Pool[MqttTransport]:
    """The :class:`Pool` to lease from -- the caller's, or a default for ``impl``.

    A default pool shares one wire per :class:`MqttBroker` and hands out
    :class:`MqttConnection` leases. ``aiomqtt`` builds the async reconnecting
    :class:`~simplyprint_ws_client.contrib.connection.aiomqtt.AioMqtt`; ``paho`` the sync
    network-thread :class:`~simplyprint_ws_client.contrib.connection.paho.Paho`.
    """
    if pool is not None:
        return pool

    existing = DEFAULT_POOLS.get(impl)
    if existing is not None:
        return existing

    def make_transport(url: yarl.URL, params: object) -> MqttTransport:
        if impl == "paho":
            return Paho(url)
        if impl == "aiomqtt":
            return AioMqtt(url, retry)
        raise ValueError(f"mqtt.connect: unknown impl {impl!r} (use 'aiomqtt'/'paho')")

    def endpoint_key(url: yarl.URL, params: object) -> MqttBroker:
        return params if isinstance(params, MqttBroker) else MqttBroker.from_url(url)

    built: Pool[MqttTransport] = Pool(
        build=make_transport,
        key=endpoint_key,
        lease_class=MqttConnection,
    )
    DEFAULT_POOLS[impl] = built
    return built


#: Process-wide default pools, one per impl, created lazily by :func:`connect`.
DEFAULT_POOLS: Dict[str, Pool[MqttTransport]] = {}


def connect(
    url: Union[str, yarl.URL],
    *,
    impl: str = "aiomqtt",
    retry: Optional[RetryPolicy] = None,
    pool: Optional[Pool[MqttTransport]] = None,
) -> MqttConnection:
    """Lease a pooled, self-healing MQTT connection to ``url``.

    Synchronous and fire-and-forget: it parses the broker out of ``url``, leases
    the shared transport (building a default per-``impl`` :class:`Pool` keyed by
    broker endpoint when ``pool`` is omitted), applies the URL's ``?topic=``
    subscriptions, and returns at once. Readiness/failure arrive as events on the
    lease's ``event_bus``; ``await conn.ready()`` waits for the first connect.

    ``retry`` defaults to a fresh :class:`RetryPolicy` (retry forever at a constant
    pace); pass one to cap attempts or set a give-up deadline. Raises
    :class:`ValueError` on a URL with no host or an unknown ``impl``.
    """
    url = yarl.URL(url) if isinstance(url, str) else url
    retry = retry or RetryPolicy()
    broker = MqttBroker.from_url(url)
    pool = build_pool(impl, retry, pool)

    lease = pool.connect(url, broker)
    assert isinstance(lease, MqttConnection)

    # Apply the URL's initial subscriptions through the lease's own seam. connect
    # is sync, so the async subscribe runs as a task on the transport's loop; the
    # interest is recorded on the lease synchronously first, so routing is correct
    # the instant the link comes up (and re-asserted on every (re)connect).
    topics = initial_topics(url)
    if topics:
        loop = lease.provider.event_loop
        for topic in topics:
            lease.topics.add(topic)
            loop.create_task(lease.backend.subscribe(topic))

    return lease


def shutdown() -> None:
    """Close every default :func:`connect` pool (no-op if unused).

    Stops the pools' fan-out and drops their bookkeeping; each transport's async
    ``stop`` is the last lease's to await, so this is the coarse process-exit hook.
    """
    for pool in DEFAULT_POOLS.values():
        pool.stop()
    DEFAULT_POOLS.clear()
