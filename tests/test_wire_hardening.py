"""Regression tests for the wire teardown/lifecycle hardening pass.

Pins: pool shutdown actually stops transports, lease close survives a broken
broker link and settles its waiters, MQTT wildcard filters route (or at least
match) correctly, keepalive start is idempotent, and credentials never leak
through reprs.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List

import pytest
import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff
from simplyprint_ws_client.wire.events import (
    Connected,
    MessageReceived,
    WireEvent,
)
from simplyprint_ws_client.wire.keepalive import Keepalive, ConnectionKeepalive
from simplyprint_ws_client.wire.lease import MqttLease
from simplyprint_ws_client.wire.mqtt import MqttBroker
from simplyprint_ws_client.wire.paho import retry_delay_bounds
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.pools import PoolRegistry
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
    MqttTransport,
    NotConnected,
    Transport,
    is_wildcard_filter,
    topic_matches,
)


def current_provider() -> EventLoopProvider:
    return EventLoopProvider(factory=asyncio.get_running_loop)


class FakeTransport(Transport):
    def __init__(self, url: yarl.URL) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[WireEvent] = EventBus()
        self.provider = current_provider()
        self.starts = 0
        self.stops = 0
        self.live = False

    @property
    def connected(self) -> bool:
        return self.live

    def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: object) -> None:
        if not self.live:
            raise NotConnected("fake transport not connected")

    async def go_up(self) -> None:
        self.generation += 1
        self.live = True
        self.state = ConnectionState.CONNECTED
        await self.events.emit(Connected(self.generation))


class FakeBrokerTransport(FakeTransport, MqttTransport):
    def __init__(self, url: yarl.URL, *, broken_unsubscribe: bool = False) -> None:
        super().__init__(url)
        self.broken_unsubscribe = broken_unsubscribe
        self.subscribed: List[str] = []

    async def subscribe(self, topic: str) -> None:
        self.subscribed.append(topic)

    async def unsubscribe(self, topic: str) -> None:
        if self.broken_unsubscribe:
            raise OSError("broker link is gone")
        self.subscribed.remove(topic)

    async def deliver(self, topic: str) -> None:
        await self.events.emit(
            MessageReceived(self.generation, SimpleNamespace(topic=topic))
        )


def make_broker_pool(transports: List[FakeBrokerTransport], **transport_kwargs) -> Pool:
    def build(url: yarl.URL, params: object) -> FakeBrokerTransport:
        transport = FakeBrokerTransport(url, **transport_kwargs)
        transports.append(transport)
        return transport

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        route=lambda message: message.topic,
        lease_class=MqttLease,
        provider=current_provider(),
    )


# --- pool shutdown stops the sockets -------------------------------------------


@pytest.mark.asyncio
async def test_pool_stop_returns_live_transports():
    transports: List[FakeBrokerTransport] = []
    pool = make_broker_pool(transports)
    lease = pool.connect(yarl.URL("mqtt://host"))

    stopped = pool.stop()

    assert stopped == [lease.transport]
    # A release after stop finds no endpoint - the caller owns the stop now.
    assert pool.release(lease) is None


@pytest.mark.asyncio
async def test_pool_registry_close_stops_transports():
    transports: List[FakeBrokerTransport] = []
    pools: PoolRegistry = PoolRegistry()
    pool = pools.get(None, None, lambda: make_broker_pool(transports))
    pool.connect(yarl.URL("mqtt://host"))

    await pools.close()

    # Previously the transports were orphaned (never stopped) on shutdown.
    assert [t.stops for t in transports] == [1]
    with pytest.raises(RuntimeError, match="closed pool registry"):
        pools.get(None, None, lambda: make_broker_pool(transports))


# --- lease close is exception-safe and settles waiters ---------------------------


@pytest.mark.asyncio
async def test_mqtt_lease_close_survives_broken_unsubscribe():
    transports: List[FakeBrokerTransport] = []
    pool = make_broker_pool(transports, broken_unsubscribe=True)
    lease = pool.connect(yarl.URL("mqtt://host"))
    await lease.subscribe("printer/report")

    await lease.close()

    assert lease.closed
    # The last lease still stopped the shared transport despite the failure.
    assert transports[0].stops == 1


@pytest.mark.asyncio
async def test_ready_resolves_false_when_lease_closes():
    transports: List[FakeBrokerTransport] = []
    pool = make_broker_pool(transports)
    lease = pool.connect(yarl.URL("mqtt://host"))

    waiter = asyncio.ensure_future(lease.ready())
    await asyncio.sleep(0)
    assert not waiter.done()

    await lease.close()
    assert await asyncio.wait_for(waiter, 1) is False


# --- MQTT wildcard semantics ------------------------------------------------------


def test_topic_matches_full_mqtt_wildcards():
    assert topic_matches("a/+/c", "a/b/c")
    assert not topic_matches("a/+/c", "a/b/d")
    assert not topic_matches("a/+", "a")
    assert topic_matches("a/#", "a")
    assert topic_matches("a/#", "a/b/c")
    assert topic_matches("#", "anything/at/all")
    assert not topic_matches("a/b", "a/b/c")
    assert not topic_matches("+", "a/b")


def test_is_wildcard_filter():
    assert is_wildcard_filter("a/+/c")
    assert is_wildcard_filter("#")
    assert is_wildcard_filter("a/#")
    assert not is_wildcard_filter("a/b/c")
    assert not is_wildcard_filter("a/b+c")  # '+' must be a whole level
    assert not is_wildcard_filter(42)


@pytest.mark.asyncio
async def test_plus_wildcard_subscription_receives_messages():
    transports: List[FakeBrokerTransport] = []
    pool = make_broker_pool(transports)
    lease = pool.connect(yarl.URL("mqtt://host"))
    other = pool.connect(yarl.URL("mqtt://host"))

    received: List[str] = []
    missed: List[str] = []
    lease.event_bus.on(MessageReceived, lambda e: received.append(e.message.topic))
    other.event_bus.on(MessageReceived, lambda e: missed.append(e.message.topic))

    await lease.subscribe("device/+/report")
    await other.subscribe("something/else")
    await transports[0].deliver("device/123/report")

    for _ in range(10):
        await asyncio.sleep(0)
        if received:
            break

    # Previously a `+` filter was registered as an exact route and silently
    # received nothing.
    assert received == ["device/123/report"]
    assert missed == []

    await lease.close()
    await other.close()


# --- keepalive idempotent start ----------------------------------------------------


@pytest.mark.asyncio
async def test_keepalive_start_is_idempotent():
    transports: List[FakeBrokerTransport] = []
    pool = make_broker_pool(transports)
    lease = pool.connect(yarl.URL("mqtt://host"))

    keepalive = ConnectionKeepalive(lease, Keepalive(interval=60))
    assert keepalive.start() is keepalive
    first_task = keepalive.task
    # A second start must not raise ("listener already registered") nor orphan
    # the first run task.
    assert keepalive.start() is keepalive
    assert keepalive.task is first_task

    await lease.close()


# --- paho retry mapping + credential redaction ---------------------------------------


def test_retry_delay_bounds_maps_backoff_envelope():
    policy = RetryPolicy(backoff=ConstantBackoff(7))
    assert retry_delay_bounds(policy) == (7, 7)

    fast = RetryPolicy(backoff=ConstantBackoff(0))
    low, high = retry_delay_bounds(fast)
    assert low >= 1 and high >= low


def test_mqtt_broker_repr_redacts_password():
    broker = MqttBroker("host", 8883, "user", "hunter2")
    assert "hunter2" not in repr(broker)
    assert "hunter2" not in str(broker)
    assert "hunter2" not in f"{broker!r}"
