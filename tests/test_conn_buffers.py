"""Event delivery tests for pooled connection leases."""

from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest
import yarl

from simplyprint_ws_client.contrib.connection import Connection
from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    ConnectionEvent,
    MessageReceived,
)
from simplyprint_ws_client.contrib.connection.pool import Pool
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import Transport
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider


class FakeTransport(Transport):
    def __init__(self, url: yarl.URL) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[ConnectionEvent] = EventBus()
        self.live = False
        self.stops = 0

    @property
    def connected(self) -> bool:
        return self.live

    def start(self) -> None:
        self.live = True
        self.state = ConnectionState.CONNECTED

    async def stop(self) -> None:
        self.stops += 1
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: object) -> None:
        return None

    def route(self, message: object) -> Optional[str]:
        return None


def current_provider() -> EventLoopProvider:
    return EventLoopProvider(loop=asyncio.get_event_loop())


async def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


def build_pool(transports: List[FakeTransport]) -> Pool:
    def build(url: yarl.URL, params: object) -> FakeTransport:
        transport = FakeTransport(url)
        transports.append(transport)
        return transport

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_pool_queues_async_lease_handlers():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://host/path"))
    handled = []

    async def on_message(event: MessageReceived) -> None:
        await asyncio.sleep(0)
        handled.append(event.message)

    lease.event_bus.on(MessageReceived, on_message)

    await transports[0].events.emit(MessageReceived(1, "frame"))

    await wait_for(lambda: handled == ["frame"])
    assert handled == ["frame"]
    await lease.close()


@pytest.mark.asyncio
async def test_pool_preserves_delivery_order():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://host/path"))
    order = []

    async def on_connected(event: Connected) -> None:
        await asyncio.sleep(0)
        order.append(("connected", event.generation))

    async def on_message(event: MessageReceived) -> None:
        await asyncio.sleep(0)
        order.append(("message", event.message))

    lease.event_bus.on(Connected, on_connected)
    lease.event_bus.on(MessageReceived, on_message)

    await transports[0].events.emit(Connected(1))
    await transports[0].events.emit(MessageReceived(1, "one"))
    await transports[0].events.emit(MessageReceived(1, "two"))

    await wait_for(
        lambda: order == [("connected", 1), ("message", "one"), ("message", "two")]
    )
    assert order == [("connected", 1), ("message", "one"), ("message", "two")]
    await lease.close()


@pytest.mark.asyncio
async def test_closed_lease_stops_receiving_pool_events():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://host/path"))
    got = []

    lease.event_bus.on(MessageReceived, lambda event: got.append(event.message))
    await lease.close()
    await transports[0].events.emit(MessageReceived(1, "late"))

    assert got == []


@pytest.mark.asyncio
async def test_one_bad_lease_listener_does_not_block_other_leases():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    url = yarl.URL("ws://host/path")
    bad = pool.connect(url)
    good = pool.connect(url)
    got = []

    async def fails(event: MessageReceived) -> None:
        raise RuntimeError("bad listener")

    bad.event_bus.on(MessageReceived, fails)
    good.event_bus.on(MessageReceived, lambda event: got.append(event.message))

    await transports[0].events.emit(MessageReceived(1, "frame"))

    await wait_for(lambda: got == ["frame"])
    assert got == ["frame"]
    await bad.close()
    await good.close()


@pytest.mark.asyncio
async def test_slow_lease_handler_does_not_block_pool_fanout():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://host/path"))
    release = asyncio.Event()
    handled = []

    async def slow(event: MessageReceived) -> None:
        handled.append(event.message)
        await release.wait()

    lease.event_bus.on(MessageReceived, slow)

    await transports[0].events.emit(MessageReceived(1, "one"))
    await wait_for(lambda: handled == ["one"])

    await asyncio.wait_for(
        transports[0].events.emit(MessageReceived(1, "two")), timeout=0.1
    )
    assert handled == ["one"]

    release.set()
    await wait_for(lambda: handled == ["one", "two"])
    await lease.close()


@pytest.mark.asyncio
async def test_lease_close_cancels_owned_delivery_task():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://host/path"))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_finishes(_event: MessageReceived) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    lease.event_bus.on(MessageReceived, never_finishes)
    await transports[0].events.emit(MessageReceived(1, "frame"))
    await wait_for(started.is_set)

    await lease.close()
    await wait_for(cancelled.is_set)


@pytest.mark.asyncio
async def test_direct_lease_event_bus_does_not_drop_bursts():
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease: Connection = pool.connect(yarl.URL("ws://host/path"))
    got = []

    async def on_message(event: MessageReceived) -> None:
        await asyncio.sleep(0)
        got.append(event.message)

    lease.event_bus.on(MessageReceived, on_message)

    for index in range(100):
        await lease.event_bus.emit(MessageReceived(1, index))

    assert got == list(range(100))
    await lease.close()
