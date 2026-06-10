"""Benchmark the connection runtime delivery/backpressure model.

Run with the library venv::

    .venv/bin/python scripts/bench_conn.py

Useful knobs::

    .venv/bin/python scripts/bench_conn.py --messages 200000 --slow-messages 50000
    .venv/bin/python scripts/bench_conn.py --topics 2000 --transports 20 --topic-messages 200000
    .venv/bin/python scripts/bench_conn.py --json

The default benchmark uses fake in-process transports. Opt-in flags add real
loopback WebSocket and caller-provided MQTT broker scenarios. It measures:

* end-to-end connection delivery throughput, CPU time, and tracemalloc peak;
* complete message-event processing through sync and async handlers;
* paho-style producer-thread event delivery through Courier + transport fanout;
* MQTT-style topic routing across 1000+ topics over many pooled transports;
* flaky pooled transports that disconnect/reconnect while messages are routed;
* sheddable ``AT_MOST_ONCE`` delivery under a slow consumer;
* lossless ``AT_LEAST_ONCE`` delivery under the same slow consumer;
* direct ``Courier`` ``BLOCK`` backpressure cost from a producer thread.

These are performance probes, not CI assertions. Numbers vary by host; compare
relative changes before/after a refactor on the same machine.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import yarl

from simplyprint_ws_client.common.events import EventBus
from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff

from simplyprint_ws_client.device.connection import mqtt, ws
from simplyprint_ws_client.device.connection.lease import Lease
from simplyprint_ws_client.device.connection.lease import MqttLease
from simplyprint_ws_client.common.wire.events import (
    Connected,
    Connecting,
    WireEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.common.wire.errors import TransportError
from simplyprint_ws_client.common.wire.messages import (
    MqttMessage,
    QoS,
    WsMessage,
)
from simplyprint_ws_client.device.connection.mqtt import mqtt_message_route
from simplyprint_ws_client.common.wire.policy import RetryPolicy
from simplyprint_ws_client.device.connection.pool import Pool
from simplyprint_ws_client.common.wire.state import ConnectionState
from simplyprint_ws_client.common.wire.transport import (
    MqttTransport,
    NotConnected,
    Transport,
)


DEFAULT_MESSAGES = 200_000
DEFAULT_SLOW_MESSAGES = 50_000
DEFAULT_BLOCK_MESSAGES = 10_000
DEFAULT_TOPICS = 1_000
DEFAULT_TRANSPORTS = 10
DEFAULT_TOPIC_MESSAGES = 100_000
DEFAULT_FLAKY_MESSAGES = 100_000
DEFAULT_FLAKY_INTERVAL = 1_000
DEFAULT_ACTUAL_WS_MESSAGES = 20_000
DEFAULT_ACTUAL_MQTT_MESSAGES = 20_000


@dataclass(frozen=True)
class Metric:
    name: str
    messages: int
    delivered: int
    dropped: int
    wall_seconds: float
    cpu_seconds: float
    input_per_second: float
    msg_per_second: float
    cpu_msg_per_second: float
    peak_bytes: int
    notes: str = ""


class FakeTransport(Transport):
    """A no-I/O transport whose event bus is driven by the benchmark."""

    def __init__(self, url: yarl.URL) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 1
        self.events: EventBus[WireEvent] = EventBus()
        self.live = False
        self.sent: List[object] = []

    @property
    def connected(self) -> bool:
        return self.live

    def start(self) -> None:
        self.live = True
        self.state = ConnectionState.CONNECTED

    async def stop(self) -> None:
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: object) -> None:
        if not self.live:
            raise NotConnected("fake transport is not connected")
        self.sent.append(message)

    async def push(self, message: WsMessage) -> None:
        await self.events.emit(MessageReceived(self.generation, message, message.qos))


class FakeMqttTransport(MqttTransport):
    """A no-I/O MQTT transport used to stress pool routing by topic."""

    def __init__(self, url: yarl.URL) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 1
        self.events: EventBus[WireEvent] = EventBus()
        self.live = False
        self.subscriptions: List[str] = []

    @property
    def connected(self) -> bool:
        return self.live

    def start(self) -> None:
        self.live = True
        self.state = ConnectionState.CONNECTED

    async def stop(self) -> None:
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def send(self, _message: object) -> None:
        if not self.live:
            raise NotConnected("fake mqtt transport is not connected")

    async def subscribe(self, topic: str) -> None:
        self.subscriptions.append(topic)

    async def unsubscribe(self, topic: str) -> None:
        try:
            self.subscriptions.remove(topic)
        except ValueError:
            pass

    async def push(self, message: MqttMessage) -> None:
        await self.events.emit(MessageReceived(self.generation, message, message.qos))

    async def cycle(self) -> None:
        previous_generation = self.generation
        self.live = False
        self.state = ConnectionState.DISCONNECTED
        await self.events.emit(
            Disconnected(
                previous_generation,
                TransportError("flaky benchmark drop", code="benchmark_drop"),
            )
        )

        self.generation += 1
        self.state = ConnectionState.CONNECTING
        await self.events.emit(Connecting(self.generation))

        self.live = True
        self.state = ConnectionState.CONNECTED
        await self.events.emit(Connected(self.generation))


class ActualWsServer:
    """Tiny loopback server for opt-in real WebSocket transport benchmarks."""

    def __init__(self) -> None:
        self.server = None
        self.host = "127.0.0.1"
        self.port = 0
        self.connections: List = []
        self.connected = asyncio.Event()

    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self.server = await serve(self.handler, self.host, 0)
        sock = next(iter(self.server.sockets))
        self.port = sock.getsockname()[1]

    @property
    def url(self) -> yarl.URL:
        return yarl.URL(f"ws://{self.host}:{self.port}/")

    async def handler(self, connection) -> None:
        self.connections.append(connection)
        self.connected.set()
        try:
            async for _frame in connection:
                pass
        except Exception:  # noqa: BLE001 -- client teardown is normal here
            pass
        finally:
            with contextlib.suppress(ValueError):
                self.connections.remove(connection)

    async def push(self, payload: str) -> None:
        connection = await self.current()
        await connection.send(payload)

    async def drop_current(self) -> None:
        connection = await self.current()
        await connection.close(code=1011, reason="flaky benchmark drop")

    async def current(self, timeout: float = 5.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while not self.connections:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("no websocket client connected")
            self.connected.clear()
            await asyncio.wait_for(self.connected.wait(), remaining)
        return self.connections[-1]

    async def stop(self) -> None:
        if self.server is None:
            return
        for connection in list(self.connections):
            with contextlib.suppress(Exception):
                await connection.close()
        self.server.close()
        await self.server.wait_closed()
        self.server = None


def build_pool(transports: List[FakeTransport]) -> Pool[FakeTransport]:
    def build(url: yarl.URL, _params: object) -> FakeTransport:
        transport = FakeTransport(url)
        transports.append(transport)
        return transport

    return Pool(
        build=build,
        key=lambda url, _params: str(url),
        provider=EventLoopProvider(loop=asyncio.get_running_loop()),
    )


def build_mqtt_pool(transports: List[FakeMqttTransport]) -> Pool[FakeMqttTransport]:
    def build(url: yarl.URL, _params: object) -> FakeMqttTransport:
        transport = FakeMqttTransport(url)
        transports.append(transport)
        return transport

    return Pool(
        build=build,
        key=lambda url, _params: str(url),
        route=mqtt_message_route,
        lease_class=MqttLease,
        provider=EventLoopProvider(loop=asyncio.get_running_loop()),
    )


async def wait_for(predicate: Callable[[], bool], timeout: float = 30.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise TimeoutError("benchmark condition did not complete")


async def close_lease(lease: Lease) -> None:
    await lease.close()
    await asyncio.sleep(0)


async def connection_throughput(total: int) -> Metric:
    """End-to-end lossless delivery with a trivial sync handler."""
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://bench/throughput"))
    transport = transports[0]
    message = WsMessage.text("x", qos=QoS.AT_LEAST_ONCE)
    delivered = 0

    def handler(_event: MessageReceived) -> None:
        nonlocal delivered
        delivered += 1

    lease.event_bus.on(MessageReceived, handler)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    for _ in range(total):
        await transport.push(message)
    await wait_for(lambda: delivered == total)

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    await close_lease(lease)
    return metric(
        "connection sync handler processed", total, delivered, 0, wall, cpu, peak
    )


async def connection_async_handler_throughput(total: int) -> Metric:
    """End-to-end lossless delivery with an async handler that completes immediately."""
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://bench/async-throughput"))
    transport = transports[0]
    message = WsMessage.text("x", qos=QoS.AT_LEAST_ONCE)
    delivered = 0

    async def handler(_event: MessageReceived) -> None:
        nonlocal delivered
        delivered += 1

    lease.event_bus.on(MessageReceived, handler)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    for _ in range(total):
        await transport.push(message)
    await wait_for(lambda: delivered == total)

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    await close_lease(lease)
    return metric(
        "connection async handler processed",
        total,
        delivered,
        0,
        wall,
        cpu,
        peak,
        "awaited async handler completion",
    )


async def cross_thread_transport_throughput(total: int) -> Metric:
    """Producer thread -> Courier -> transport EventBus -> pool -> lease Courier."""
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL("ws://bench/cross-thread"))
    transport = transports[0]
    event = MessageReceived(
        transport.generation,
        WsMessage.text("x", qos=QoS.AT_LEAST_ONCE),
        QoS.AT_LEAST_ONCE,
    )
    delivered = 0

    def handler(_event: MessageReceived) -> None:
        nonlocal delivered
        delivered += 1

    lease.event_bus.on(MessageReceived, handler)
    producer_courier: Courier[MessageReceived] = Courier(
        sink=transport.events.emit,
        is_async_sink=True,
        policy=OverflowPolicy.UNBOUNDED,
        provider=EventLoopProvider(loop=asyncio.get_running_loop()),
    )
    errors: List[BaseException] = []

    def producer() -> None:
        try:
            for _ in range(total):
                producer_courier.post(event)
        except BaseException as error:  # noqa: BLE001 - report producer failures
            errors.append(error)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    thread = threading.Thread(target=producer)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0)
    thread.join(1.0)
    await wait_for(lambda: delivered == total, timeout=60.0)

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    producer_courier.close(drain=False)
    await close_lease(lease)

    if errors:
        raise RuntimeError("producer failed") from errors[0]
    return metric(
        "cross-thread transport events",
        total,
        delivered,
        producer_courier.dropped,
        wall,
        cpu,
        peak,
        "paho-shaped thread hop",
    )


async def mqtt_topic_pool_throughput(
    total: int,
    topic_count: int,
    transport_count: int,
) -> Metric:
    """Process messages across many subscribed topics and pooled transports."""
    if topic_count <= 0:
        raise ValueError("topic_count must be positive")
    if transport_count <= 0:
        raise ValueError("transport_count must be positive")

    transports: List[FakeMqttTransport] = []
    pool = build_mqtt_pool(transports)
    transport_count = min(transport_count, topic_count)
    leases: List[MqttLease] = []
    topics: List[str] = []
    processed = 0

    def handler(_event: MessageReceived) -> None:
        nonlocal processed
        processed += 1

    for index in range(topic_count):
        endpoint = index % transport_count
        topic = f"bench/{endpoint}/{index}"
        lease = pool.connect(yarl.URL(f"mqtt://broker-{endpoint}/"))
        if not isinstance(lease, MqttLease):
            raise TypeError("mqtt pool did not return an MqttLease")
        await lease.subscribe(topic)
        lease.event_bus.on(MessageReceived, handler)
        leases.append(lease)
        topics.append(topic)

    messages = [MqttMessage(topic, b"x", qos=QoS.AT_LEAST_ONCE) for topic in topics]
    transport_by_topic = [
        transports[index % transport_count] for index in range(topic_count)
    ]

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    for index in range(total):
        topic_index = index % topic_count
        await transport_by_topic[topic_index].push(messages[topic_index])
    await wait_for(lambda: processed == total, timeout=120.0)

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for lease in leases:
        await close_lease(lease)

    return metric(
        f"mqtt pool {topic_count:,} topics / {transport_count:,} transports",
        total,
        processed,
        0,
        wall,
        cpu,
        peak,
        "one lease per topic, routed by topic",
    )


async def mqtt_flaky_topic_pool_throughput(
    total: int,
    topic_count: int,
    transport_count: int,
    interval: int,
) -> Metric:
    """Route messages while pooled MQTT transports disconnect and reconnect."""
    if topic_count <= 0:
        raise ValueError("topic_count must be positive")
    if transport_count <= 0:
        raise ValueError("transport_count must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")

    transports: List[FakeMqttTransport] = []
    pool = build_mqtt_pool(transports)
    transport_count = min(transport_count, topic_count)
    leases: List[MqttLease] = []
    topics: List[str] = []
    lease_counts = [0 for _ in range(transport_count)]
    processed = 0
    connecting = 0
    connected = 0
    disconnected = 0

    def on_message(_event: MessageReceived) -> None:
        nonlocal processed
        processed += 1

    def on_connecting(_event: Connecting) -> None:
        nonlocal connecting
        connecting += 1

    def on_connected(_event: Connected) -> None:
        nonlocal connected
        connected += 1

    def on_disconnected(_event: Disconnected) -> None:
        nonlocal disconnected
        disconnected += 1

    for index in range(topic_count):
        endpoint = index % transport_count
        topic = f"bench/flaky/{endpoint}/{index}"
        lease = pool.connect(yarl.URL(f"mqtt://flaky-broker-{endpoint}/"))
        if not isinstance(lease, MqttLease):
            raise TypeError("mqtt pool did not return an MqttLease")
        await lease.subscribe(topic)
        lease.event_bus.on(MessageReceived, on_message)
        lease.event_bus.on(Connecting, on_connecting)
        lease.event_bus.on(Connected, on_connected)
        lease.event_bus.on(Disconnected, on_disconnected)
        leases.append(lease)
        topics.append(topic)
        lease_counts[endpoint] += 1

    messages = [MqttMessage(topic, b"x", qos=QoS.AT_LEAST_ONCE) for topic in topics]
    transport_by_topic = [
        transports[index % transport_count] for index in range(topic_count)
    ]

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    cycles = 0
    expected_lifecycle = 0
    for index in range(total):
        topic_index = index % topic_count
        await transport_by_topic[topic_index].push(messages[topic_index])
        sent = index + 1
        if sent < total and sent % interval == 0:
            await wait_for(lambda: processed == sent, timeout=120.0)
            transport_index = cycles % transport_count
            expected_lifecycle += lease_counts[transport_index]
            await transports[transport_index].cycle()
            cycles += 1

    await wait_for(lambda: processed == total, timeout=120.0)
    await wait_for(
        lambda: connecting == expected_lifecycle
        and connected == expected_lifecycle
        and disconnected == expected_lifecycle,
        timeout=120.0,
    )

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for lease in leases:
        await close_lease(lease)

    return metric(
        f"mqtt flaky pool {topic_count:,} topics / {transport_count:,} transports",
        total,
        processed,
        0,
        wall,
        cpu,
        peak,
        (
            f"{cycles:,} drops/reconnects, "
            f"{expected_lifecycle * 3:,} lifecycle handler calls"
        ),
    )


def fast_retry() -> RetryPolicy:
    return RetryPolicy(backoff=ConstantBackoff(0.0), max_attempts=3)


def skipped_metric(name: str, notes: str) -> Metric:
    return Metric(
        name=name,
        messages=0,
        delivered=0,
        dropped=0,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        input_per_second=0.0,
        msg_per_second=0.0,
        cpu_msg_per_second=0.0,
        peak_bytes=0,
        notes=notes,
    )


async def actual_ws_loopback_throughput(total: int, impl: str) -> Metric:
    """Measure a real ws:// loopback through the public ws.connect front door."""
    server = ActualWsServer()
    lease = None
    pool = None
    try:
        await server.start()
    except ImportError as error:
        return skipped_metric(f"actual websocket {impl}", f"skipped: {error}")

    try:
        provider = EventLoopProvider(loop=asyncio.get_running_loop())
        pool = ws.build_pool(impl, None, provider)
        lease = ws.connect(
            server.url,
            impl=impl,
            retry=fast_retry(),
            pool=pool,
            provider=provider,
        )
        processed = 0

        def handler(_event: MessageReceived) -> None:
            nonlocal processed
            processed += 1

        lease.event_bus.on(MessageReceived, handler)
        if not await lease.ready(timeout=5.0):
            return skipped_metric(
                f"actual websocket {impl}",
                "skipped: connection did not become ready",
            )

        gc.collect()
        tracemalloc.start()
        tracemalloc.clear_traces()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        for _ in range(total):
            await server.push("x")
        await wait_for(lambda: processed == total, timeout=120.0)

        cpu = time.process_time() - cpu_start
        wall = time.perf_counter() - wall_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return metric(
            f"actual websocket {impl}",
            total,
            processed,
            0,
            wall,
            cpu,
            peak,
            "real loopback server push through ws.connect",
        )
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        if lease is not None:
            await close_lease(lease)
        if pool is not None:
            pool.stop()
        await server.stop()


async def actual_ws_flaky_loopback_throughput(
    total: int,
    impl: str,
    interval: int,
) -> Metric:
    """Measure real WebSocket reconnect churn through the public front door."""
    if interval <= 0:
        raise ValueError("interval must be positive")

    server = ActualWsServer()
    lease = None
    pool = None
    try:
        await server.start()
    except ImportError as error:
        return skipped_metric(f"actual websocket flaky {impl}", f"skipped: {error}")

    try:
        provider = EventLoopProvider(loop=asyncio.get_running_loop())
        pool = ws.build_pool(impl, None, provider)
        lease = ws.connect(
            server.url,
            impl=impl,
            retry=fast_retry(),
            pool=pool,
            provider=provider,
        )
        processed = 0
        connected = 0
        disconnected = 0

        def on_message(_event: MessageReceived) -> None:
            nonlocal processed
            processed += 1

        def on_connected(_event: Connected) -> None:
            nonlocal connected
            connected += 1

        def on_disconnected(_event: Disconnected) -> None:
            nonlocal disconnected
            disconnected += 1

        lease.event_bus.on(MessageReceived, on_message)
        lease.event_bus.on(Connected, on_connected)
        lease.event_bus.on(Disconnected, on_disconnected)
        if not await lease.ready(timeout=5.0):
            return skipped_metric(
                f"actual websocket flaky {impl}",
                "skipped: connection did not become ready",
            )

        connected_before = connected
        disconnected_before = disconnected
        gc.collect()
        tracemalloc.start()
        tracemalloc.clear_traces()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        sent = 0
        cycles = 0
        while sent < total:
            batch = min(interval, total - sent)
            for _ in range(batch):
                await server.push("x")
            sent += batch
            await wait_for(lambda: processed == sent, timeout=120.0)
            if sent < total:
                old_generation = lease.generation
                await server.drop_current()
                cycles += 1
                await wait_for(
                    lambda: lease.connected and lease.generation > old_generation,
                    timeout=10.0,
                )
                await server.current()

        await wait_for(lambda: processed == total, timeout=120.0)
        await wait_for(
            lambda: connected - connected_before >= cycles
            and disconnected - disconnected_before >= cycles,
            timeout=120.0,
        )

        cpu = time.process_time() - cpu_start
        wall = time.perf_counter() - wall_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return metric(
            f"actual websocket flaky {impl}",
            total,
            processed,
            0,
            wall,
            cpu,
            peak,
            f"{cycles:,} server-side drops and reconnects",
        )
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        if lease is not None:
            await close_lease(lease)
        if pool is not None:
            pool.stop()
        await server.stop()


async def actual_mqtt_broker_throughput(
    total: int,
    url: str,
    impl: str,
    topic: Optional[str],
) -> Metric:
    """Measure a real broker only when the caller provides one explicitly."""
    parsed = yarl.URL(url)
    benchmark_topic = topic or f"simplyprint/bench/{time.monotonic_ns()}"
    lease = None
    pool = None
    try:
        provider = EventLoopProvider(loop=asyncio.get_running_loop())
        retry = fast_retry()
        pool = mqtt.build_pool(impl, retry, None, provider)
        lease = mqtt.connect(
            parsed,
            impl=impl,
            retry=retry,
            pool=pool,
            provider=provider,
        )
        processed = 0

        def handler(_event: MessageReceived) -> None:
            nonlocal processed
            processed += 1

        lease.event_bus.on(MessageReceived, handler)
        if not await lease.ready(timeout=5.0):
            return skipped_metric(
                f"actual mqtt {impl}",
                "skipped: connection did not become ready",
            )
        await lease.subscribe(benchmark_topic)

        message = MqttMessage(benchmark_topic, b"x", qos=QoS.AT_LEAST_ONCE)

        gc.collect()
        tracemalloc.start()
        tracemalloc.clear_traces()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        for _ in range(total):
            await lease.send(message)
        await wait_for(lambda: processed == total, timeout=120.0)

        cpu = time.process_time() - cpu_start
        wall = time.perf_counter() - wall_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return metric(
            f"actual mqtt {impl}",
            total,
            processed,
            0,
            wall,
            cpu,
            peak,
            f"broker {parsed.host}:{parsed.port or 1883}, topic {benchmark_topic}",
        )
    except ImportError as error:
        return skipped_metric(f"actual mqtt {impl}", f"skipped: {error}")
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        if lease is not None:
            await close_lease(lease)
        if pool is not None:
            pool.stop()


async def connection_slow_consumer(total: int, qos: QoS) -> Metric:
    """Flood one lease while its async handler yields once per message."""
    transports: List[FakeTransport] = []
    pool = build_pool(transports)
    lease = pool.connect(yarl.URL(f"ws://bench/slow/{qos.name}"))
    transport = transports[0]
    message = WsMessage.text("x", qos=qos)
    delivered = 0

    async def handler(_event: MessageReceived) -> None:
        nonlocal delivered
        delivered += 1
        await asyncio.sleep(0)

    lease.event_bus.on(MessageReceived, handler)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    for _ in range(total):
        await transport.push(message)

    if qos is QoS.AT_LEAST_ONCE:
        await wait_for(lambda: delivered == total, timeout=60.0)
    else:
        await wait_for(
            lambda: delivered + lease._courier.dropped >= total,  # noqa: SLF001
            timeout=60.0,
        )

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    dropped = lease._courier.dropped  # noqa: SLF001 - benchmark introspection
    await close_lease(lease)
    name = (
        "connection slow consumer sheddable"
        if qos is QoS.AT_MOST_ONCE
        else "connection slow consumer lossless"
    )
    notes = "bounded drops expected" if qos is QoS.AT_MOST_ONCE else "no drops expected"
    return metric(name, total, delivered, dropped, wall, cpu, peak, notes)


async def courier_block_backpressure(total: int, maxsize: int) -> Metric:
    """Measure producer-thread backpressure with Courier BLOCK."""
    delivered = 0

    async def sink(_item: int) -> None:
        nonlocal delivered
        delivered += 1
        await asyncio.sleep(0)

    courier: Courier[int] = Courier(
        sink=sink,
        is_async_sink=True,
        policy=OverflowPolicy.BLOCK,
        maxsize=maxsize,
        provider=EventLoopProvider(loop=asyncio.get_running_loop()),
    )

    errors: List[BaseException] = []

    def producer() -> None:
        try:
            for index in range(total):
                courier.post(index)
        except BaseException as error:  # noqa: BLE001 - report producer failures
            errors.append(error)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    thread = threading.Thread(target=producer)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0)
    thread.join(1.0)
    await wait_for(lambda: delivered == total, timeout=60.0)

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    courier.close(drain=False)

    if errors:
        raise RuntimeError("producer failed") from errors[0]
    return metric(
        f"courier BLOCK backpressure maxsize={maxsize}",
        total,
        delivered,
        courier.dropped,
        wall,
        cpu,
        peak,
        "producer thread blocks when queue is full",
    )


def metric(
    name: str,
    messages: int,
    delivered: int,
    dropped: int,
    wall_seconds: float,
    cpu_seconds: float,
    peak_bytes: int,
    notes: str = "",
) -> Metric:
    return Metric(
        name=name,
        messages=messages,
        delivered=delivered,
        dropped=dropped,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        input_per_second=messages / wall_seconds if wall_seconds > 0 else 0.0,
        msg_per_second=delivered / wall_seconds if wall_seconds > 0 else 0.0,
        cpu_msg_per_second=delivered / cpu_seconds if cpu_seconds > 0 else 0.0,
        peak_bytes=peak_bytes,
        notes=notes,
    )


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{value} B"


def print_table(metrics: List[Metric]) -> None:
    rows = [
        (
            item.name,
            f"{item.messages:,}",
            f"{item.delivered:,}",
            f"{item.dropped:,}",
            f"{item.wall_seconds:,.4f}s",
            f"{item.cpu_seconds:,.4f}s",
            f"{item.input_per_second:,.0f}",
            f"{item.msg_per_second:,.0f}",
            human_bytes(item.peak_bytes),
            item.notes,
        )
        for item in metrics
    ]
    headers = (
        "scenario",
        "input",
        "processed",
        "dropped",
        "wall",
        "cpu",
        "input/s",
        "processed/s",
        "peak",
        "notes",
    )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def line(values: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    print(line(headers))
    print(line(tuple("-" * width for width in widths)))
    for row in rows:
        print(line(row))


async def run_benchmarks(args: argparse.Namespace) -> List[Metric]:
    metrics = [await connection_throughput(args.messages)]
    metrics.append(await connection_async_handler_throughput(args.messages))
    metrics.append(await cross_thread_transport_throughput(args.messages))
    metrics.append(
        await mqtt_topic_pool_throughput(
            args.topic_messages,
            args.topics,
            args.transports,
        )
    )
    metrics.append(
        await mqtt_flaky_topic_pool_throughput(
            args.flaky_messages,
            args.topics,
            args.transports,
            args.flaky_interval,
        )
    )
    metrics.append(await connection_slow_consumer(args.slow_messages, QoS.AT_MOST_ONCE))
    metrics.append(
        await connection_slow_consumer(args.slow_messages, QoS.AT_LEAST_ONCE)
    )
    metrics.append(
        await courier_block_backpressure(args.block_messages, args.block_maxsize)
    )
    if args.actual_ws:
        impls = (
            ("websockets", "aiohttp")
            if args.actual_ws_impl == "both"
            else (args.actual_ws_impl,)
        )
        for impl in impls:
            metrics.append(
                await actual_ws_loopback_throughput(args.actual_ws_messages, impl)
            )
            metrics.append(
                await actual_ws_flaky_loopback_throughput(
                    args.actual_ws_messages,
                    impl,
                    args.flaky_interval,
                )
            )
    if args.actual_mqtt_url:
        metrics.append(
            await actual_mqtt_broker_throughput(
                args.actual_mqtt_messages,
                args.actual_mqtt_url,
                args.actual_mqtt_impl,
                args.actual_mqtt_topic,
            )
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES)
    parser.add_argument("--slow-messages", type=int, default=DEFAULT_SLOW_MESSAGES)
    parser.add_argument("--block-messages", type=int, default=DEFAULT_BLOCK_MESSAGES)
    parser.add_argument("--block-maxsize", type=int, default=64)
    parser.add_argument("--topics", type=int, default=DEFAULT_TOPICS)
    parser.add_argument("--transports", type=int, default=DEFAULT_TRANSPORTS)
    parser.add_argument("--topic-messages", type=int, default=DEFAULT_TOPIC_MESSAGES)
    parser.add_argument("--flaky-messages", type=int, default=DEFAULT_FLAKY_MESSAGES)
    parser.add_argument("--flaky-interval", type=int, default=DEFAULT_FLAKY_INTERVAL)
    parser.add_argument("--actual-ws", action="store_true")
    parser.add_argument(
        "--actual-ws-impl",
        choices=("websockets", "aiohttp", "both"),
        default="websockets",
    )
    parser.add_argument(
        "--actual-ws-messages", type=int, default=DEFAULT_ACTUAL_WS_MESSAGES
    )
    parser.add_argument("--actual-mqtt-url")
    parser.add_argument(
        "--actual-mqtt-impl",
        choices=("aiomqtt", "paho"),
        default="aiomqtt",
    )
    parser.add_argument("--actual-mqtt-topic")
    parser.add_argument(
        "--actual-mqtt-messages", type=int, default=DEFAULT_ACTUAL_MQTT_MESSAGES
    )
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = asyncio.run(run_benchmarks(args))
    if args.json:
        print(json.dumps([asdict(item) for item in metrics], indent=2))
        return

    print(f"connection runtime benchmark (python {sys.version.split()[0]})")
    print("fake by default; optional real sockets; compare changes on the same host")
    print()
    print_table(metrics)


if __name__ == "__main__":
    main()
