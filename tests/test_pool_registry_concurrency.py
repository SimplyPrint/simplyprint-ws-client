from __future__ import annotations

import asyncio
import threading
from typing import Optional

import pytest
import yarl

from simplyprint_ws_client.common.asyncio.concurrent import await_concurrent_future
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.wire import mqtt
from simplyprint_ws_client.wire import async_mqtt
from simplyprint_ws_client.wire.async_mqtt import AsyncMqtt
from simplyprint_ws_client.wire.errors import TransportError
from simplyprint_ws_client.wire.messages import MqttMessage
from simplyprint_ws_client.wire.events import Connected, WireEvent
from simplyprint_ws_client.wire.options import WireKeepalive
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.pools import PoolRegistry
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import Transport
from tests.mqtt_broker import Broker
from tests.silent_server import SilentServer


class LoopThread:
    """One test-owned event loop running on its own thread."""

    def __init__(self, name: str) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        def selector_heartbeat() -> None:
            # Supported runtimes can occasionally lose a cross-thread selector
            # wake-up. Real app/web loops already have timers; give this otherwise
            # completely idle test loop the same bounded wake cadence.
            if self.loop.is_running():
                self.loop.call_later(0.01, selector_heartbeat)

        self.loop.call_soon(selector_heartbeat)
        self.ready.set()
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    def start(self) -> None:
        self.thread.start()
        assert self.ready.wait(1)

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()


class LoopBoundTransport(Transport):
    def __init__(
        self,
        url: yarl.URL,
        provider: EventLoopProvider,
        *,
        stop_entered: Optional[asyncio.Event] = None,
        stop_release: Optional[asyncio.Event] = None,
    ) -> None:
        self.url = url
        self.provider = provider
        self.events: EventBus[WireEvent] = EventBus(provider)
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.stop_entered = stop_entered
        self.stop_release = stop_release
        self.stop_loop: Optional[asyncio.AbstractEventLoop] = None
        self.stops = 0

    @property
    def connected(self) -> bool:
        return self.state is ConnectionState.CONNECTED

    def start(self) -> None:
        self.state = ConnectionState.CONNECTED

    async def stop(self) -> None:
        self.stop_loop = asyncio.get_running_loop()
        self.stops += 1
        if self.stop_entered is not None:
            self.stop_entered.set()
        if self.stop_release is not None:
            await self.stop_release.wait()
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: object) -> None:
        return None

    def trip(self, generation: int, reason: Exception) -> None:
        if generation == self.generation:
            self.state = ConnectionState.DISCONNECTED


class GatedStartTransport(LoopBoundTransport):
    """Expands the synchronous start window so stop can contend deterministically."""

    def __init__(
        self,
        url: yarl.URL,
        provider: EventLoopProvider,
        start_entered: threading.Event,
        start_release: threading.Event,
    ) -> None:
        super().__init__(url, provider)
        self.start_entered = start_entered
        self.start_release = start_release

    def start(self) -> None:
        self.start_entered.set()
        assert self.start_release.wait(1)
        super().start()


def _pool(
    provider: EventLoopProvider,
    transports: list[LoopBoundTransport],
    *,
    stop_entered: Optional[asyncio.Event] = None,
    stop_release: Optional[asyncio.Event] = None,
) -> Pool[LoopBoundTransport]:
    def build(url: yarl.URL, _params: object) -> LoopBoundTransport:
        transport = LoopBoundTransport(
            url,
            provider,
            stop_entered=stop_entered,
            stop_release=stop_release,
        )
        transports.append(transport)
        return transport

    return Pool(build=build, key=lambda url, _params: str(url), provider=provider)


async def _install_pool(registry: PoolRegistry[LoopBoundTransport], host: str):
    provider = EventLoopProvider(asyncio.get_running_loop())
    transports: list[LoopBoundTransport] = []
    pool = registry.get(provider, None, lambda: _pool(provider, transports))
    lease = pool.connect(yarl.URL(f"ws://{host}"))
    return pool, lease, transports[0]


@pytest.mark.asyncio
async def test_registry_closes_each_pool_on_its_owner_loop() -> None:
    registry: PoolRegistry[LoopBoundTransport] = PoolRegistry()
    first_owner = LoopThread("first-pool-owner")
    second_owner = LoopThread("second-pool-owner")
    first_owner.start()
    second_owner.start()
    first = second = None
    try:
        first = await await_concurrent_future(
            first_owner.submit(_install_pool(registry, "first"))
        )
        second = await await_concurrent_future(
            second_owner.submit(_install_pool(registry, "second"))
        )
        first_pool, first_lease, first_transport = first
        second_pool, second_lease, second_transport = second

        assert first_pool is not second_pool
        await registry.close()

        assert first_transport.stop_loop is first_owner.loop
        assert second_transport.stop_loop is second_owner.loop
        assert first_transport.stops == second_transport.stops == 1
        with pytest.raises(RuntimeError, match="closed pool registry"):
            registry.get(None, None, lambda: first_pool)
        with pytest.raises(RuntimeError, match="stopped pool"):
            first_pool.connect("ws://late")

        # Sequential close joins the completed terminal barrier.
        await registry.close()
        await await_concurrent_future(first_owner.submit(first_lease.close()))
        await await_concurrent_future(second_owner.submit(second_lease.close()))
    finally:
        first_owner.stop()
        second_owner.stop()


def test_pool_stop_cannot_overtake_inflight_transport_start() -> None:
    loop = asyncio.new_event_loop()
    provider = EventLoopProvider(loop)
    start_entered = threading.Event()
    start_release = threading.Event()
    transport: Optional[GatedStartTransport] = None

    def build(url: yarl.URL, _params: object) -> GatedStartTransport:
        nonlocal transport
        transport = GatedStartTransport(url, provider, start_entered, start_release)
        return transport

    pool = Pool(build=build, key=lambda url, _params: str(url), provider=provider)
    lease_result = []
    stopped_result = []
    connect_thread = threading.Thread(
        target=lambda: lease_result.append(pool.connect("ws://race"))
    )
    stop_finished = threading.Event()

    def stop_pool() -> None:
        stopped_result.extend(pool.stop())
        stop_finished.set()

    stop_thread = threading.Thread(target=stop_pool)
    connect_thread.start()
    assert start_entered.wait(1)
    stop_thread.start()

    # stop must wait for the atomic connect/start section; otherwise it could
    # collect the wire and connect would resume by starting that orphan.
    assert not stop_finished.wait(0.05)
    start_release.set()
    connect_thread.join(timeout=1)
    stop_thread.join(timeout=1)

    assert not connect_thread.is_alive()
    assert not stop_thread.is_alive()
    assert transport is not None
    assert stopped_result == [transport]
    assert transport.connected
    assert pool.endpoints == {}
    with pytest.raises(RuntimeError, match="stopped pool"):
        pool.connect("ws://late")

    asyncio.run(transport.stop())
    asyncio.run(lease_result[0].close())
    loop.close()


@pytest.mark.asyncio
async def test_registry_close_finishes_after_caller_cancellation() -> None:
    registry: PoolRegistry[LoopBoundTransport] = PoolRegistry()
    provider = EventLoopProvider(asyncio.get_running_loop())
    transports: list[LoopBoundTransport] = []
    stop_entered = asyncio.Event()
    stop_release = asyncio.Event()
    pool = registry.get(
        provider,
        None,
        lambda: _pool(
            provider,
            transports,
            stop_entered=stop_entered,
            stop_release=stop_release,
        ),
    )
    lease = pool.connect("ws://blocking-stop")

    close_task = asyncio.create_task(registry.close())
    await stop_entered.wait()
    close_task.cancel()
    await asyncio.sleep(0)

    assert not close_task.done()
    with pytest.raises(RuntimeError, match="closed pool registry"):
        registry.get(None, None, lambda: pool)
    with pytest.raises(RuntimeError, match="stopped pool"):
        pool.connect("ws://orphan")

    stop_release.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    await registry.close()
    assert transports[0].stops == 1
    await lease.close()


# --------------------------------------------------------------------------- #
# The production MQTT wire (AsyncMqtt) over real loopback sockets.
# --------------------------------------------------------------------------- #


async def _install_mqtt_pool(registry, url: str):
    provider = EventLoopProvider(asyncio.get_running_loop())
    pool = mqtt.pool_for(registry, provider, WireKeepalive(interval=20))
    lease = mqtt.connect(url, pool=pool)
    return pool, lease


@pytest.mark.asyncio
async def test_registry_closes_live_mqtt_after_owner_loop_has_closed() -> None:
    """The MQTT wire lives on its owner loop: it is torn down with the loop (no
    leaked socket, no reconnect), and a later registry close neither hangs
    nor raises on the dead loop."""
    broker = await Broker().start()
    registry = PoolRegistry()
    owner = LoopThread("stopped-mqtt-owner")
    owner.start()
    try:
        pool, lease = await await_concurrent_future(
            owner.submit(_install_mqtt_pool(registry, broker.url))
        )
        assert isinstance(lease.transport, AsyncMqtt)
        await broker.wait_for(lambda: len(broker.sessions) == 1)
        owner.stop()

        await asyncio.wait_for(registry.close(), timeout=2)

        await broker.wait_for(lambda: not broker.sessions)
        assert len(broker.connects) == 1  # nothing reconnected after the stop
        assert pool.endpoints == {}
        await asyncio.wait_for(lease.close(), timeout=2)
    finally:
        owner.stop()
        await broker.stop()


PAYLOAD = 32 << 20  # far beyond any loopback socket buffer


@pytest.mark.asyncio
async def test_last_mqtt_lease_keeps_owner_loop_responsive_during_slow_stop(
    monkeypatch,
) -> None:
    """The slow stop: the broker stopped reading with a publish half-sent, so
    closing may wait (bounded) for the socket. However long it takes, that wait
    must not starve unrelated printer work on the owner loop."""
    # Keep the bounded close wait short, if the wire has one.
    monkeypatch.setattr(async_mqtt, "CLOSE_TIMEOUT", 0.5, raising=False)
    server = await SilentServer(stall_after_connack=True).start()
    registry = PoolRegistry()
    provider = EventLoopProvider(asyncio.get_running_loop())
    pool = mqtt.pool_for(registry, provider, WireKeepalive(interval=20))
    lease = mqtt.connect(server.url(), pool=pool)
    stuck_send = None
    try:
        assert await lease.ready(timeout=5)
        stuck_send = asyncio.ensure_future(
            lease.send(MqttMessage("device/SN/request", b"x" * PAYLOAD))
        )
        await asyncio.sleep(0.05)  # the broker isn't reading: bytes pile up
        assert not stuck_send.done()

        loop = asyncio.get_running_loop()
        close_task = asyncio.create_task(lease.close())
        started = last = loop.time()
        worst_gap = 0.0
        while not close_task.done():
            await asyncio.sleep(0.01)
            now = loop.time()
            worst_gap, last = max(worst_gap, now - last), now
            assert now - started < 5, "lease.close() did not finish"
        await close_task
        assert worst_gap < 0.25  # the loop kept turning throughout the stop
        assert pool.endpoints == {}
    finally:
        if stuck_send is not None:
            stuck_send.cancel()
            await asyncio.gather(stuck_send, return_exceptions=True)
        await registry.close()
        server.release.set()
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["mqtt", "mqtts"])
async def test_unresponsive_broker_cannot_block_transport_shutdown(scheme) -> None:
    """A printer that accepts TCP and then says nothing (no CONNACK; for mqtts, no TLS handshake) leaves the
    wire stuck in open. Closing the last lease must still finish promptly and
    close the socket."""
    server = await SilentServer().start()
    provider = EventLoopProvider(asyncio.get_running_loop())
    pool = mqtt.pool_for(PoolRegistry(), provider, WireKeepalive(interval=20))
    lease = mqtt.connect(server.url(scheme), pool=pool)
    try:
        await server.wait_for(lambda: server.accepted == 1)
        assert not lease.connected

        await asyncio.wait_for(lease.close(), timeout=0.5)

        assert pool.endpoints == {}
        await server.wait_for(lambda: server.closed == 1, timeout=2)
        assert server.accepted == 1  # and nothing retried after the close
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_stalled_broker_stop_releases_the_socket_and_blocked_send(
    monkeypatch,
) -> None:
    """A broker that stopped reading with a publish half-sent. ``stop`` must
    fail the blocked send (not leave it hanging forever) and drop the socket:
    when the broker reads again it finds the connection gone, not the rest of
    a publish from a transport that was stopped."""
    monkeypatch.setattr(async_mqtt, "CLOSE_TIMEOUT", 0.2, raising=False)
    server = await SilentServer(stall_after_connack=True).start()
    transport = AsyncMqtt(
        yarl.URL(server.url()), EventLoopProvider(asyncio.get_running_loop())
    )
    connected = asyncio.Event()
    transport.events.on(Connected, lambda _event: connected.set())
    transport.start()
    stuck_send = None
    try:
        await asyncio.wait_for(connected.wait(), timeout=5)
        stuck_send = asyncio.ensure_future(
            transport.send(MqttMessage("device/SN/request", b"x" * PAYLOAD))
        )
        await asyncio.sleep(0.05)
        assert not stuck_send.done()

        await asyncio.wait_for(transport.stop(), timeout=2)

        done, _ = await asyncio.wait({stuck_send}, timeout=1)
        assert stuck_send in done
        assert isinstance(stuck_send.exception(), TransportError)
        server.release.set()
        await server.wait_for(lambda: server.closed == 1, timeout=5)
        assert len(server.received[0]) < PAYLOAD
    finally:
        if stuck_send is not None:
            stuck_send.cancel()
            await asyncio.gather(stuck_send, return_exceptions=True)
        await transport.stop()
        server.release.set()
        await server.stop()
