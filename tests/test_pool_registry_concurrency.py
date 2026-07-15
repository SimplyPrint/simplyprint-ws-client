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
from simplyprint_ws_client.wire.events import WireEvent
from simplyprint_ws_client.wire.options import WireKeepalive
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.pools import PoolRegistry
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import Transport


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


class QuietPahoClient:
    def __init__(self) -> None:
        self.on_pre_connect = None
        self.on_connect = None
        self.on_connect_fail = None
        self.on_message = None
        self.on_disconnect = None
        self.stops = 0
        self.stop_thread: Optional[str] = None

    def username_pw_set(self, username, password) -> None:
        return None

    def connect_async(self, host, port, keepalive) -> None:
        return None

    def loop_start(self) -> int:
        return 0

    def loop_stop(self) -> int:
        self.stops += 1
        self.stop_thread = threading.current_thread().name
        return 0

    def disconnect(self) -> None:
        return None

    def is_connected(self) -> bool:
        return False


class BlockingStopPahoClient(QuietPahoClient):
    def __init__(self) -> None:
        super().__init__()
        self.stop_entered = threading.Event()
        self.stop_release = threading.Event()
        self.calls: list[str] = []

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def loop_stop(self) -> int:
        self.calls.append("loop_stop")
        self.stop_entered.set()
        self.stop_release.wait()
        return super().loop_stop()


async def _install_paho_pool(registry, client: QuietPahoClient):
    provider = EventLoopProvider(asyncio.get_running_loop())
    pool = mqtt.pool_for(registry, provider, WireKeepalive(interval=20))
    lease = mqtt.connect("mqtt://printer", pool=pool)
    return pool, lease


@pytest.mark.asyncio
async def test_registry_stops_live_paho_after_owner_loop_has_closed(
    monkeypatch,
) -> None:
    client = QuietPahoClient()
    monkeypatch.setattr(
        mqtt, "default_paho_client", lambda _url, _logger, **_kwargs: client
    )
    registry = PoolRegistry()
    owner = LoopThread("stopped-paho-owner")
    owner.start()
    pool, lease = await await_concurrent_future(
        owner.submit(_install_paho_pool(registry, client))
    )
    owner.stop()

    await registry.close()

    assert client.stops == 1
    assert client.stop_thread is not None
    assert client.stop_thread != "stopped-paho-owner"
    await lease.close()


@pytest.mark.asyncio
async def test_last_paho_lease_keeps_owner_loop_responsive_during_slow_stop(
    monkeypatch,
) -> None:
    client = BlockingStopPahoClient()
    monkeypatch.setattr(
        mqtt, "default_paho_client", lambda _url, _logger, **_kwargs: client
    )
    registry = PoolRegistry()
    provider = EventLoopProvider(asyncio.get_running_loop())
    pool = mqtt.pool_for(registry, provider, WireKeepalive(interval=20))
    lease = mqtt.connect("mqtt://printer", pool=pool)

    close_task = asyncio.create_task(lease.close())
    try:
        await asyncio.wait_for(
            asyncio.to_thread(client.stop_entered.wait),
            timeout=5,
        )
        assert not close_task.done()

        # A slow paho thread join must not starve unrelated printer work.
        heartbeats = 0
        for _ in range(100):
            await asyncio.sleep(0)
            heartbeats += 1
        assert heartbeats == 100
        assert not close_task.done()
    finally:
        client.stop_release.set()
        await close_task

    assert client.calls == ["disconnect", "loop_stop"]
    assert client.stops == 1
    assert client.stop_thread is not None
    assert client.stop_thread != threading.current_thread().name
    assert pool.endpoints == {}
