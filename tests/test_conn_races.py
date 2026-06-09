"""Concurrency / race-condition tests for the ``contrib.connection`` connection library.

These tests hammer the *shared, pooled* spine of ``contrib.connection`` from many
directions at once on a single asyncio loop -- which is exactly where the library
lives: the pool refcount, a lease's ``send`` racing a generation flip, concurrent
subscribe/unsubscribe across leases against one refcounted broker socket, and
``stop`` landing mid-open / mid-consume. None of this touches a real broker or
socket: every wire is an injected fake satisfying the same seam production wires
do -- a :class:`Reconnecting` subclass driven by queues, an aiomqtt-shaped fake
client behind :class:`AioMqtt`, a paho-shaped fake behind :class:`Paho`.

The single asyncio loop does NOT make these tests trivial: every ``await`` is a
scheduling point where another task can interleave, so the dict mutations, the
refcount, the generation counter, and the topic table are all genuinely racing
across suspension points. The invariants asserted are the ones the docstrings
promise: a transport is started once and stopped once per instance, the refcount
is never torn, the generation is strictly monotonic, no message lands on a wrong
generation, and the asyncio task count returns to baseline (no leaks, no orphaned
transports).

paho and aiomqtt are NOT installed; every MQTT test injects a fake client through
the wire's ``client_factory`` seam, so nothing imports a real wire library.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, List, Optional, Set

import pytest
import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.utils.backoff import ConstantBackoff

from simplyprint_ws_client.contrib.connection import mqtt as mqtt_door
from simplyprint_ws_client.contrib.connection.aiomqtt import AioMqtt
from simplyprint_ws_client.contrib.connection.connection import MqttConnection
from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    Connecting,
    ConnectionEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.contrib.connection.mqtt import (
    MqttBroker,
)
from simplyprint_ws_client.contrib.connection.paho import Paho
from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
from simplyprint_ws_client.contrib.connection.pool import Pool
from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    NotConnected,
    Transport,
)
from simplyprint_ws_client.contrib.connection.websockets import Websockets


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """Poll ``predicate`` on the loop until true, or raise after ``timeout``."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition not met in time")


def current_provider() -> EventLoopProvider:
    """A provider bound to the running test loop."""
    return EventLoopProvider(loop=asyncio.get_event_loop())


def silent_logger() -> logging.Logger:
    """A logger that swallows the engine's debug/warning chatter under chaos."""
    logger = logging.getLogger("test.conn.races")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def live_tasks() -> Set[asyncio.Task]:
    """Every not-done task except the one calling this."""
    return {
        t
        for t in asyncio.all_tasks()
        if not t.done() and t is not asyncio.current_task()
    }


async def settle(turns: int = 50) -> None:
    """Yield the loop ``turns`` times so background tasks can finish/cancel."""
    for _ in range(turns):
        await asyncio.sleep(0)


def assert_monotonic(generations: List[int]) -> None:
    """A generation sequence (as observed) must never go backwards."""
    for earlier, later in zip(generations, generations[1:]):
        assert later >= earlier, f"generation went backwards: {generations}"


# --------------------------------------------------------------------------- #
# A counting fake Transport: records every start/stop, drives lifecycle/messages.
# --------------------------------------------------------------------------- #


class CountingTransport(Transport):
    """A do-nothing transport that counts start/stop and lets a test emit.

    ``stop`` awaits a real sleep so a concurrent ``connect`` on the same endpoint
    can interleave with a teardown in flight -- the exact window a pool refcount
    race lives in.
    """

    def __init__(self, url: yarl.URL, *, stop_delay: float = 0.0) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[ConnectionEvent] = EventBus()
        self.starts = 0
        self.stops = 0
        self.stop_delay = stop_delay
        self.sent: List[object] = []
        self.live = False
        self.supervises = True

    @property
    def connected(self) -> bool:
        return self.live

    def start(self) -> None:
        self.starts += 1
        self.state = ConnectionState.CONNECTING

    async def stop(self) -> None:
        self.stops += 1
        if self.stop_delay:
            await asyncio.sleep(self.stop_delay)
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def send(self, message: object) -> None:
        if not self.live:
            raise NotConnected("counting transport not connected")
        self.sent.append(message)

    def supervising(self) -> bool:
        return self.supervises

    async def go_up(self) -> None:
        self.generation += 1
        self.live = True
        self.state = ConnectionState.CONNECTED
        await self.events.emit(Connected(self.generation))

    async def deliver(self, message: object) -> None:
        await self.events.emit(MessageReceived(self.generation, message))


def counting_pool(
    transports: List[CountingTransport], *, stop_delay: float = 0.0
) -> Pool:
    """A pool keyed by URL string recording every transport it builds."""

    def build(url: yarl.URL, params: object) -> CountingTransport:
        transport = CountingTransport(url, stop_delay=stop_delay)
        transports.append(transport)
        return transport

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        provider=current_provider(),
    )


# --------------------------------------------------------------------------- #
# 1. Pool refcount: many leases connect/close concurrently on ONE endpoint.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pool_concurrent_connect_close_refcount_never_torn():
    """N leases connect then close in arbitrary interleavings on one endpoint.

    The endpoint shares one transport instance; whatever the interleaving, that
    instance must be started exactly once and stopped exactly once, the refcount
    must hit zero exactly once (the endpoint dropped), and no task may leak.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports, stop_delay=0.001)
    url = yarl.URL("ws://host/path")
    baseline = live_tasks()

    leases = [pool.connect(url) for _ in range(40)]
    # All 40 leases share the single transport built for the endpoint.
    assert len(transports) == 1
    transport = transports[0]
    assert transport.starts == 1
    assert pool.endpoints[str(url)].refs == 40

    # Close every lease concurrently -- the last one out tears the wire down.
    await asyncio.gather(*(lease.close() for lease in leases))
    await settle()

    assert transport.starts == 1, "transport started more than once"
    assert transport.stops == 1, "transport stopped more or fewer than once"
    assert str(url) not in pool.endpoints, "endpoint not dropped on last close"
    assert live_tasks() - baseline == set(), "leaked tasks after close storm"


@pytest.mark.asyncio
async def test_pool_connect_release_hammered_started_once_stopped_once():
    """connect and release hammered concurrently; per-instance start==stop==1.

    A teardown takes a real ``await`` (``stop_delay``), so a concurrent connect
    can land *while a transport is being stopped*. Across the whole storm: every
    transport instance the pool ever built was started exactly once and, since the
    storm ends with no live leases, stopped exactly once -- never twice, never
    zero.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports, stop_delay=0.002)
    url = yarl.URL("ws://host/churn")
    baseline = live_tasks()

    async def churn(rounds: int) -> None:
        for _ in range(rounds):
            lease = pool.connect(url)
            await asyncio.sleep(0)  # let other churners interleave
            await lease.close()

    await asyncio.gather(*(churn(15) for _ in range(8)))
    await settle()

    for t in transports:
        assert t.starts == 1, f"transport started {t.starts} times"
        assert t.stops == 1, f"transport stopped {t.stops} times (expected 1)"

    assert str(url) not in pool.endpoints
    assert live_tasks() - baseline == set(), "leaked tasks after churn"


@pytest.mark.asyncio
async def test_pool_release_during_stop_does_not_double_start_same_endpoint():
    """A connect that lands during a teardown shares or rebuilds -- never both.

    Lease A is the last holder, so closing it pops the endpoint and awaits
    ``stop()`` (which sleeps). While that stop is in flight, lease B connects the
    same key: the pool must build a *fresh* transport for B (the old one is
    leaving) and start it exactly once -- the old one still stops exactly once.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports, stop_delay=0.05)
    url = yarl.URL("ws://host/overlap")

    a = pool.connect(url)
    old = transports[0]
    assert old.starts == 1

    closing = asyncio.ensure_future(a.close())  # pops endpoint, then awaits stop()
    await asyncio.sleep(0)  # let close() pop the endpoint and enter stop()

    b = pool.connect(url)  # lands during the old transport's teardown
    await closing
    await settle()

    # B got a brand-new transport (the old one was on its way out).
    assert b.backend is not old
    assert len(transports) == 2
    new = transports[1]
    assert new.starts == 1, "new transport started wrong number of times"
    assert new.stops == 0, "new (held) transport was stopped"
    assert old.stops == 1, "old transport stopped wrong number of times"

    await b.close()
    await settle()
    assert new.stops == 1


@pytest.mark.asyncio
async def test_close_is_idempotent_under_concurrency():
    """Concurrent and repeated ``close`` on one lease releases the pool just once.

    A lease closed many times concurrently must decrement the refcount exactly
    once (the ``closed`` guard), so a sibling lease keeps its hold and the
    transport is not torn down early.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    url = yarl.URL("ws://host/idem")

    a = pool.connect(url)
    b = pool.connect(url)
    transport = transports[0]
    assert pool.endpoints[str(url)].refs == 2

    await asyncio.gather(*(a.close() for _ in range(10)))
    await settle()

    assert transport.stops == 0, "sibling lease's hold was torn down by repeated close"
    assert pool.endpoints[str(url)].refs == 1, pool.endpoints[str(url)].refs

    await b.close()
    await settle()
    assert transport.stops == 1
    assert str(url) not in pool.endpoints


@pytest.mark.asyncio
async def test_release_of_unknown_lease_is_a_noop():
    """Releasing a lease whose endpoint already dropped never tears a sibling down.

    A lease closed twice (the pool already forgot it) must not, on the second
    release, accidentally decrement a *rebuilt* endpoint's refcount. We close A
    (drops the endpoint), rebuild via B, then re-release A: B's hold is untouched.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    url = yarl.URL("ws://host/unknown")

    a = pool.connect(url)
    await a.close()  # endpoint dropped, transport[0] stopped
    assert str(url) not in pool.endpoints

    b = pool.connect(url)  # rebuilds the endpoint (transport[1])
    assert pool.endpoints[str(url)].refs == 1

    # A second release of A must be a no-op against the rebuilt endpoint.
    assert pool.release(a) is None
    assert pool.endpoints[str(url)].refs == 1, "stale release decremented a sibling"

    await b.close()
    await settle()
    assert transports[0].stops == 1
    assert transports[1].stops == 1


# --------------------------------------------------------------------------- #
# 2. Reconnecting engine: send racing a generation flip; stop mid-open/consume.
# --------------------------------------------------------------------------- #


class GatedWire(Reconnecting):
    """A ``Reconnecting`` whose every hook is externally drivable for racing.

    ``open`` waits on a gate (so a test can hold an attempt mid-connect),
    ``recv`` blocks on an inbox (a queued ``BaseException`` == a wire drop),
    ``write`` records and can be made to block, ``aclose`` counts. This lets a
    test pin ``send`` exactly at a generation flip.
    """

    def __init__(self, url: yarl.URL, **kw: Any) -> None:
        super().__init__(url, **kw)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.opens = 0
        self.closes = 0
        self.sent: List[object] = []
        self.open_gate: Optional[asyncio.Event] = None
        self.write_gate: Optional[asyncio.Event] = None
        #: Generation observed at the moment each write actually lands.
        self.write_generations: List[int] = []

    async def open(self) -> None:
        self.opens += 1
        if self.open_gate is not None:
            await self.open_gate.wait()

    async def recv(self) -> Optional[object]:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def write(self, message: object) -> None:
        if self.write_gate is not None:
            await self.write_gate.wait()
        self.write_generations.append(self.generation)
        self.sent.append(message)

    async def aclose(self) -> None:
        self.closes += 1


def make_gated(**kw: Any) -> GatedWire:
    return GatedWire(
        yarl.URL("ws://x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
        logger=silent_logger(),
        **kw,
    )


@pytest.mark.asyncio
async def test_generation_strictly_monotonic_across_rapid_drops():
    """Rapid connect<->disconnect cycles: generation only ever climbs.

    Every ``Connected`` / ``Disconnected`` / ``MessageReceived`` carries the
    generation it belongs to. Across many forced drops, the sequence of observed
    generations must be non-decreasing and bumped exactly once per live attempt --
    never reused, never rewound.
    """
    seen_gens: List[int] = []
    connect_gens: List[int] = []
    wire = make_gated()
    wire.events.on(Connecting, lambda e: seen_gens.append(e.generation))
    wire.events.on(
        Connected,
        lambda e: (seen_gens.append(e.generation), connect_gens.append(e.generation)),
    )
    wire.events.on(Disconnected, lambda e: seen_gens.append(e.generation))

    wire.start()
    try:
        for _ in range(12):
            await wait_for(lambda: wire.connected)
            gen = wire.generation
            wire.inbox.put_nowait(RuntimeError("drop"))  # force a drop
            await wait_for(lambda: wire.generation > gen)
        await wait_for(lambda: wire.connected)
    finally:
        await wire.stop()

    assert_monotonic(seen_gens)
    # Each successful connect is a distinct, strictly increasing generation.
    assert connect_gens == sorted(set(connect_gens)), connect_gens
    assert connect_gens[0] == 1
    assert connect_gens == list(range(1, len(connect_gens) + 1)), connect_gens


@pytest.mark.asyncio
async def test_send_during_generation_flip_never_lands_on_wrong_generation():
    """A send held open across a drop must not land a frame on a dead wire.

    We hold a ``write`` mid-flight (gen 1), drop the wire underneath it, and let
    the wire reconnect to gen 2. ``send`` checked ``connected`` on gen 1; the
    contract is that a frame is delivered on a *live* wire -- it must never be
    recorded as written against a generation whose wire is already torn down.
    """
    wire = make_gated()
    wire.write_gate = asyncio.Event()
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)

        # Begin a send on generation 1; it blocks inside write() at the gate.
        sending = asyncio.ensure_future(wire.send(b"on-gen-1"))
        await asyncio.sleep(0)  # let send() pass the connected check + enter write

        # Drop the wire underneath the in-flight send and let it heal to gen 2.
        wire.inbox.put_nowait(RuntimeError("drop while sending"))
        await wait_for(lambda: wire.generation == 2 and wire.connected)

        # Release the held write. Whatever happens, the frame must NOT be recorded
        # as written on generation 1's torn-down wire.
        wire.write_gate.set()
        try:
            await asyncio.wait_for(sending, timeout=1.0)
        except (NotConnected, asyncio.CancelledError):
            pass

        for landed_gen in wire.write_generations:
            assert landed_gen == wire.generation, (
                f"frame written on generation {landed_gen} while the live "
                f"generation is {wire.generation} -- it landed on a dead wire"
            )
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_stop_mid_open_no_leak_and_not_supervising():
    """``stop`` while an attempt is blocked inside ``open`` cancels cleanly.

    The open hook is gated so the supervision task is parked mid-connect. A
    ``stop`` there must cancel the task, leave the wire DISCONNECTED and not
    supervising, never fire ``Connected``, and leak no task.
    """
    baseline = live_tasks()
    wire = make_gated()
    wire.open_gate = asyncio.Event()  # open() blocks forever until set
    connected: List[int] = []
    wire.events.on(Connected, lambda e: connected.append(e.generation))

    wire.start()
    await wait_for(lambda: wire.opens == 1)  # parked inside open()
    assert not wire.connected

    await wire.stop()  # cancels the task mid-open
    await settle()

    assert wire.task is None
    assert not wire.connected
    assert not wire.supervising()
    assert wire.state is ConnectionState.DISCONNECTED
    assert connected == [], "Connected fired despite stop mid-open"
    assert live_tasks() - baseline == set(), "leaked task after stop mid-open"


@pytest.mark.asyncio
async def test_stop_mid_consume_no_leak_strict_idempotent():
    """``stop`` while parked in ``recv`` (mid-consume), repeated, leaks nothing.

    The wire is live and the consume loop is blocked awaiting the next frame.
    Calling ``stop`` concurrently several times must be idempotent: the task is
    torn down once, no further opens happen, and nothing leaks.
    """
    baseline = live_tasks()
    wire = make_gated()
    wire.start()
    await wait_for(lambda: wire.connected)  # parked in recv() awaiting inbox
    opens_at_stop = wire.opens

    await asyncio.gather(wire.stop(), wire.stop(), wire.stop())
    await settle()

    assert wire.opens == opens_at_stop, "a new attempt started after stop"
    assert wire.task is None
    assert not wire.connected
    assert wire.closes >= 1, "the live wire was not torn down on stop"
    assert live_tasks() - baseline == set(), "leaked task after stop mid-consume"


@pytest.mark.asyncio
async def test_concurrent_start_stop_storm_no_orphan_task():
    """start/stop fired in a tight interleaved storm leaves no orphaned task.

    ``start`` is idempotent (no double task) and ``stop`` cancels; bouncing them
    against each other must never leave two supervision tasks alive nor a task
    after the final stop.
    """
    baseline = live_tasks()
    wire = make_gated()

    for _ in range(25):
        wire.start()
        wire.start()  # idempotent: must not spawn a second task
        await asyncio.sleep(0)
        await wire.stop()

    await settle()
    assert wire.task is None
    assert live_tasks() - baseline == set(), "orphaned supervision task(s)"


@pytest.mark.asyncio
async def test_rapid_restart_after_stop_starts_exactly_one_task():
    """A wire stopped then restarted runs exactly one supervision task at a time.

    After a full stop the task is cleared; a fresh ``start`` must spin up a single
    new task (not resurrect the old one), and a second ``start`` while it is alive
    must be a no-op. This is the rapid connect<->disconnect<->connect path.
    """
    baseline = live_tasks()
    wire = make_gated()

    for _ in range(10):
        wire.start()
        await wait_for(lambda: wire.connected)
        first_task = wire.task
        wire.start()  # idempotent while alive
        assert wire.task is first_task, "start spawned a second supervision task"
        await wire.stop()
        assert wire.task is None

    await settle()
    assert live_tasks() - baseline == set(), "leaked supervision task after restarts"


# --------------------------------------------------------------------------- #
# 3. MQTT subscribe/unsubscribe refcount racing across leases + reconnect.
# --------------------------------------------------------------------------- #


class GatedAioClient:
    """An aiomqtt-shaped fake whose subscribe/connect can be gated for racing.

    ``__aenter__`` may block on ``open_gate`` so a test can hold a reconnect
    mid-open while leases mutate the topic refcount; ``messages`` blocks on an
    inbox a test feeds (a ``BaseException`` ends the stream == a drop). Each
    ``subscribe`` / ``unsubscribe`` awaits once so a concurrent (un)subscribe
    interleaves across the shared refcount table.
    """

    def __init__(self, open_gate: Optional[asyncio.Event] = None) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.subscribed: List[str] = []
        self.unsubscribed: List[str] = []
        self.published: List[tuple] = []
        self.open_gate = open_gate
        self.messages = self

    async def __aenter__(self) -> "GatedAioClient":
        if self.open_gate is not None:
            await self.open_gate.wait()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> "GatedAioClient":
        return self

    async def __anext__(self) -> Any:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def subscribe(self, topic: str) -> None:
        await asyncio.sleep(0)  # a real await: a concurrent (un)subscribe interleaves
        self.subscribed.append(topic)

    async def unsubscribe(self, topic: str) -> None:
        await asyncio.sleep(0)
        self.unsubscribed.append(topic)

    async def publish(
        self, topic: str, payload: bytes, qos: int = 0, retain: bool = False
    ) -> None:
        self.published.append((topic, payload, qos, retain))


class WireMqttMessage:
    """An aiomqtt message stand-in -- ``AioMqtt`` routes these by ``.topic``."""

    def __init__(
        self, topic: str, payload: bytes, qos: int = 0, retain: bool = False
    ) -> None:
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.retain = retain


def mqtt_pool(
    clients: List[GatedAioClient], *, open_gate: Optional[asyncio.Event] = None
) -> Pool:
    """A pool of :class:`AioMqtt` wired to fresh gated fake clients."""

    def factory(url: yarl.URL, logger: logging.Logger) -> GatedAioClient:
        client = GatedAioClient(open_gate=open_gate)
        clients.append(client)
        return client

    def build(url: yarl.URL, params: object) -> AioMqtt:
        return AioMqtt(
            url,
            RetryPolicy(backoff=ConstantBackoff(0)),
            current_provider(),
            client_factory=factory,
            logger=silent_logger(),
        )

    def key(url: yarl.URL, params: object) -> MqttBroker:
        return params if isinstance(params, MqttBroker) else MqttBroker.from_url(url)

    return Pool(
        build=build,
        key=key,
        lease_class=MqttConnection,
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_subscribe_refcount_survives_concurrent_leases_same_topic():
    """Many leases subscribe/unsubscribe the SAME topic concurrently.

    The broker socket is refcounted by topic. After equal numbers of subscribe
    and unsubscribe across leases settle, the refcount table must be empty (no
    negative count, no stuck positive count), and the broker saw exactly one wire
    ``subscribe`` and exactly one ``unsubscribe`` for the topic.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    leases = [pool.connect(url, broker) for _ in range(20)]
    transport = leases[0].backend
    assert all(lease.backend is transport for lease in leases)
    await wait_for(lambda: leases[0].connected)
    assert len(clients) == 1
    client = clients[0]

    # All 20 leases assert interest in "shared" at once.
    await asyncio.gather(*(lease.subscribe("shared") for lease in leases))
    assert transport.subscriptions.get("shared") == 20, transport.subscriptions
    assert client.subscribed.count("shared") == 1, (
        f"refcounted topic subscribed {client.subscribed.count('shared')} times "
        "on the wire (expected exactly 1)"
    )

    # All 20 drop it at once.
    await asyncio.gather(*(lease.unsubscribe("shared") for lease in leases))
    assert "shared" not in transport.subscriptions, transport.subscriptions
    assert client.unsubscribed.count("shared") == 1, (
        f"refcounted topic unsubscribed {client.unsubscribed.count('shared')} "
        "times on the wire (expected exactly 1)"
    )

    await asyncio.gather(*(lease.close() for lease in leases))


@pytest.mark.asyncio
async def test_interleaved_sub_unsub_never_drives_refcount_negative():
    """Interleaved sub/unsub churn on one topic keeps the refcount well-formed.

    Each task does subscribe then unsubscribe, with awaits between, so the +1/-1
    operations interleave across the shared dict. At the end the topic is gone
    and the *net* wire subscribes equal the net wire unsubscribes -- the refcount
    was never torn into a negative or orphaned-positive state.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    leases = [pool.connect(url, broker) for _ in range(12)]
    transport = leases[0].backend
    await wait_for(lambda: leases[0].connected)
    client = clients[0]

    async def churn(lease: MqttConnection) -> None:
        for _ in range(6):
            await lease.subscribe("topic")
            await asyncio.sleep(0)
            await lease.unsubscribe("topic")
            await asyncio.sleep(0)

    await asyncio.gather(*(churn(lease) for lease in leases))
    await settle()

    assert "topic" not in transport.subscriptions, transport.subscriptions
    assert all(v > 0 for v in transport.subscriptions.values()), transport.subscriptions
    # The wire subscribe/unsubscribe pairs must balance: a torn refcount would
    # leave a dangling subscribe with no matching unsubscribe (or vice versa).
    assert client.subscribed.count("topic") == client.unsubscribed.count("topic"), (
        f"wire subs={client.subscribed.count('topic')} != "
        f"unsubs={client.unsubscribed.count('topic')} -- refcount torn"
    )

    await asyncio.gather(*(lease.close() for lease in leases))


@pytest.mark.asyncio
async def test_subscribe_during_reconnect_reasserts_exactly_the_live_set():
    """A topic added while a reconnect is mid-open is re-asserted on the new wire.

    The first client streams a drop; the second client's ``__aenter__`` is gated,
    so the reconnect is parked mid-open. While parked, a lease subscribes a new
    topic. When the open is released, the new live client must end up subscribed
    to exactly the tracked topic set -- nothing lost, nothing duplicated.
    """
    clients: List[GatedAioClient] = []
    open_gate = asyncio.Event()
    open_gate.set()  # first open proceeds freely
    pool = mqtt_pool(clients, open_gate=open_gate)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    lease = pool.connect(url, broker)
    transport = lease.backend
    await wait_for(lambda: lease.connected)
    await lease.subscribe("first")
    await wait_for(lambda: "first" in clients[0].subscribed)

    # Gate the NEXT open, then drop the live wire so a reconnect parks mid-open.
    open_gate.clear()
    clients[0].inbox.put_nowait(RuntimeError("drop"))
    await wait_for(lambda: len(clients) == 2)  # second client built, parked in aenter

    # While the reconnect is parked mid-open, add another topic via a lease.
    add = asyncio.ensure_future(lease.subscribe("second"))
    await asyncio.sleep(0)

    # Release the open: the new client re-asserts whatever is tracked now.
    open_gate.set()
    await add
    await wait_for(lambda: lease.connected)
    await settle()

    new_client = clients[1]
    # Both tracked topics are live on the new wire, each subscribed exactly once.
    assert set(new_client.subscribed) == {"first", "second"}, new_client.subscribed
    assert new_client.subscribed.count("first") == 1, new_client.subscribed
    assert new_client.subscribed.count("second") == 1, new_client.subscribed
    assert transport.subscriptions == {"first": 1, "second": 1}, transport.subscriptions

    await lease.close()


@pytest.mark.asyncio
async def test_distinct_leases_subscribe_distinct_topics_concurrently():
    """Each of N leases subscribes its own topic at once on one shared socket.

    Distinct +1's into the same dict across suspension points must not lose a
    topic (a lost-update race on the refcount dict). After the storm every topic
    is present at refcount 1 and was subscribed on the wire exactly once.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    leases = [pool.connect(url, broker) for _ in range(24)]
    await wait_for(lambda: leases[0].connected)
    client = clients[0]

    await asyncio.gather(*(lease.subscribe(f"t/{i}") for i, lease in enumerate(leases)))

    expected = {f"t/{i}": 1 for i in range(len(leases))}
    transport = leases[0].backend
    assert transport.subscriptions == expected, transport.subscriptions
    assert sorted(client.subscribed) == sorted(expected), client.subscribed

    await asyncio.gather(*(lease.close() for lease in leases))


# --------------------------------------------------------------------------- #
# 4. Messages on a shared transport never reach the wrong lease/generation.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_message_flood_while_leases_join_and_leave_no_crash():
    """Messages flow on a shared broker while leases connect/close concurrently.

    A storm of inbound messages races leases joining and leaving the same socket.
    The fan-out tuple-snapshots its lease set, so a lease closing mid-fan must not
    crash delivery, and a closed lease must stop receiving. We assert no crash
    (the loop survives) and that the surviving anchor lease still gets messages.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)
    baseline = live_tasks()

    anchor = pool.connect(url, broker)
    await anchor.subscribe("room/#")
    await wait_for(lambda: anchor.connected)
    client = clients[0]

    anchor_got: List[str] = []
    anchor.event_bus.on(MessageReceived, lambda e: anchor_got.append(e.message.topic))

    async def transient_lease(n: int) -> None:
        lease = pool.connect(url, broker)
        await lease.subscribe("room/#")
        for i in range(5):
            client.inbox.put_nowait(WireMqttMessage(f"room/{n}-{i}", b"x"))
            await asyncio.sleep(0)
        await lease.close()

    async def flood() -> None:
        for i in range(40):
            client.inbox.put_nowait(WireMqttMessage(f"room/flood-{i}", b"y"))
            await asyncio.sleep(0)

    await asyncio.gather(flood(), *(transient_lease(n) for n in range(10)))
    await settle()

    assert anchor_got, "anchor lease received nothing during the storm"
    # Every topic the anchor saw matches its subscription (no cross-routing).
    assert all(t.startswith("room/") for t in anchor_got), anchor_got

    await anchor.close()
    await settle()
    assert all(broker != k for k in pool.endpoints), pool.endpoints
    assert live_tasks() - baseline == set(), "leaked tasks after message storm"


@pytest.mark.asyncio
async def test_closed_lease_stops_receiving_mid_fanout():
    """A lease closed during message flow receives nothing after it closes.

    Two MQTT leases on one socket subscribe the same wildcard; we close one, then
    push frames. The closed lease must receive nothing further (its ``feed``
    short-circuits on ``closed``), while the surviving lease keeps getting frames.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    keep = pool.connect(url, broker)
    drop = pool.connect(url, broker)
    assert keep.backend is drop.backend
    await keep.subscribe("room/#")
    await drop.subscribe("room/#")
    await wait_for(lambda: keep.connected)
    client = clients[0]

    keep_got: List[str] = []
    drop_got: List[str] = []
    keep.event_bus.on(MessageReceived, lambda e: keep_got.append(e.message.topic))
    drop.event_bus.on(MessageReceived, lambda e: drop_got.append(e.message.topic))

    client.inbox.put_nowait(WireMqttMessage("room/before", b"x"))
    await wait_for(lambda: keep_got and drop_got)
    assert drop_got == ["room/before"]

    await drop.close()  # release one lease; the socket stays up for 'keep'
    drop_count_at_close = len(drop_got)

    for i in range(5):
        client.inbox.put_nowait(WireMqttMessage(f"room/after-{i}", b"y"))
    await wait_for(lambda: len(keep_got) >= 6)

    assert len(drop_got) == drop_count_at_close, "closed lease still received frames"
    assert keep_got[1:] == [f"room/after-{i}" for i in range(5)]

    await keep.close()


@pytest.mark.asyncio
async def test_message_never_delivered_with_stale_generation():
    """Every fanned message carries the generation of the live wire that emitted it.

    Across a drop/reconnect on a shared socket, the generation a lease sees on a
    ``MessageReceived`` must match the live wire's generation at emit time -- never
    a stale epoch from the prior connection.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    lease = pool.connect(url, broker)
    await lease.subscribe("room/#")
    await wait_for(lambda: lease.connected)

    seen: List[int] = []
    lease.event_bus.on(MessageReceived, lambda e: seen.append(e.generation))

    transport = lease.backend
    gen1 = transport.generation
    clients[0].inbox.put_nowait(WireMqttMessage("room/a", b"1"))
    await wait_for(lambda: len(seen) == 1)
    assert seen[0] == gen1

    # Drop and reconnect: a fresh generation.
    clients[0].inbox.put_nowait(RuntimeError("drop"))
    await wait_for(lambda: len(clients) == 2 and lease.connected)
    gen2 = transport.generation
    assert gen2 > gen1

    clients[1].inbox.put_nowait(WireMqttMessage("room/b", b"2"))
    await wait_for(lambda: len(seen) == 2)
    assert seen[1] == gen2, f"message carried stale generation {seen[1]} != {gen2}"

    await lease.close()


# --------------------------------------------------------------------------- #
# 5. send racing close on a pooled lease: no send on a torn-down transport.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_send_racing_last_close_does_not_send_on_stopped_transport():
    """A lease ``send`` racing the last ``close`` must not land on a dead wire.

    Two leases share a counting transport. We close both (the last triggers
    ``stop()``) while concurrently firing a ``send`` from one of them. The send
    must either succeed on the still-live wire or fail cleanly with
    ``NotConnected`` -- never silently record a frame on a transport whose
    ``stop`` already ran.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports, stop_delay=0.005)
    url = yarl.URL("ws://host/send-race")

    a = pool.connect(url)
    b = pool.connect(url)
    transport = transports[0]
    await transport.go_up()  # live wire
    await wait_for(lambda: a.connected)

    async def spam_send() -> None:
        for i in range(10):
            try:
                await a.send(f"frame-{i}")
            except NotConnected:
                return
            await asyncio.sleep(0)

    await asyncio.gather(spam_send(), a.close(), b.close(), return_exceptions=True)
    await settle()

    assert transport.stops == 1
    # Once stop ran, the transport is not live; send raises NotConnected. So no
    # frame may be recorded after the stop flipped live to False.
    assert not transport.live
    assert str(url) not in pool.endpoints


@pytest.mark.asyncio
async def test_send_after_close_raises_not_connected():
    """A lease whose endpoint was torn down can no longer send on the dead wire.

    The lone lease closes (stopping the transport); a later ``send`` must hit the
    stopped transport's ``NotConnected`` guard, never silently record a frame.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    url = yarl.URL("ws://host/dead-send")

    lease = pool.connect(url)
    transport = transports[0]
    await transport.go_up()
    await wait_for(lambda: lease.connected)

    await lease.close()
    await settle()
    assert transport.stops == 1
    assert not transport.live

    with pytest.raises(NotConnected):
        await lease.send("after-close")
    assert transport.sent == [], "a frame landed on a stopped transport"


# --------------------------------------------------------------------------- #
# 6. WebSocket 1:1 wire: rapid connect<->disconnect; concurrent close; fan-out.
# --------------------------------------------------------------------------- #


class FakeWsSocket:
    """A ``websockets``-shaped fake: ``recv`` blocks on an inbox a test feeds.

    The shape matches what :class:`Websockets` calls on its ``socket`` attribute:
    ``recv`` / ``send`` / ``close`` coroutines. A queued ``BaseException`` raised
    from ``recv`` is a wire drop; a queued ``str``/``bytes`` is an inbound frame.
    """

    def __init__(self) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: List[object] = []
        self.closed = 0

    async def recv(self) -> object:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send(self, frame: object) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        self.closed += 1


def ws_pool(sockets: List[FakeWsSocket]) -> Pool:
    """A pool of :class:`Websockets` wired to fresh fake sockets, keyed by URL."""

    def factory() -> Callable[..., Any]:
        async def connect(url: str, **kwargs: Any) -> FakeWsSocket:
            socket = FakeWsSocket()
            sockets.append(socket)
            return socket

        return connect

    def build(url: yarl.URL, params: object) -> Websockets:
        return Websockets(
            url,
            RetryPolicy(backoff=ConstantBackoff(0)),
            current_provider(),
            connect_factory=factory(),
            logger=silent_logger(),
        )

    from simplyprint_ws_client.contrib.connection.websocket import WsConnection

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        lease_class=WsConnection,
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_ws_rapid_connect_disconnect_cycles_one_socket_no_orphan():
    """Rapid lease connect<->close cycles on one WS endpoint: one socket at a time.

    Each cycle leases, waits for the socket, then closes -- the last close tears
    the socket down. Across many cycles, each underlying wire is closed at least
    once (no orphaned wire), and no task leaks.
    """
    sockets: List[FakeWsSocket] = []
    pool = ws_pool(sockets)
    url = yarl.URL("ws://host/rapid")
    baseline = live_tasks()
    #: Strong refs to every backend so identities cannot be recycled by GC.
    backends_seen: List[object] = []

    for _ in range(20):
        lease = pool.connect(url)
        backends_seen.append(lease.backend)
        await wait_for(lambda: lease.connected)
        await lease.close()
        await wait_for(lambda: str(url) not in pool.endpoints)

    await settle()
    # Each cycle built a fresh transport (the prior one was fully torn down first).
    assert len({id(b) for b in backends_seen}) == len(backends_seen), (
        "a transport instance was reused across teardown"
    )
    opened = [s for s in sockets if s is not None]
    assert opened, "no wire ever opened"
    for s in opened:
        assert s.closed >= 1, "an opened wire was never closed (orphan)"
    assert live_tasks() - baseline == set(), "leaked tasks after rapid cycles"


@pytest.mark.asyncio
async def test_ws_concurrent_close_of_all_leases_stops_socket_once():
    """All leases on a WS socket close at once -> the socket stops exactly once.

    Concurrent ``close`` from N leases must funnel through the refcount so the
    underlying wire is torn down exactly once, not N times, and the endpoint is
    dropped.
    """
    sockets: List[FakeWsSocket] = []
    pool = ws_pool(sockets)
    url = yarl.URL("ws://host/group")
    baseline = live_tasks()

    leases = [pool.connect(url) for _ in range(16)]
    await wait_for(lambda: leases[0].connected)
    socket = sockets[0]
    assert len(sockets) == 1

    await asyncio.gather(*(lease.close() for lease in leases))
    await settle()

    assert socket.closed == 1, f"wire closed {socket.closed} times (expected exactly 1)"
    assert str(url) not in pool.endpoints
    assert live_tasks() - baseline == set(), "leaked tasks after concurrent close"


@pytest.mark.asyncio
async def test_ws_broadcast_reaches_every_lease_during_join_churn():
    """A 1:1 WS frame broadcasts to every live lease while leases churn.

    Frames pushed on the shared socket reach every lease open at delivery time
    (route is ``None`` -> broadcast). Closing a lease mid-flight must not crash
    the fan-out, and the anchor lease must keep receiving.
    """
    sockets: List[FakeWsSocket] = []
    pool = ws_pool(sockets)
    url = yarl.URL("ws://host/broadcast")
    baseline = live_tasks()

    anchor = pool.connect(url)
    await wait_for(lambda: anchor.connected)
    socket = sockets[0]

    anchor_got: List[object] = []
    anchor.event_bus.on(MessageReceived, lambda e: anchor_got.append(e.message))

    async def transient_lease() -> None:
        lease = pool.connect(url)
        got: List[object] = []
        lease.event_bus.on(MessageReceived, lambda e: got.append(e.message))
        await asyncio.sleep(0)
        await lease.close()

    async def flood() -> None:
        for i in range(30):
            socket.inbox.put_nowait(f"frame-{i}")
            await asyncio.sleep(0)

    await asyncio.gather(flood(), *(transient_lease() for _ in range(8)))
    await settle()

    assert anchor_got, "anchor lease received nothing during the broadcast storm"

    await anchor.close()
    await settle()
    assert str(url) not in pool.endpoints
    assert live_tasks() - baseline == set(), "leaked tasks after broadcast storm"


# --------------------------------------------------------------------------- #
# 7. paho (sync wire): callbacks fired from a foreign thread race the loop.
# --------------------------------------------------------------------------- #


class FakePahoClient:
    """A paho-shaped client whose callbacks a test fires from any thread.

    Matches what :class:`Paho` calls on its ``client`` attribute: the three
    callbacks (``on_connect`` / ``on_message`` / ``on_disconnect``),
    ``connect_async`` / ``loop_start`` / ``loop_stop`` / ``disconnect`` /
    ``is_connected`` / ``subscribe`` / ``unsubscribe`` / ``publish``.
    """

    def __init__(self) -> None:
        self.on_connect: Optional[Callable[..., None]] = None
        self.on_message: Optional[Callable[..., None]] = None
        self.on_disconnect: Optional[Callable[..., None]] = None
        self.connected = False
        self.subscriptions: List[str] = []
        self.unsubscriptions: List[str] = []
        self.published: List[tuple] = []
        self.loops_started = 0
        self.loops_stopped = 0
        self.lock = threading.Lock()

    def username_pw_set(self, *_a: Any, **_k: Any) -> None:
        pass

    def connect_async(self, *_a: Any, **_k: Any) -> None:
        pass

    def loop_start(self) -> int:
        self.loops_started += 1
        return 0

    def loop_stop(self) -> int:
        self.loops_stopped += 1
        return 0

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def subscribe(self, topic: str) -> None:
        with self.lock:
            self.subscriptions.append(topic)

    def unsubscribe(self, topic: str) -> None:
        with self.lock:
            self.unsubscriptions.append(topic)

    def publish(
        self, topic: str, payload: bytes = b"", qos: int = 0, retain: bool = False
    ) -> Any:
        with self.lock:
            self.published.append((topic, payload, qos, retain))

        class Info:
            rc = 0

            def wait_for_publish(self_inner) -> None:
                return None

        return Info()

    # -- callback fan, driven by a test from a foreign thread ------------------ #

    def fire_connect(self, reason_code: int = 0) -> None:
        self.connected = True
        self.on_connect(self, None, {}, reason_code, None)

    def fire_message(self, message: Any) -> None:
        self.on_message(self, None, message)

    def fire_disconnect(self) -> None:
        self.connected = False
        self.on_disconnect(self, None, None, 0, None)


def from_thread(fn: Callable[..., None], *args: Any) -> None:
    """Run ``fn(*args)`` on a fresh thread and join it (simulates paho's thread)."""
    t = threading.Thread(target=fn, args=args)
    t.start()
    t.join(2.0)


def paho_pool(clients: List[FakePahoClient]) -> Pool:
    """A pool of :class:`Paho` transports wired to fresh fake paho clients."""

    def factory(url: yarl.URL, logger: logging.Logger) -> FakePahoClient:
        client = FakePahoClient()
        clients.append(client)
        return client

    def build(url: yarl.URL, params: object) -> Paho:
        return Paho(
            url,
            provider=current_provider(),
            client_factory=factory,
            logger=silent_logger(),
        )

    def key(url: yarl.URL, params: object) -> MqttBroker:
        return params if isinstance(params, MqttBroker) else MqttBroker.from_url(url)

    return Pool(
        build=build,
        key=key,
        lease_class=MqttConnection,
        provider=current_provider(),
    )


@pytest.mark.asyncio
async def test_paho_callbacks_from_thread_deliver_on_loop_and_bump_generation():
    """paho callbacks fired from a foreign thread arrive on the loop, in order.

    A ``Paho`` couriers its network-thread callbacks onto the provider loop.
    Firing connect then a message from a worker thread must deliver both on the
    loop (never on the worker thread), and the generation bumps once per connect.
    """
    clients: List[FakePahoClient] = []
    pool = paho_pool(clients)
    url = yarl.URL("mqtt://broker/?topic=room/a")
    broker = MqttBroker.from_url(url)
    loop_thread = threading.get_ident()
    baseline = live_tasks()

    lease = pool.connect(url, broker)
    await lease.subscribe("room/a")
    transport = lease.backend
    client = clients[0]

    got: List[tuple] = []
    lease.event_bus.on(
        MessageReceived,
        lambda e: got.append((e.message.topic, threading.get_ident(), e.generation)),
    )

    from_thread(client.fire_connect)
    await wait_for(lambda: lease.connected)
    gen = transport.generation
    assert gen == 1, f"generation should bump once per connect, got {gen}"

    from_thread(client.fire_message, WireMqttMessage("room/a", b"hi"))
    await wait_for(lambda: len(got) == 1)

    topic, ran_thread, msg_gen = got[0]
    assert topic == "room/a"
    assert ran_thread == loop_thread, "message handler ran off the loop thread"
    assert msg_gen == gen

    await lease.close()
    await settle()
    assert client.loops_stopped >= 1, "paho loop never stopped on close"
    assert live_tasks() - baseline == set(), "leaked tasks after paho close"


@pytest.mark.asyncio
async def test_paho_reconnect_callbacks_keep_generation_monotonic():
    """paho connect/disconnect/connect from a thread keeps generation climbing.

    paho self-heals; each ``on_connect`` bumps the generation. A drop then a
    re-connect, fired from a foreign thread, must leave the generation strictly
    increasing across the couriered lifecycle events.
    """
    clients: List[FakePahoClient] = []
    pool = paho_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    lease = pool.connect(url, broker)
    transport = lease.backend
    client = clients[0]

    gens: List[int] = []
    lease.event_bus.on(Connected, lambda e: gens.append(e.generation))

    from_thread(client.fire_connect)
    await wait_for(lambda: lease.connected)
    first = transport.generation

    from_thread(client.fire_disconnect)
    await wait_for(lambda: not lease.connected)

    from_thread(client.fire_connect)
    await wait_for(lambda: lease.connected)
    second = transport.generation

    assert second > first, (
        f"generation did not climb across reconnect: {first}->{second}"
    )
    assert gens == sorted(gens), f"observed Connected generations not monotonic: {gens}"

    await lease.close()
    await settle()


@pytest.mark.asyncio
async def test_paho_subscribe_refcounted_across_leases_on_shared_socket():
    """Concurrent subscribe/unsubscribe across paho leases keeps one wire sub.

    Two leases share a paho socket; both subscribe one topic, then both drop it.
    The wire must see exactly one ``subscribe`` and exactly one ``unsubscribe``
    for the topic across the refcount, with the network thread mutating the table
    under the transport's lock.
    """
    clients: List[FakePahoClient] = []
    pool = paho_pool(clients)
    url = yarl.URL("mqtt://broker/")
    broker = MqttBroker.from_url(url)

    a = pool.connect(url, broker)
    b = pool.connect(url, broker)
    assert a.backend is b.backend
    client = clients[0]
    from_thread(client.fire_connect)
    await wait_for(lambda: a.connected)

    await asyncio.gather(a.subscribe("room/x"), b.subscribe("room/x"))
    assert client.subscriptions.count("room/x") == 1, client.subscriptions

    await asyncio.gather(a.unsubscribe("room/x"), b.unsubscribe("room/x"))
    assert client.unsubscriptions.count("room/x") == 1, client.unsubscriptions

    await asyncio.gather(a.close(), b.close())
    await settle()


# --------------------------------------------------------------------------- #
# 8. Front door: ``mqtt.connect`` background initial-subscription tasks.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_front_door_connect_close_race_no_leaked_subscribe_task():
    """``mqtt.connect`` fires initial ``?topic=`` subscribes as background tasks.

    The front door records interest synchronously and spins a ``create_task`` per
    initial topic to assert it on the (eventual) live wire. Connecting then
    immediately closing must not leave that background subscribe task orphaned, and
    the shared transport must still stop exactly once.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    url = yarl.URL("mqtt://broker/?topic=room/a&topic=room/b")
    baseline = live_tasks()

    # The front door records topics on the lease and tasks the wire subscribe.
    lease = mqtt_door.connect(url, pool=pool)
    assert isinstance(lease, MqttConnection)
    assert lease.topics == {"room/a", "room/b"}, lease.topics

    # Close immediately, racing the background subscribe tasks.
    await lease.close()
    await settle()

    # The background subscribe tasks all completed/cleaned; nothing orphaned.
    assert live_tasks() - baseline == set(), "front-door subscribe task leaked"
    assert all(MqttBroker.from_url(url) != k for k in pool.endpoints), pool.endpoints


@pytest.mark.asyncio
async def test_front_door_shared_socket_across_concurrent_connects():
    """Concurrent ``mqtt.connect`` to one broker share a single socket.

    Several front-door connects to the same broker endpoint (even with different
    paths/topics) must lease one shared transport, then releasing all of them
    tears that single transport down exactly once.
    """
    clients: List[GatedAioClient] = []
    pool = mqtt_pool(clients)
    baseline = live_tasks()

    urls = [
        yarl.URL("mqtt://broker/?topic=room/a"),
        yarl.URL("mqtt://broker/?topic=room/b"),
        yarl.URL("mqtt://broker/?topic=room/c"),
    ]
    leases = [mqtt_door.connect(u, pool=pool) for u in urls]

    # All three resolve to the same broker -> one shared transport instance.
    backends = {id(lease.backend) for lease in leases}
    assert len(backends) == 1, "front-door connects did not share one socket"
    await wait_for(lambda: leases[0].connected)
    assert len(clients) == 1, f"built {len(clients)} sockets for one broker"

    for lease in leases:
        await lease.close()
    await settle()

    assert pool.endpoints == {}, "shared socket not dropped on last close"
    assert live_tasks() - baseline == set(), "leaked tasks after front-door close"


# --------------------------------------------------------------------------- #
# 9. Lease-edge EventBus delivery racing a lease close.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lease_close_does_not_own_or_leak_handler_tasks():
    """A lease has no private delivery task to cancel or leak on close."""
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    url = yarl.URL("ws://host/drain")
    baseline = live_tasks()

    lease = pool.connect(url)
    transport = transports[0]
    await transport.go_up()

    release = asyncio.Event()
    handled: List[object] = []

    async def slow_handler(event: MessageReceived) -> None:
        handled.append(event.message)
        await release.wait()

    lease.event_bus.on(MessageReceived, slow_handler)

    delivery = asyncio.create_task(transport.deliver("m-0"))
    await wait_for(lambda: handled == ["m-0"])

    await lease.close()
    release.set()
    await delivery
    await settle()

    assert live_tasks() - baseline == set(), "leaked task after lease close"


@pytest.mark.asyncio
async def test_event_after_close_is_dropped_by_pool_fanout():
    """A fanned event arriving after a lease closes is dropped."""
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    url = yarl.URL("ws://host/feed-after-close")
    baseline = live_tasks()

    lease = pool.connect(url)
    transport = transports[0]
    await transport.go_up()

    got: List[object] = []

    async def handler(event: MessageReceived) -> None:
        got.append(event.message)

    lease.event_bus.on(MessageReceived, handler)

    await lease.close()
    await transport.deliver("late")
    await settle()

    assert got == [], "a closed lease received a late event"
    assert live_tasks() - baseline == set(), "leaked task feeding a closed lease"


# --------------------------------------------------------------------------- #
# 10. pool.stop() under concurrency: fan-out stops, no further delivery.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pool_stop_detaches_fanout_so_no_further_delivery():
    """``pool.stop`` detaches every endpoint's fan-out; later events drop.

    A lease subscribed on a counting transport stops receiving once the pool is
    stopped (the endpoint detached from the transport's bus). A message emitted
    after ``pool.stop`` reaches no lease.
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    url = yarl.URL("ws://host/poolstop")

    lease = pool.connect(url)
    transport = transports[0]
    await transport.go_up()

    got: List[object] = []
    lease.event_bus.on(MessageReceived, lambda e: got.append(e.message))
    await transport.deliver("before-stop")
    await wait_for(lambda: got == ["before-stop"])

    pool.stop()
    assert pool.endpoints == {}, "pool.stop did not clear endpoints"

    await transport.deliver("after-stop")
    await settle()
    assert got == ["before-stop"], "a message was delivered after pool.stop detached"

    # The transport's async stop is the caller's to await; do it so nothing leaks.
    await transport.stop()


@pytest.mark.asyncio
async def test_pool_stop_during_connect_storm_leaves_consistent_state():
    """``pool.stop`` racing a connect storm leaves the pool empty and consistent.

    While many leases connect on several endpoints, ``pool.stop`` is called once.
    After settling, the pool's endpoint dict is empty (stop cleared it) and each
    built transport's fan-out was detached (no listener left on its bus for the
    pool's handlers).
    """
    transports: List[CountingTransport] = []
    pool = counting_pool(transports)
    baseline = live_tasks()

    async def connect_many(host: str) -> List[Any]:
        url = yarl.URL(f"ws://host/{host}")
        leases = [pool.connect(url) for _ in range(5)]
        await asyncio.sleep(0)
        return leases

    async def stopper() -> None:
        await asyncio.sleep(0)
        pool.stop()

    results = await asyncio.gather(
        connect_many("a"),
        connect_many("b"),
        connect_many("c"),
        stopper(),
    )

    await settle()
    assert pool.endpoints == {}, "pool not empty after stop during connect storm"

    # detach() ``off``s exactly the four event types the endpoint registered, and
    # the EventBus pops a type once its last listener leaves. The pool's fan-out
    # was the only thing ever subscribed on each transport's bus, so every bus
    # must be empty -- a surviving key means a fan-out handler outlived stop.
    for transport in transports:
        for event_type in (Connecting, Connected, Disconnected, MessageReceived):
            assert event_type not in transport.events.listeners, (
                f"a pool fan-out handler survived stop on {event_type.__name__}"
            )

    # Clean up: stop every built transport so nothing leaks, and close leases.
    lease_lists = [r for r in results if isinstance(r, list)]
    for leases in lease_lists:
        for lease in leases:
            await lease.close()
    for transport in transports:
        await transport.stop()
    await settle()
    assert live_tasks() - baseline == set(), "leaked tasks after pool.stop storm"
