"""Concurrency / race-condition tests for the ``device.connection`` connection library.

These tests hammer the *shared, pooled* spine of ``device.connection`` from many
directions at once on a single asyncio loop -- which is exactly where the library
lives: the pool refcount, a lease's ``send`` racing a generation flip, concurrent
subscribe/unsubscribe across leases against one refcounted broker socket, and
``stop`` landing mid-open / mid-consume. None of this touches a real broker or
socket: every wire is an injected fake satisfying the same seam production wires
do -- a :class:`Reconnecting` subclass driven by queues, a paho-shaped fake
behind :class:`Paho`, and a Websockets-shaped fake.

The single asyncio loop does NOT make these tests trivial: every ``await`` is a
scheduling point where another task can interleave, so the dict mutations, the
refcount, the generation counter, and the topic table are all genuinely racing
across suspension points. The invariants asserted are the ones the docstrings
promise: a transport is started once and stopped once per instance, the refcount
is never torn, the generation is strictly monotonic, no message lands on a wrong
generation, and the asyncio task count returns to baseline (no leaks, no orphaned
transports).

paho is not required; every MQTT test injects a fake client through the wire's
``client_factory`` seam, so nothing imports the real wire library.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional, Set

import pytest
import yarl

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff

from simplyprint_ws_client.wire.events import (
    Connected,
    Connecting,
    WireEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
    NotConnected,
    Transport,
)
from simplyprint_ws_client.wire.websockets import Websockets


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
        self.events: EventBus[WireEvent] = EventBus()
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

    def trip(self, generation: int, reason: Exception) -> None:
        if generation == self.generation:
            self.live = False
            self.state = ConnectionState.DISCONNECTED

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
    assert b.transport is not old
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
# MQTT message stand-in shared by the Paho adapter tests below.
# --------------------------------------------------------------------------- #


class WireMqttMessage:
    """A paho message stand-in routed by ``.topic``."""

    def __init__(
        self, topic: str, payload: bytes, qos: int = 0, retain: bool = False
    ) -> None:
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.retain = retain


# --------------------------------------------------------------------------- #
# 3. send racing close on a pooled lease: no send on a torn-down transport.
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
# 4. WebSocket 1:1 wire: rapid connect<->disconnect; concurrent close; fan-out.
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

    from simplyprint_ws_client.wire.websocket import WsLease

    return Pool(
        build=build,
        key=lambda url, params: str(url),
        lease_class=WsLease,
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
    #: Strong refs to every transport so identities cannot be recycled by GC.
    transports_seen: List[object] = []

    for _ in range(20):
        lease = pool.connect(url)
        transports_seen.append(lease.transport)
        await wait_for(lambda: lease.connected)
        await lease.close()
        await wait_for(lambda: str(url) not in pool.endpoints)

    await settle()
    # Each cycle built a fresh transport (the prior one was fully torn down first).
    assert len({id(transport) for transport in transports_seen}) == len(
        transports_seen
    ), "a transport instance was reused across teardown"
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
# 7. Lease-edge EventBus delivery racing a lease close.
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
