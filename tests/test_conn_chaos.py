"""Chaos / failure-injection tests for the supervised reconnect engine.

These hammer :class:`~simplyprint_ws_client.wire.reconnect.Reconnecting`
through a single controllable fake wire (``ChaosWire``) whose four hooks --
``open`` / ``recv`` / ``write`` / ``aclose`` -- a test scripts call-by-call. No
real socket, broker, or wire library is touched; ``paho`` and ``aiomqtt`` are not
imported at all. The fake is the only thing under test's control: every retry,
drop, skipped frame, and teardown failure is injected through it.

What each region pins (the engine's contract, read straight off ``reconnect.py``):

* ``open`` raising then succeeding -> the loop retries and eventually connects,
  and the generation only bumps on the *successful* open;
* ``recv`` raising mid-stream -> the live attempt ends, ``Disconnected`` is tagged
  with a structured error preserving the native exception, and a fresh attempt
  brings the wire back;
* ``recv`` returning ``None`` -> the frame is skipped (no ``MessageReceived``), the
  stream keeps flowing, and no generation churn happens;
* ``write`` raising -> it propagates to the ``send`` caller AND trips the
  supervised attempt (a wire that fails a write is dead or dying);
* ``open`` hanging -> bounded by ``open_timeout``; a timeout is a failed attempt;
* ``aclose`` raising -> swallowed by ``teardown``; supervision survives it;
* ``RetryPolicy`` ``max_attempts`` / ``give_up_after`` exhaustion -> the loop stops,
  stays ``DISCONNECTED``, ``supervising()`` is ``False``, and a lease's
  ``ready()`` resolves ``False``;
* ``FatalError`` vs ``TransientError`` -> both keep retrying, both ride through to
  ``Disconnected.code``;
* generation bumps exactly once per *established* attempt, and ``Disconnected``
  carries the generation of the link that dropped;
* cancellation at every await point (open / recv / backoff sleep / stop) leaves no
  leaked task and emits no spurious ``Connected``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional

import pytest
import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff

from simplyprint_ws_client.wire.events import (
    Connected,
    Connecting,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.pool import Pool
from simplyprint_ws_client.wire.reconnect import Reconnecting
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
    FatalError,
    NotConnected,
    TransientError,
)


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
        await asyncio.sleep(0.002)
    raise AssertionError("condition not met in time")


def current_provider() -> EventLoopProvider:
    """A provider bound to the running test loop (the engine spawns its task here)."""
    return EventLoopProvider(loop=asyncio.get_event_loop())


def silent_logger() -> logging.Logger:
    """A logger that swallows the engine's debug/warning chatter during chaos."""
    logger = logging.getLogger("test.conn.chaos")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# The controllable fake wire.
# --------------------------------------------------------------------------- #


class ChaosWire(Reconnecting):
    """A :class:`Reconnecting` whose four hooks a test drives call-by-call.

    ``open`` consults ``open_script`` (a list of either ``None`` for success or a
    ``BaseException`` instance to raise) by attempt index, falling back to plain
    success once the script is exhausted. ``recv`` blocks on an inbox the test
    feeds: a queued ``BaseException`` is raised (a mid-stream drop), a queued
    ``None`` is returned verbatim (a skipped frame), anything else is the message.
    ``write`` raises if ``write_error`` is set. ``aclose`` raises if
    ``aclose_error`` is set (the engine must swallow it).

    Every hook records its call count and the at-entry generation so a test can
    assert the loop's ordering and that the generation bumps exactly once per
    established attempt.
    """

    def __init__(
        self,
        url: yarl.URL,
        *,
        open_script: Optional[List[Optional[BaseException]]] = None,
        write_error: Optional[BaseException] = None,
        aclose_error: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(url, **kwargs)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.open_script = list(open_script or [])
        self.write_error = write_error
        self.aclose_error = aclose_error

        self.opens = 0
        self.recvs = 0
        self.writes = 0
        self.closes = 0
        #: generation observed at the entry of each successful consume loop start.
        self.gen_at_connected: List[int] = []
        #: every payload handed to write (only those that did not raise).
        self.sent: List[object] = []

    async def open(self) -> None:
        index = self.opens
        self.opens += 1
        outcome = self.open_script[index] if index < len(self.open_script) else None
        if outcome is not None:
            raise outcome
        self.gen_at_connected.append(self.generation)

    async def recv(self) -> Optional[object]:
        self.recvs += 1
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def write(self, message: object) -> None:
        self.writes += 1
        if self.write_error is not None:
            raise self.write_error
        self.sent.append(message)

    async def aclose(self) -> None:
        self.closes += 1
        if self.aclose_error is not None:
            raise self.aclose_error


def make_wire(**kwargs: Any) -> ChaosWire:
    """A ChaosWire on the running loop with a zero backoff unless overridden."""
    policy = kwargs.pop("policy", RetryPolicy(backoff=ConstantBackoff(0)))
    return ChaosWire(
        yarl.URL("ws://chaos/x"),
        policy=policy,
        provider=current_provider(),
        logger=silent_logger(),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# open() transient failures then success.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_open_raises_transient_then_succeeds_bumps_generation_once():
    """Two failed opens then a good one: generation bumps exactly once, on success."""
    conns: List[int] = []
    downs: List[Any] = []
    wire = make_wire(
        open_script=[TransientError("nope-1"), TransientError("nope-2")],
    )
    wire.events.on(Connected, lambda e: conns.append(e.generation))
    wire.events.on(Disconnected, lambda e: downs.append((e.generation, e.code)))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)

        # Three opens total: two failures (no bump) then a success (one bump).
        assert wire.opens == 3
        assert wire.generation == 1
        assert conns == [1]  # Connected emitted exactly once, on the live gen

        # Each failed open emitted a Disconnected on the *un-bumped* generation 0,
        # tagged with the transient that ended it.
        assert len(downs) == 2
        assert downs[0][0] == 0 and isinstance(downs[0][1], TransientError)
        assert downs[1][0] == 0 and isinstance(downs[1][1], TransientError)
        assert str(downs[0][1]) == "nope-1"
        assert str(downs[1][1]) == "nope-2"
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_connecting_emitted_per_attempt_before_each_open():
    """A Connecting precedes every attempt, all on the pre-bump generation."""
    connectings: List[int] = []
    wire = make_wire(open_script=[TransientError("x"), TransientError("y")])
    wire.events.on(Connecting, lambda e: connectings.append(e.generation))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)
        # Attempt 1 + attempt 2 (both failed) + attempt 3 (success) each emit one
        # Connecting; the two before the live open carry generation 0.
        assert len(connectings) == 3
        assert connectings[0] == 0
        assert connectings[1] == 0
        assert connectings[2] == 0  # still 0: bump happens *after* open returns
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# recv() raising mid-stream.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recv_raises_midstream_drops_and_reconnects_fresh():
    """A mid-stream recv failure ends the attempt, tags Disconnected, reconnects."""
    downs: List[Any] = []
    ups: List[int] = []
    wire = make_wire()
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.events.on(Disconnected, lambda e: downs.append((e.generation, e.code)))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)

        boom = RuntimeError("mid-stream blowup")
        wire.inbox.put_nowait(boom)  # ends attempt 1 from inside consume()

        await wait_for(lambda: len(downs) == 1)
        # The Disconnected carries the dropped link's generation and wraps the
        # native error without losing the original exception object.
        assert downs[0][0] == 1
        assert isinstance(downs[0][1], TransientError)
        assert downs[0][1].transport_error is boom
        assert downs[0][1].__cause__ is boom

        await wait_for(lambda: wire.generation == 2 and wire.connected)
        assert ups == [1, 2]  # one bump per established attempt, no double-bump
        assert wire.closes >= 1  # the dropped wire was torn down
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_recv_raising_transient_and_fatal_both_retry_and_tag_code():
    """Both error families keep the loop retrying; both ride through to the code."""
    downs: List[Any] = []
    wire = make_wire()
    wire.events.on(Disconnected, lambda e: downs.append(e.code))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)

        transient = TransientError("blip")
        wire.inbox.put_nowait(transient)
        await wait_for(lambda: len(downs) == 1)
        await wait_for(lambda: wire.connected)  # retried back up

        fatal = FatalError("rejected")
        wire.inbox.put_nowait(fatal)
        await wait_for(lambda: len(downs) == 2)
        await wait_for(lambda: wire.connected)  # STILL retries despite "fatal"

        assert downs[0] is transient
        assert downs[1] is fatal
        assert wire.supervising() is True  # never gave up on either
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# recv() returning None -> skipped frames.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recv_none_is_skipped_no_message_no_generation_churn():
    """A None from recv emits no MessageReceived and does not end the attempt."""
    msgs: List[object] = []
    downs: List[Any] = []
    wire = make_wire()
    wire.events.on(MessageReceived, lambda e: msgs.append(e.message))
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)

        # Interleave skipped frames with a real one.
        wire.inbox.put_nowait(None)
        wire.inbox.put_nowait(None)
        wire.inbox.put_nowait("real-frame")
        wire.inbox.put_nowait(None)

        await wait_for(lambda: msgs == ["real-frame"])
        # Let the trailing None drain, then assert nothing else surfaced.
        await asyncio.sleep(0.02)

        assert msgs == ["real-frame"]  # the Nones produced no MessageReceived
        assert downs == []  # a skip is not a drop
        assert wire.generation == 1  # no churn from skipped frames
        assert wire.connected
        assert wire.recvs >= 4  # all four frames were pulled
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# write() raising. NEW CONTRACT (the keepalive-wedge fix): a wire that fails a
# write is dead or dying, so the failed send *trips* the attempt -- the loop
# tears down and reconnects -- instead of leaving a dead wire looking
# CONNECTED. The caller still gets the original exception.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_write_raising_propagates_to_send_caller_and_trips_the_link():
    """A write failure surfaces to send()'s caller AND ends the attempt: the
    supervised loop tears the dead wire down and brings up a fresh one."""
    boom = OSError("socket write failed")
    downs: List[Any] = []
    wire = make_wire(write_error=boom)
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)

        with pytest.raises(OSError) as excinfo:
            await wire.send("payload")
        assert excinfo.value is boom

        # The failed send trips the attempt: one Disconnected for the dead
        # wire, then a reconnect on the next generation.
        wire.write_error = None
        await wait_for(lambda: len(downs) == 1)
        await wait_for(lambda: wire.connected and wire.generation == 2)
        await wire.send("payload")
        assert wire.sent == ["payload"]
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_send_raises_not_connected_before_open_and_after_giveup():
    """send() refuses when there is no live wire (never started; gave up)."""
    wire = make_wire(open_script=[TransientError("x")] * 50)  # never connects
    wire.policy = RetryPolicy(backoff=ConstantBackoff(0), max_attempts=2)

    # Never started: no wire at all.
    with pytest.raises(NotConnected):
        await wire.send("nope")

    wire.start()
    try:
        await wait_for(lambda: not wire.supervising())
        # Gave up: still no live wire, so send still refuses.
        with pytest.raises(NotConnected):
            await wire.send("still-nope")
        assert wire.writes == 0  # write() never even reached
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# aclose() raising -> swallowed by teardown.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aclose_raising_is_swallowed_and_loop_survives():
    """aclose blowing up on teardown must not kill supervision."""
    downs: List[Any] = []
    wire = make_wire(aclose_error=RuntimeError("close exploded"))
    wire.events.on(Disconnected, lambda e: downs.append(e.code))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)

        drop = TransientError("drop-1")
        wire.inbox.put_nowait(drop)
        # teardown() calls aclose() which raises -- it must be swallowed and the
        # loop must reconnect to a fresh, live attempt.
        await wait_for(lambda: wire.generation == 2 and wire.connected)

        assert downs == [drop]  # the drop's own code, not the aclose error
        assert wire.closes >= 1  # aclose was attempted and its raise eaten
        assert wire.supervising() is True
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_aclose_raising_on_stop_still_stops_cleanly():
    """A teardown aclose failure during stop() must not leak the task."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    wire = make_wire(aclose_error=RuntimeError("close exploded"))
    wire.start()
    await wait_for(lambda: wire.connected)

    await wire.stop()  # cancels the task; teardown's aclose raises and is eaten
    await asyncio.sleep(0.02)

    assert wire.task is None
    assert not wire.supervising()
    assert wire.state is ConnectionState.DISCONNECTED
    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


@pytest.mark.asyncio
async def test_state_is_disconnected_after_stop_from_live_wire():
    """A bare stop() of a healthy, idle wire must settle state to DISCONNECTED.

    The lease handle reads ``state`` straight through to the transport, so a
    consumer inspecting a closed connection must never see a stale CONNECTED.
    """
    wire = make_wire()
    wire.start()
    await wait_for(lambda: wire.connected)
    assert wire.state is ConnectionState.CONNECTED

    await wire.stop()
    await asyncio.sleep(0.02)

    assert not wire.connected
    assert wire.state is ConnectionState.DISCONNECTED


# --------------------------------------------------------------------------- #
# RetryPolicy exhaustion: max_attempts and give_up_after.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_attempts_exhaustion_stays_disconnected():
    """max_attempts caps failed attempts; the loop gives up and stays down."""
    downs: List[Any] = []
    wire = make_wire(
        open_script=[TransientError(f"f{i}") for i in range(10)],
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=3),
    )
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: not wire.supervising())

        assert wire.gave_up is True
        assert wire.supervising() is False
        assert wire.connected is False
        assert wire.state is ConnectionState.DISCONNECTED
        # Exactly one Disconnected per failed attempt up to the cap.
        assert len(downs) == 3
        assert all(isinstance(d.code, TransientError) for d in downs)
        assert wire.opens == 3  # no further attempts after give-up

        # No live wire ever existed, so generation never advanced.
        assert wire.generation == 0
        assert all(d.generation == 0 for d in downs)

        # The give-up is sticky across loop turns.
        await asyncio.sleep(0.02)
        assert wire.opens == 3
        assert not wire.supervising()
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_give_up_after_deadline_exhaustion_stays_disconnected():
    """give_up_after caps wall-clock; the loop gives up and stays DISCONNECTED."""
    downs: List[Any] = []
    # backoff of 0.05s with a 0.12s deadline: the loop can only afford a couple
    # of retries before elapsed+delay crosses the deadline.
    wire = make_wire(
        open_script=[TransientError(f"f{i}") for i in range(100)],
        policy=RetryPolicy(backoff=ConstantBackoff(0.05), give_up_after=0.12),
    )
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: not wire.supervising(), timeout=3.0)

        assert wire.gave_up is True
        assert wire.connected is False
        assert wire.state is ConnectionState.DISCONNECTED
        assert len(downs) >= 1
        assert all(isinstance(d.code, TransientError) for d in downs)

        opens_at_giveup = wire.opens
        await asyncio.sleep(0.2)
        assert wire.opens == opens_at_giveup  # no more attempts after the deadline
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_giveup_resets_after_a_successful_connect():
    """A success resets the attempt bookkeeping so a later drop retries afresh."""
    # Fail twice, then succeed; with max_attempts=3 a naive (non-resetting) counter
    # would give up after the post-connect drop. The engine resets on connect, so
    # it must reconnect again instead of giving up.
    downs: List[Any] = []
    wire = make_wire(
        open_script=[TransientError("a"), TransientError("b")],  # then success
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=3),
    )
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)
        assert len(downs) == 2  # the two failed opens

        # Drop the live wire; because the attempt counter reset on connect, the
        # loop has its full budget again and reconnects (does not give up).
        wire.inbox.put_nowait(TransientError("post-connect drop"))
        await wait_for(lambda: wire.generation == 2 and wire.connected)
        assert wire.supervising() is True
        assert wire.gave_up is False
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# Generation accounting: exactly one bump per established attempt.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_one_generation_bump_per_established_attempt_over_many_drops():
    """Across many connect/drop cycles, generation == number of established links."""
    ups: List[int] = []
    downs: List[int] = []
    wire = make_wire()
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.events.on(Disconnected, lambda e: downs.append(e.generation))
    wire.start()
    try:
        established = 0
        for _ in range(6):
            await wait_for(
                lambda: wire.connected and wire.generation == established + 1
            )
            established += 1
            wire.inbox.put_nowait(TransientError(f"drop-{established}"))
            await wait_for(lambda: len(downs) == established)
            # Disconnected carries the generation of the link that just dropped.
            assert downs[-1] == established

        await wait_for(lambda: wire.connected and wire.generation == established + 1)
        # Connected fired once per established link, strictly increasing by one.
        assert ups == list(range(1, established + 2))
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# Cancellation at every await point: no leaked task, no spurious Connected.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel_while_blocked_in_open_leaves_no_task():
    """stop() while the loop is parked inside open() cancels cleanly."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    started = asyncio.Event()
    release = asyncio.Event()
    ups: List[int] = []

    class StuckOpenWire(ChaosWire):
        async def open(self) -> None:
            started.set()
            await release.wait()  # park here until cancelled
            await super().open()

    wire = StuckOpenWire(
        yarl.URL("ws://chaos/open"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
        logger=silent_logger(),
    )
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.start()
    await wait_for(started.is_set)

    await wire.stop()  # cancels the task mid-open
    await asyncio.sleep(0.02)

    assert wire.task is None
    assert not wire.connected
    assert not wire.supervising()
    assert ups == []  # never connected, so no spurious Connected
    assert wire.generation == 0
    # A stopped wire reports no live link in its lifecycle state. The loop set
    # CONNECTING just before parking in open(); stop() must not leave it there.
    assert wire.state is ConnectionState.DISCONNECTED

    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


@pytest.mark.asyncio
async def test_cancel_while_blocked_in_recv_leaves_no_task():
    """stop() while parked in recv() (a live, idle wire) cancels cleanly."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    wire = make_wire()
    wire.start()
    await wait_for(lambda: wire.connected and wire.recvs >= 1)  # parked in recv

    await wire.stop()
    await asyncio.sleep(0.02)

    assert wire.task is None
    assert not wire.connected
    assert not wire.supervising()
    # After a stop the wire is down: its state must reflect DISCONNECTED, not the
    # stale CONNECTED it held while parked in recv().
    assert wire.state is ConnectionState.DISCONNECTED

    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


@pytest.mark.asyncio
async def test_cancel_while_blocked_in_backoff_sleep_leaves_no_task():
    """stop() while parked in the inter-attempt backoff sleep cancels cleanly."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    # A long backoff and a forever-failing open: the loop spends its time asleep
    # between attempts, which is where stop() must be able to interrupt it.
    wire = make_wire(
        open_script=[TransientError("x")] * 100,
        policy=RetryPolicy(backoff=ConstantBackoff(30)),
    )
    wire.start()
    await wait_for(lambda: wire.opens >= 1)  # one failed attempt, now sleeping
    await wait_for(lambda: wire.state is ConnectionState.DISCONNECTED)

    await wire.stop()  # must wake from the 30s sleep, not hang
    await asyncio.sleep(0.02)

    assert wire.task is None
    assert not wire.supervising()

    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_quiet():
    """Calling stop() twice (and on an unstarted wire) is harmless."""
    wire = make_wire()
    await wire.stop()  # never started -> no-op
    assert wire.task is None

    wire.start()
    await wait_for(lambda: wire.connected)
    await wire.stop()
    await wire.stop()  # second stop is a no-op, must not raise
    assert wire.task is None
    assert not wire.supervising()


@pytest.mark.asyncio
async def test_start_is_idempotent_no_double_supervisor():
    """A second start() while already supervising does not spawn a second task."""
    wire = make_wire()
    wire.start()
    await wait_for(lambda: wire.connected)
    task = wire.task
    wire.start()  # idempotent: same task, no second supervisor
    assert wire.task is task
    try:
        await asyncio.sleep(0.02)
        assert wire.opens == 1  # exactly one live attempt, not two
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# Lease integration: ready() resolves False on a permanent give-up.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lease_ready_false_when_transport_gives_up():
    """A lease over a giving-up transport resolves ready() to False, not hang."""
    wire = make_wire(
        open_script=[TransientError(f"f{i}") for i in range(10)],
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=2),
    )
    pool = Pool(
        build=lambda url, params: wire,
        key=lambda url, params: "k",
        provider=current_provider(),
    )
    lease = pool.connect(wire.url)

    waiter = asyncio.ensure_future(lease.ready(timeout=2.0))

    assert await waiter is False  # terminal give-up resolves ready() False
    assert not wire.supervising()
    await wire.stop()


@pytest.mark.asyncio
async def test_lease_ready_true_then_engine_drop_does_not_resolve_false():
    """An ordinary drop (still supervising) must not resolve a pending ready()."""
    wire = make_wire()
    pool = Pool(
        build=lambda url, params: wire,
        key=lambda url, params: "k",
        provider=current_provider(),
    )
    lease = pool.connect(wire.url)
    assert await lease.ready(timeout=2.0) is True

    # A transient drop is NOT terminal; a fresh ready() must wait for the next
    # Connected rather than resolving False on the interim Disconnected.
    await wait_for(lambda: wire.connected)
    waiter = asyncio.ensure_future(lease.ready(timeout=2.0))
    await asyncio.sleep(0)
    wire.inbox.put_nowait(TransientError("interim drop"))

    assert await waiter is True  # resolved by the reconnect, not the drop
    await wire.stop()


# --------------------------------------------------------------------------- #
# teardown() is run on EVERY attempt -- including a FAILED open. A half-open
# wire must always be torn down before the next try.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_failed_open_is_still_torn_down_before_retry():
    """Each failed open() is followed by an aclose() of the half-open wire.

    The supervise loop's ``finally`` runs ``teardown()`` -> ``aclose()`` on every
    iteration, success or failure, so a wire that raised mid-open never leaks a
    half-built socket into the next attempt.
    """
    wire = make_wire(
        open_script=[TransientError("a"), TransientError("b")],  # then success
    )
    wire.start()
    try:
        await wait_for(lambda: wire.connected)
        # 3 opens (2 failed + 1 live). The two failures were each torn down; the
        # live one has NOT been closed yet (it is still up), so closes == 2.
        assert wire.opens == 3
        assert wire.closes == 2, "each failed open must be followed by an aclose"
        assert wire.connected
        assert wire.generation == 1
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_aclose_raising_on_a_failed_open_is_swallowed():
    """aclose() blowing up while tearing down a *failed* open must not stop the loop.

    This exercises the teardown path on the failure branch (no live wire ever
    existed), distinct from tearing down an established link that later dropped.
    """
    wire = make_wire(
        open_script=[TransientError("a"), TransientError("b")],  # then success
        aclose_error=RuntimeError("teardown of half-open exploded"),
    )
    wire.start()
    try:
        # Despite every failed open's teardown raising, the loop reaches a live
        # connection -- the aclose error is eaten on the failure branch too.
        await wait_for(lambda: wire.connected and wire.generation == 1)
        assert wire.opens == 3
        assert wire.closes == 2
        assert wire.supervising() is True
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# Any exception kind ends an attempt and rides through Disconnected.code as a
# structured transport error that preserves the native exception.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_plain_exception_from_open_retries_and_wraps_native_error():
    """A bare Exception from open() retries and is preserved under the wrapper."""
    downs: List[Any] = []
    boom = ValueError("not a typed wire error at all")
    wire = make_wire(open_script=[boom])  # one plain failure, then success
    wire.events.on(Disconnected, lambda e: downs.append(e.code))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)
        assert wire.opens == 2  # retried past the plain Exception
        assert wire.generation == 1
        assert len(downs) == 1
        assert isinstance(downs[0], TransientError)
        assert downs[0].transport_error is boom
        assert downs[0].__cause__ is boom
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_plain_exception_from_recv_retries_and_wraps_native_error():
    """A bare Exception from recv() drops, reconnects, and preserves the native error."""
    downs: List[Any] = []
    wire = make_wire()
    wire.events.on(Disconnected, lambda e: downs.append(e.code))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)
        boom = KeyError("weird mid-stream failure")
        wire.inbox.put_nowait(boom)
        await wait_for(lambda: len(downs) == 1)
        assert isinstance(downs[0], TransientError)
        assert downs[0].transport_error is boom
        assert downs[0].__cause__ is boom
        await wait_for(lambda: wire.generation == 2 and wire.connected)
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# A clean stop() of a live wire emits NO Disconnected (the loop breaks before
# announcing a drop). Only a *drop* -- never an intentional teardown -- speaks.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_clean_stop_emits_no_disconnected():
    """stop() on a healthy live wire tears down quietly: no Disconnected event."""
    downs: List[Any] = []
    ups: List[int] = []
    wire = make_wire()
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    await wait_for(lambda: wire.connected)

    await wire.stop()
    await asyncio.sleep(0.02)

    assert ups == [1]  # came up exactly once
    assert downs == []  # an intentional stop is silent -- no spurious drop
    assert wire.closes == 1  # but the wire WAS torn down
    assert wire.state is ConnectionState.DISCONNECTED


# --------------------------------------------------------------------------- #
# Connecting's generation across a drop: a recovery attempt's Connecting still
# carries the *previous* (dropped) generation, since the bump is post-open.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_connecting_after_a_drop_carries_the_dropped_generation():
    """The Connecting of a recovery attempt rides the just-dropped generation."""
    connectings: List[int] = []
    wire = make_wire()
    wire.events.on(Connecting, lambda e: connectings.append(e.generation))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)
        # First Connecting was on gen 0 (pre-first-open).
        assert connectings == [0]

        wire.inbox.put_nowait(TransientError("drop"))
        await wait_for(lambda: wire.generation == 2 and wire.connected)
        # The recovery's Connecting fired before the bump-to-2, so it carries the
        # generation of the link that just died: 1.
        assert connectings == [0, 1]
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# MessageReceived always carries the live link's generation, and a drop after
# some messages tags Disconnected with that same generation.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_message_received_carries_live_generation_and_drop_matches():
    """Every MessageReceived is tagged with the generation it arrived on."""
    msgs: List[Any] = []
    downs: List[int] = []
    wire = make_wire()
    wire.events.on(MessageReceived, lambda e: msgs.append((e.generation, e.message)))
    wire.events.on(Disconnected, lambda e: downs.append(e.generation))
    wire.start()
    try:
        await wait_for(lambda: wire.generation == 1 and wire.connected)
        wire.inbox.put_nowait("m1")
        wire.inbox.put_nowait("m2")
        await wait_for(lambda: len(msgs) == 2)
        assert msgs == [(1, "m1"), (1, "m2")]

        wire.inbox.put_nowait(TransientError("drop after messages"))
        await wait_for(lambda: len(downs) == 1)
        assert downs == [1]  # the drop is on the same epoch the messages rode

        await wait_for(lambda: wire.generation == 2 and wire.connected)
        wire.inbox.put_nowait("m3")
        await wait_for(lambda: len(msgs) == 3)
        assert msgs[-1] == (2, "m3")  # next epoch's messages carry the new gen
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# max_attempts boundary: 1 means "give up after the first failed attempt".
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_attempts_one_gives_up_after_a_single_failure():
    """max_attempts=1 stops after exactly one failed open with one Disconnected."""
    downs: List[Any] = []
    wire = make_wire(
        open_script=[TransientError("only-try")] * 5,
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=1),
    )
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: not wire.supervising())
        assert wire.opens == 1  # gave up immediately after the first failure
        assert wire.gave_up is True
        assert wire.connected is False
        assert wire.generation == 0  # never established
        assert len(downs) == 1
        assert isinstance(downs[0].code, TransientError)
        assert downs[0].generation == 0
        assert wire.state is ConnectionState.DISCONNECTED
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# After a permanent give-up, ready() resolves False even when called fresh
# (the wait must not hang against a dead supervisor), and state is read True.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lease_ready_false_when_called_after_giveup_already_happened():
    """A ready() entered *after* the engine already gave up resolves False at once."""
    wire = make_wire(
        open_script=[TransientError(f"f{i}") for i in range(10)],
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=2),
    )
    pool = Pool(
        build=lambda url, params: wire,
        key=lambda url, params: "k",
        provider=current_provider(),
    )
    lease = pool.connect(wire.url)
    await wait_for(lambda: not wire.supervising())

    # The give-up already happened and no future Connected/Disconnected will fire.
    # ready() must short-circuit on the dead supervisor rather than block forever.
    assert await lease.ready(timeout=1.0) is False
    await wire.stop()


# --------------------------------------------------------------------------- #
# Cancellation parked in teardown's aclose: stop() during a slow aclose must
# still cancel cleanly and leave no leaked task.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel_while_blocked_in_teardown_aclose_leaves_no_task():
    """stop() while the loop is parked inside a slow aclose cancels cleanly."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    in_aclose = asyncio.Event()
    release = asyncio.Event()

    class SlowCloseWire(ChaosWire):
        async def aclose(self) -> None:
            self.closes += 1
            in_aclose.set()
            await release.wait()  # park inside teardown until cancelled

    wire = SlowCloseWire(
        yarl.URL("ws://chaos/aclose"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
        logger=silent_logger(),
    )
    wire.start()
    await wait_for(lambda: wire.connected and wire.recvs >= 1)

    # Drop the wire so the loop enters teardown and parks in the slow aclose.
    wire.inbox.put_nowait(TransientError("drop into slow aclose"))
    await wait_for(in_aclose.is_set)

    await wire.stop()  # cancels the task while it is awaiting inside aclose
    await asyncio.sleep(0.02)

    assert wire.task is None
    assert not wire.supervising()
    assert wire.state is ConnectionState.DISCONNECTED

    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


# --------------------------------------------------------------------------- #
# A drop storm: many rapid-fire drops, each producing exactly one established
# link and one Disconnected, with generations strictly contiguous (no skips,
# no double-bumps, no lost Disconnected).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_drop_storm_generations_stay_contiguous_and_balanced():
    """Twenty connect/drop cycles produce 20 strictly increasing generations."""
    ups: List[int] = []
    downs: List[int] = []
    wire = make_wire()
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.events.on(Disconnected, lambda e: downs.append(e.generation))
    wire.start()
    try:
        cycles = 20
        for n in range(1, cycles + 1):
            await wait_for(lambda: wire.connected and wire.generation == n)
            wire.inbox.put_nowait(TransientError(f"storm-{n}"))
            await wait_for(lambda: len(downs) == n)
            assert downs[-1] == n  # each drop tagged with the gen that died

        await wait_for(lambda: wire.connected and wire.generation == cycles + 1)
        # Connected fired once per established link: 1..cycles+1, strictly +1.
        assert ups == list(range(1, cycles + 2))
        # Disconnected fired once per dropped link: 1..cycles, strictly +1.
        assert downs == list(range(1, cycles + 1))
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# write() failing persistently: every failed send trips its attempt, so the
# wire cycles through supervised reconnects -- it never wedges CONNECTED on a
# dead socket and never stops supervising. The caller always gets a typed
# refusal: the write's own error on a live wire, NotConnected between attempts.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_repeated_write_failures_cycle_supervised_reconnects():
    """Persistently failing send()s churn supervised reconnects, never a wedge."""
    wire = make_wire(write_error=ConnectionResetError("write keeps failing"))
    wire.start()
    try:
        await wait_for(lambda: wire.connected)
        for _ in range(25):
            with pytest.raises((ConnectionResetError, NotConnected)):
                await wire.send(b"x")
            await asyncio.sleep(0)  # let the trip/reconnect interleave
        assert wire.supervising()  # the loop never gives up on write failures
        assert wire.sent == []  # ...and no failed send was reported delivered

        # The moment writes heal, the supervised wire settles and delivers.
        wire.write_error = None
        await wait_for(lambda: wire.connected)
        await wire.send(b"ok")
        assert wire.sent == [b"ok"]
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# stop() entered concurrently with a fresh start() / a connect in flight: the
# wire settles DISCONNECTED with no leaked task regardless of timing.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stop_immediately_after_start_before_connect_leaks_nothing():
    """start() then an immediate stop() (before the first open completes) is clean."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowOpenWire(ChaosWire):
        async def open(self) -> None:
            started.set()
            await release.wait()
            await super().open()

    wire = SlowOpenWire(
        yarl.URL("ws://chaos/race"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
        logger=silent_logger(),
    )
    ups: List[int] = []
    wire.events.on(Connected, lambda e: ups.append(e.generation))
    wire.start()
    await wait_for(started.is_set)  # parked mid-open, never connected

    await wire.stop()  # cancel before the open could ever complete
    await asyncio.sleep(0.02)

    assert ups == []  # no Connected ever emitted
    assert wire.generation == 0
    assert wire.task is None
    assert not wire.supervising()
    assert wire.state is ConnectionState.DISCONNECTED

    after = {t for t in asyncio.all_tasks() if not t.done()}
    leaked = after - before - {asyncio.current_task()}
    assert leaked == set(), f"leaked tasks: {leaked}"


# --------------------------------------------------------------------------- #
# A restart after a permanent give-up: start() clears the give-up flag and the
# supervisor runs again (the policy bookkeeping is per-run, not sticky).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_start_after_giveup_resumes_supervision():
    """After giving up, a fresh start() re-arms the loop and can reconnect."""
    wire = make_wire(
        open_script=[TransientError("a"), TransientError("b")],
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=2),
    )
    wire.start()
    try:
        await wait_for(lambda: not wire.supervising())
        assert wire.gave_up is True
        assert wire.opens == 2

        # The next attempt index (>= len(open_script)) succeeds, so a fresh start()
        # must clear gave_up and bring the wire up.
        wire.start()
        await wait_for(lambda: wire.connected)
        assert wire.gave_up is False
        assert wire.supervising() is True
        assert wire.generation == 1
    finally:
        await wire.stop()


@pytest.mark.asyncio
async def test_generation_is_monotonic_across_giveup_then_restart():
    """A connect, drop-to-giveup, then restart never rewinds the generation epoch.

    ``generation`` is a monotonic connection epoch -- it must keep counting up
    across a permanent give-up and a later restart, never reset to a value a stale
    waiter could mistake for a fresh link.
    """
    # Open #0 succeeds; every later open fails. With max_attempts=2 the post-drop
    # recovery exhausts the budget and gives up after one live link (gen 1).
    wire = make_wire(
        open_script=[None, TransientError("d1"), TransientError("d2")],
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=2),
    )
    wire.start()
    try:
        await wait_for(lambda: wire.connected and wire.generation == 1)
        wire.inbox.put_nowait(TransientError("drop into giveup"))
        await wait_for(lambda: not wire.supervising())
        assert wire.generation == 1  # the give-up did not advance or rewind it

        # Allow success again and restart: the next epoch must be 2, not 1.
        wire.open_script = [None]
        wire.opens = 0  # the success branch reads index 0 -> None (connects)
        wire.start()
        await wait_for(lambda: wire.connected)
        assert wire.generation == 2  # strictly continues from the prior epoch
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# ready(timeout=None) against a permanent give-up must resolve False, not hang
# forever waiting on a Connected that will never come.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lease_ready_no_timeout_resolves_false_on_giveup():
    """ready() with no timeout still resolves False when the engine gives up."""
    wire = make_wire(
        open_script=[TransientError(f"f{i}") for i in range(10)],
        policy=RetryPolicy(backoff=ConstantBackoff(0), max_attempts=2),
    )
    pool = Pool(
        build=lambda url, params: wire,
        key=lambda url, params: "k",
        provider=current_provider(),
    )
    lease = pool.connect(wire.url)

    waiter = asyncio.ensure_future(lease.ready())  # timeout=None

    # An outer guard ensures a regression that hangs fails loudly instead of
    # stalling the suite forever.
    assert await asyncio.wait_for(waiter, timeout=3.0) is False
    assert not wire.supervising()
    await wire.stop()


# --------------------------------------------------------------------------- #
# Closing a lease while one of its async handlers is parked mid-coroutine does not
# leak a lease-owned delivery task; delivery is the caller's awaited EventBus emit.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_lease_close_while_handler_is_parked_no_leak():
    """The lease owns no delivery task that can survive close()."""
    before = {t for t in asyncio.all_tasks() if not t.done()}
    wire = make_wire()
    pool = Pool(
        build=lambda url, params: wire,
        key=lambda url, params: "k",
        provider=current_provider(),
    )
    lease = pool.connect(wire.url)

    handler_entered = asyncio.Event()

    async def parked_handler(_: MessageReceived) -> None:
        handler_entered.set()
        await asyncio.sleep(3600)

    lease.event_bus.on(MessageReceived, parked_handler)
    try:
        await wait_for(lambda: wire.connected)
        wire.inbox.put_nowait("msg")
        await wait_for(handler_entered.is_set)

        await lease.close()
        await asyncio.sleep(0.02)

        after = {t for t in asyncio.all_tasks() if not t.done()}
        leaked = after - before - {asyncio.current_task(), wire.task}
        assert leaked == set(), f"leaked delivery task: {leaked}"
    finally:
        await wire.stop()


# --------------------------------------------------------------------------- #
# open() is bounded at the supervise level: a hung connect is a failed attempt
# (backoff, retry), never CONNECTING forever.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hung_open_times_out_and_the_next_attempt_proceeds():
    """A blocked open() ends after open_timeout with a TransientError-coded
    Disconnected, the half-open wire is torn down, and the loop retries."""
    release = asyncio.Event()
    opens = []

    class HungOpenWire(ChaosWire):
        async def open(self) -> None:
            opens.append(self.generation)
            if len(opens) == 1:
                await release.wait()  # first attempt hangs
            await super().open()

    downs: List[Any] = []
    wire = HungOpenWire(
        yarl.URL("ws://chaos/x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
        logger=silent_logger(),
    )
    wire.open_timeout = 0.05
    wire.events.on(Disconnected, lambda e: downs.append(e))
    wire.start()
    try:
        await wait_for(lambda: wire.connected and wire.generation == 1)
        assert len(opens) == 2  # attempt 1 timed out, attempt 2 connected
        assert len(downs) == 1
        assert isinstance(downs[0].code, TransientError)
        assert "open timed out" in str(downs[0].code)
        assert wire.closes >= 1  # the half-open attempt was torn down
    finally:
        release.set()
        await wire.stop()


@pytest.mark.asyncio
async def test_open_timeout_none_keeps_open_unbounded():
    """open_timeout=None preserves the legacy unbounded open()."""
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowOpenWire(ChaosWire):
        async def open(self) -> None:
            started.set()
            await release.wait()
            await super().open()

    wire = SlowOpenWire(
        yarl.URL("ws://chaos/x"),
        policy=RetryPolicy(backoff=ConstantBackoff(0)),
        provider=current_provider(),
        logger=silent_logger(),
    )
    wire.open_timeout = None
    wire.start()
    try:
        await asyncio.wait_for(started.wait(), 2.0)
        await asyncio.sleep(0.1)  # far past any small bound: still CONNECTING
        assert wire.state is ConnectionState.CONNECTING
        release.set()
        await wait_for(lambda: wire.connected)
    finally:
        release.set()
        await wire.stop()
