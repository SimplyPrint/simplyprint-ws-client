"""Benchmark the new contrib.connection stack against main's hand-rolled ws_protocol loop.

Run with the library venv (never uv)::

    .venv/bin/python scripts/bench_conn.py

It compares three things, each with fake in-process wires (no network):

1. inbound throughput -- messages/sec driven through to a trivial handler
   (NEW: transport bus -> pool fan-out -> lease -> handler;
    MAIN: Connection._loop -> poll() -> event_bus -> handler);
2. reconnect cost -- wall time for 1000 drop/reconnect cycles;
3. peak memory -- tracemalloc peak under a 100k-message flood with a slow handler.

The two stacks are genuinely different shapes, so a perfectly fair single number
is not always possible; every place the comparison is uneven is called out in the
printed methodology caveats and, where it matters, measured both ways.

MAIN's Connection is imported from a detached worktree at /tmp/spwc-main (set up by
the caller) so this branch's rewritten connection.py never shadows it. MAIN's
relative imports (.events, .messages, ..config, ..shared) are bound to the LIVE
installed package -- those modules are byte-identical across the two trees, so this
loads one engine, not two.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import sys
import time
import tracemalloc
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import yarl

MAIN_CONNECTION_PATH = (
    "/tmp/spwc-main/simplyprint_ws_client/core/ws_protocol/connection.py"
)

INBOUND_MESSAGES = 200_000
RECONNECT_CYCLES = 1_000
FLOOD_MESSAGES = 100_000


# --------------------------------------------------------------------------- #
# Load MAIN's Connection from the worktree, bound to the live package.
# --------------------------------------------------------------------------- #


def load_main_connection() -> Any:
    """Import main's ws_protocol connection module from the worktree in isolation."""
    spec = importlib.util.spec_from_file_location(
        "simplyprint_ws_client.core.ws_protocol.main_connection",
        MAIN_CONNECTION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load main connection from {MAIN_CONNECTION_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Bind relative imports (.events / .messages / ..config / ..shared) to the
    # live, installed package -- those files are identical across the two trees.
    module.__package__ = "simplyprint_ws_client.core.ws_protocol"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Fakes for MAIN: an aiohttp-shaped session / websocket with no network.
# --------------------------------------------------------------------------- #


class FakeFrame:
    """An aiohttp-shaped inbound frame: a type and a data payload."""

    __slots__ = ("type", "data")

    def __init__(self, kind: Any, data: Any) -> None:
        self.type = kind
        self.data = data


class FloodWebSocket:
    """A fake aiohttp ws that yields a fixed number of text frames, then blocks.

    ``receive`` hands back ``total`` identical text frames carrying ``payload``
    (valid ServerMsg JSON), then waits forever -- so MAIN's poll loop streams the
    whole flood and never sees a close (the bench cancels the loop when done).
    """

    def __init__(self, text_type: Any, payload: str, total: int) -> None:
        self.text_type = text_type
        self.payload = payload
        self.total = total
        self.sent = 0
        self.closed = False
        self.close_code: Optional[int] = None

    async def receive(self) -> FakeFrame:
        if self.sent < self.total:
            self.sent += 1
            return FakeFrame(self.text_type, self.payload)
        # Flood exhausted: park until the loop is cancelled.
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send_str(self, data: str) -> None:
        pass

    async def close(self, code: int = 0, message: bytes = b"") -> None:
        self.closed = True
        self.close_code = code


class DropWebSocket:
    """A fake aiohttp ws whose first ``receive`` reports a closed connection.

    MAIN's ``poll`` raises ``ConnectionResetError`` on a CLOSE frame, which drives
    one reconnect cycle. Each new connect builds a fresh one of these.
    """

    def __init__(self, close_type: Any) -> None:
        self.close_type = close_type
        self.closed = False
        self.close_code: Optional[int] = None

    async def receive(self) -> FakeFrame:
        return FakeFrame(self.close_type, None)

    async def send_str(self, data: str) -> None:
        pass

    async def close(self, code: int = 0, message: bytes = b"") -> None:
        self.closed = True
        self.close_code = code


class FakeSession:
    """A fake aiohttp ClientSession whose ws_connect hands back a scripted ws."""

    def __init__(self, make_ws: Callable[[], Any]) -> None:
        self.make_ws = make_ws
        self.closed = False

    async def ws_connect(self, url: Any, **kwargs: Any) -> Any:
        return self.make_ws()

    async def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Helpers for NEW: a backoff that never sleeps.
# --------------------------------------------------------------------------- #


def new_imports() -> Any:
    """Import the new conn pieces (kept here so the top of the file stays light)."""
    from simplyprint_ws_client.contrib.connection.connection import WsConnection
    from simplyprint_ws_client.contrib.connection.events import Connected, MessageReceived
    from simplyprint_ws_client.contrib.connection.messages import QoS
    from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
    from simplyprint_ws_client.contrib.connection.pool import Pool
    from simplyprint_ws_client.contrib.connection.reconnect import Reconnecting
    from simplyprint_ws_client.contrib.connection.transport import TransientError, WsTransport
    from simplyprint_ws_client.contrib.connection.websocket import WsTextMessage

    return {
        "WsConnection": WsConnection,
        "Connected": Connected,
        "MessageReceived": MessageReceived,
        "QoS": QoS,
        "RetryPolicy": RetryPolicy,
        "Pool": Pool,
        "Reconnecting": Reconnecting,
        "TransientError": TransientError,
        "WsTransport": WsTransport,
        "WsTextMessage": WsTextMessage,
    }


class ZeroBackoff:
    """A backoff that never sleeps -- isolates reconnect code-path cost from delay."""

    def delay(self) -> float:
        return 0.0

    def reset(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Metric 1: inbound throughput.
# --------------------------------------------------------------------------- #


async def new_inbound(api: Any, total: int, parse: bool) -> float:
    """Drive ``total`` frames through transport -> pool -> lease -> handler.

    With ``parse`` the handler runs the same pydantic ServerMsg validation MAIN
    does, so the two stacks can be compared on equal work as well as raw machinery.
    """
    from simplyprint_ws_client.core.ws_protocol.messages import ServerMsg

    Reconnecting = api["Reconnecting"]
    WsTransport = api["WsTransport"]
    MessageReceived = api["MessageReceived"]
    payload = json.dumps({"type": "pong"})

    class FloodTransport(WsTransport, Reconnecting):
        def __init__(self, url: yarl.URL) -> None:
            super().__init__(url)
            self.sent = 0
            self.done = asyncio.Event()

        async def open(self) -> None:
            pass

        async def recv(self) -> Optional[object]:
            if self.sent < total:
                self.sent += 1
                return payload
            self.done.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def write(self, message: object) -> None:
            pass

        async def aclose(self) -> None:
            pass

    pool = api["Pool"](
        build=lambda url, params: FloodTransport(url),
        key=lambda url, params: str(url),
        lease_class=api["WsConnection"],
    )
    lease = pool.connect(yarl.URL("ws://bench/"))
    count = 0

    def handler(event: object) -> None:
        nonlocal count
        if parse:
            ServerMsg.model_validate_json(payload)
        count += 1

    lease.event_bus.on(MessageReceived, handler)

    transport = lease.backend
    start = time.perf_counter()
    transport.start()
    await transport.done.wait()
    elapsed = time.perf_counter() - start
    await transport.stop()
    assert count == total, (count, total)
    return total / elapsed


async def main_inbound(main_mod: Any, total: int) -> float:
    """Drive ``total`` frames through MAIN's _loop -> poll() -> event_bus -> handler."""
    from aiohttp import WSMsgType

    from simplyprint_ws_client.core.ws_protocol.events import ConnectionIncomingEvent

    payload = json.dumps({"type": "pong"})
    ws = FloodWebSocket(WSMsgType.TEXT, payload, total)
    session = FakeSession(lambda: ws)

    conn = main_mod.Connection(session=session)
    conn.use_running_loop()

    count = 0
    finished = asyncio.get_running_loop().create_future()

    async def handler(*args: Any, **kwargs: Any) -> None:
        nonlocal count
        count += 1
        if count == total and not finished.done():
            finished.set_result(None)

    conn.event_bus.on(ConnectionIncomingEvent, handler)

    start = time.perf_counter()
    await conn.connect()
    await finished
    elapsed = time.perf_counter() - start

    conn.stop()
    # Cancel the parked loop task.
    task = conn._loop_task.task
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return total / elapsed


# --------------------------------------------------------------------------- #
# Metric 2: reconnect cost.
# --------------------------------------------------------------------------- #


async def new_reconnect(api: Any, cycles: int) -> float:
    """Time ``cycles`` open/drop cycles on the real Reconnecting supervise loop.

    Completion is read off ``transport.generation`` (bumped once per successful
    open) rather than an async-event handler -- the same generation-counter signal
    MAIN is measured by, so both sides count the identical thing.
    """
    Reconnecting = api["Reconnecting"]
    WsTransport = api["WsTransport"]
    TransientError = api["TransientError"]
    RetryPolicy = api["RetryPolicy"]

    class FlappyTransport(WsTransport, Reconnecting):
        async def open(self) -> None:
            pass

        async def recv(self) -> Optional[object]:
            # Every connection drops on its first recv -> one reconnect cycle.
            raise TransientError("drop")

        async def write(self, message: object) -> None:
            pass

        async def aclose(self) -> None:
            pass

    policy = RetryPolicy(backoff=ZeroBackoff())
    transport = FlappyTransport(yarl.URL("ws://bench/"), policy)

    start = time.perf_counter()
    transport.start()
    while transport.generation < cycles:
        await asyncio.sleep(0)
    elapsed = time.perf_counter() - start
    await transport.stop()
    assert transport.generation >= cycles, transport.generation
    return elapsed


async def main_reconnect(main_mod: Any, cycles: int) -> float:
    """Time ``cycles`` reconnect cycles on MAIN's _loop with backoff neutralized.

    MAIN hardcodes ``ConstantBackoff()`` (5 s) inside ``_loop`` -- not injectable --
    so to measure the loop's reconnect *code path* (task churn + events + connect +
    close) rather than a fixed sleep we swap the module's ConstantBackoff for a
    zero-delay stub. The caveat note records that this is a deliberate adjustment.

    Completion is read off ``conn.v`` -- MAIN's connection generation, bumped once
    per drop -- rather than a ConnectionEstablishedEvent handler. Under a zero-delay
    flap MAIN's _loop saturates the loop and starves the async emit_task handlers
    (they pile up unbounded), so an event-based signal lags by tens of thousands of
    cycles; the generation counter is the honest, scheduling-independent signal.
    """
    from aiohttp import WSMsgType

    original_backoff = main_mod.ConstantBackoff
    main_mod.ConstantBackoff = lambda *a, **k: ZeroBackoff()
    try:
        session = FakeSession(lambda: DropWebSocket(WSMsgType.CLOSE))
        conn = main_mod.Connection(session=session)
        conn.use_running_loop()

        start = time.perf_counter()
        await conn.connect()
        while conn.v < cycles:
            await asyncio.sleep(0)
        elapsed = time.perf_counter() - start

        conn.stop()
        task = conn._loop_task.task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return elapsed
    finally:
        main_mod.ConstantBackoff = original_backoff


# --------------------------------------------------------------------------- #
# Metric 3: peak memory under a 100k flood with a slow handler.
# --------------------------------------------------------------------------- #


async def new_flood_peak(api: Any, total: int, sheddable: bool) -> Tuple[int, int]:
    """Peak tracemalloc bytes flooding NEW's lease drain with a slow async handler.

    With ``sheddable`` the flood is AT_MOST_ONCE, so the per-lease HandlerDrain's
    bound governs peak memory: the producer outruns the slow handler and the drain
    sheds oldest past its maxsize instead of growing without limit. With
    ``sheddable`` False the flood is AT_LEAST_ONCE (lossless): the drain may NOT
    drop, so the backlog grows like MAIN's -- the honest other half of the picture.
    Returns (peak_bytes, handled_count); handled < total proves the bound shed.
    """
    WsTransport = api["WsTransport"]
    Reconnecting = api["Reconnecting"]
    MessageReceived = api["MessageReceived"]
    WsTextMessage = api["WsTextMessage"]
    QoS = api["QoS"]

    class IdleTransport(WsTransport, Reconnecting):
        async def open(self) -> None:
            pass

        async def recv(self) -> Optional[object]:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def write(self, message: object) -> None:
            pass

        async def aclose(self) -> None:
            pass

    pool = api["Pool"](
        build=lambda url, params: IdleTransport(url),
        key=lambda url, params: str(url),
        lease_class=api["WsConnection"],
    )
    lease = pool.connect(yarl.URL("ws://bench/"))

    handled = 0

    async def slow_handler(event: object) -> None:
        nonlocal handled
        handled += 1
        await asyncio.sleep(0)  # yield: producer outruns the handler

    lease.event_bus.on(MessageReceived, slow_handler)

    qos = QoS.AT_MOST_ONCE if sheddable else QoS.AT_LEAST_ONCE
    message = WsTextMessage("x", qos=qos)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    for _ in range(total):
        lease.feed(MessageReceived(0, message))
    # Let the bounded drain run to completion.
    while lease.drain.task is not None and not lease.drain.task.done():
        await asyncio.sleep(0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    await lease.close()
    return peak, handled


async def main_flood_peak(main_mod: Any, total: int) -> Tuple[int, int]:
    """Peak tracemalloc bytes flooding MAIN's event_bus with a slow async handler.

    MAIN delivers each inbound message with ``emit_task`` -> ``run_coroutine_threadsafe``,
    creating one coroutine/future per message with no bound. With a slow handler the
    scheduled work piles up; this measures the peak that pile reaches. We drive
    poll() directly (the realistic per-message delivery path) rather than spinning
    the whole _loop, so the measurement is just the delivery + handler cost.
    """
    from aiohttp import WSMsgType

    from simplyprint_ws_client.core.ws_protocol.events import ConnectionIncomingEvent

    payload = json.dumps({"type": "pong"})
    ws = FloodWebSocket(WSMsgType.TEXT, payload, total)
    session = FakeSession(lambda: ws)

    conn = main_mod.Connection(session=session)
    conn.use_running_loop()
    conn.ws = ws  # poll() reads self.ws directly

    handled = 0

    async def slow_handler(*args: Any, **kwargs: Any) -> None:
        nonlocal handled
        handled += 1
        await asyncio.sleep(0)

    conn.event_bus.on(ConnectionIncomingEvent, slow_handler)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    for _ in range(total):
        await conn.poll()  # parse + emit_task per message, exactly as the loop does
    # Drain everything emit_task scheduled.
    while handled < total:
        await asyncio.sleep(0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, handled


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{n} B"


def ratio_note(new: float, main: float, lower_is_better: bool) -> str:
    if new <= 0 or main <= 0:
        return ""
    if lower_is_better:
        if new < main:
            return f"NEW {main / new:.2f}x cheaper"
        return f"NEW {new / main:.2f}x more expensive"
    if new > main:
        return f"NEW {new / main:.2f}x faster"
    return f"NEW {main / new:.2f}x slower"


def run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


def main() -> None:
    main_mod = load_main_connection()
    api = new_imports()

    print("Benchmarking NEW contrib.connection vs MAIN core/ws_protocol Connection")
    print(f"(fake in-process wires, no network; python {sys.version.split()[0]})")
    print()

    # -- Metric 1: inbound throughput -- #
    new_raw_tput = run(new_inbound(api, INBOUND_MESSAGES, parse=False))
    new_parse_tput = run(new_inbound(api, INBOUND_MESSAGES, parse=True))
    main_tput = run(main_inbound(main_mod, INBOUND_MESSAGES))

    # -- Metric 2: reconnect cost -- #
    new_recon = run(new_reconnect(api, RECONNECT_CYCLES))
    main_recon = run(main_reconnect(main_mod, RECONNECT_CYCLES))

    # -- Metric 3: peak memory under flood -- #
    new_peak_shed, new_handled_shed = run(
        new_flood_peak(api, FLOOD_MESSAGES, sheddable=True)
    )
    new_peak_loss, new_handled_loss = run(
        new_flood_peak(api, FLOOD_MESSAGES, sheddable=False)
    )
    main_peak, main_handled = run(main_flood_peak(main_mod, FLOOD_MESSAGES))

    rows: List[Tuple[str, str, str, str]] = [
        (
            "inbound throughput (raw passthrough)",
            f"{new_raw_tput:,.0f} msg/s",
            f"{main_tput:,.0f} msg/s",
            ratio_note(new_raw_tput, main_tput, lower_is_better=False),
        ),
        (
            "inbound throughput (handler parses JSON, equal work)",
            f"{new_parse_tput:,.0f} msg/s",
            f"{main_tput:,.0f} msg/s",
            ratio_note(new_parse_tput, main_tput, lower_is_better=False),
        ),
        (
            f"reconnect cost ({RECONNECT_CYCLES} cycles, 0 backoff)",
            f"{new_recon * 1e3:,.1f} ms",
            f"{main_recon * 1e3:,.1f} ms",
            ratio_note(new_recon, main_recon, lower_is_better=True),
        ),
        (
            f"peak memory ({FLOOD_MESSAGES:,} flood, sheddable / AT_MOST_ONCE)",
            human_bytes(new_peak_shed),
            human_bytes(main_peak),
            ratio_note(float(new_peak_shed), float(main_peak), lower_is_better=True),
        ),
        (
            f"peak memory ({FLOOD_MESSAGES:,} flood, lossless / AT_LEAST_ONCE)",
            human_bytes(new_peak_loss),
            human_bytes(main_peak),
            ratio_note(float(new_peak_loss), float(main_peak), lower_is_better=True),
        ),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    new_w = max(len(r[1]) for r in rows + [("", "NEW", "", "")]) + 2
    main_w = max(len(r[2]) for r in rows + [("", "", "MAIN", "")]) + 2

    header = (
        f"{'metric'.ljust(label_w)}{'NEW'.ljust(new_w)}{'MAIN'.ljust(main_w)}verdict"
    )
    print(header)
    print("-" * (label_w + new_w + main_w + 24))
    for label, new_v, main_v, note in rows:
        print(f"{label.ljust(label_w)}{new_v.ljust(new_w)}{main_v.ljust(main_w)}{note}")

    print()
    print("Takeaways:")
    print(
        "  * inbound (raw): "
        + (
            "NEW is faster -- its lease delivers wire-shaped objects with no "
            "per-message parse, where MAIN runs pydantic ServerMsg validation in poll()."
            if new_raw_tput > main_tput
            else "MAIN is faster on the raw path."
        )
    )
    print(
        "  * inbound (equal work): with the same JSON parse on both sides the gap is "
        + (
            f"NEW {new_parse_tput / main_tput:.2f}x"
            if new_parse_tput >= main_tput
            else f"MAIN {main_tput / new_parse_tput:.2f}x"
        )
        + " -- this isolates the dispatch machinery from the parse MAIN bakes in."
    )
    print(
        "  * reconnect: with backoff neutralized on both sides, "
        + ratio_note(new_recon, main_recon, lower_is_better=True)
        + " -- pure connect/drop/event/close code-path cost per cycle."
    )
    print(
        "  * peak memory (sheddable): NEW shed "
        + f"{FLOOD_MESSAGES - new_handled_shed:,}/{FLOOD_MESSAGES:,}"
        + " AT_MOST_ONCE msgs via its bounded per-lease drain ("
        + human_bytes(new_peak_shed)
        + "); MAIN handled "
        + f"{main_handled:,}/{FLOOD_MESSAGES:,}"
        + " by creating one unbounded coroutine/future per message ("
        + human_bytes(main_peak)
        + ", "
        + ratio_note(float(new_peak_shed), float(main_peak), lower_is_better=True)
        + ")."
    )
    print(
        "  * peak memory (lossless): NEW may not drop AT_LEAST_ONCE, so its drain "
        + f"handled {new_handled_loss:,}/{FLOOD_MESSAGES:,} and peaked at "
        + human_bytes(new_peak_loss)
        + " -- "
        + ratio_note(float(new_peak_loss), float(main_peak), lower_is_better=True)
        + " vs MAIN; both back up under a lossless flood, but NEW queues plain "
        "coroutines in one drain task while MAIN schedules a future per message."
    )

    print()
    print("Methodology caveats (where a perfectly fair comparison is not possible):")
    print(
        "  1. The stacks are different shapes. NEW's transport hands the lease a "
        "wire-shaped object (str/bytes/WsMessage) and never parses a brand protocol; "
        "MAIN's poll() parses pydantic ServerMsg before dispatch. The 'raw' row is "
        "NEW's real contract; the 'equal work' row adds the same parse to NEW so the "
        "dispatch machinery can be compared apart from the parse."
    )
    print(
        "  2. MAIN hardcodes ConstantBackoff() (5 s) inside _loop and it is not "
        "injectable. To measure reconnect *code path* cost rather than a fixed sleep, "
        "both sides use a zero-delay backoff (NEW via RetryPolicy, MAIN by swapping "
        "the module's ConstantBackoff). In production MAIN's reconnect is dominated by "
        "that fixed 5 s sleep; NEW's pace is a configurable RetryPolicy."
    )
    print(
        "  3. Flood memory is measured with a SHEDDABLE (AT_MOST_ONCE) stream so NEW's "
        "bound is exercised. NEW treats lossless (AT_LEAST_ONCE / lifecycle) traffic "
        "as never-drop, so a lossless flood would back up like MAIN's; the win shown "
        "is specifically for droppable telemetry under a slow consumer."
    )
    print(
        "  4. MAIN's flood is driven through poll() directly (parse + emit_task per "
        "message) -- the exact per-message delivery the _loop performs -- rather than "
        "spinning the whole loop, so the number is delivery + handler scheduling, not "
        "loop scaffolding. emit_task uses run_coroutine_threadsafe, one future per msg."
    )
    print(
        "  5. tracemalloc adds overhead and counts Python allocations only (not "
        "interpreter/C buffers); treat peaks as relative, not absolute RSS."
    )
    print(
        "  6. All numbers are single-run on one machine with fakes; absolute values "
        "vary by host. The ratios and the structural differences are the signal."
    )


if __name__ == "__main__":
    main()
