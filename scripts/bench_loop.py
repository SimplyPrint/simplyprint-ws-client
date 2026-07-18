"""Benchmark app-loop responsiveness under blocking work (the loop-health probes).

Run with the library venv::

    .venv/bin/python scripts/bench_loop.py

Useful knobs::

    .venv/bin/python scripts/bench_loop.py --cameras 3 --beat-ms 10
    .venv/bin/python scripts/bench_loop.py --flushes 50 --connect-timeout 2.0
    .venv/bin/python scripts/bench_loop.py --process     # also stress PROCESS workers
    .venv/bin/python scripts/bench_loop.py --json

Companion to ``scripts/bench_conn.py``. Where bench_conn measures connection
*delivery throughput*, this measures what a connection benchmark cannot see: how
long the single app/scheduler loop thread is *unresponsive* when a blocking
operation runs on it. The metric is loop-latency, not message rate.

A background "heartbeat" coroutine wakes every ``--beat-ms`` and records the gap
between successive wakeups. On a healthy loop every gap ~= the interval; when
something blocks the loop thread, exactly one gap balloons to the block
duration. We report max/p99/mean gap, total time the loop was stalled past a
threshold, and heartbeat throughput (beats/s) -- so a 2s thread-join or a
no-timeout socket connect shows up as a concrete millisecond stall.

Each blocker is measured INLINE (run directly on the loop, as today) vs
OFFLOADED (run via a bounded executor / single reaper, as the plan proposes), on
the SAME real library code, so the before/after is a measured delta, not a
claim. Scenarios:

* camera worker stop -- a REAL ``WorkerPool`` THREAD worker whose producer
  ignores its stop event; ``WorkerHandle.stop()`` then blocks for ``JOIN_TIMEOUT``
  (~2s). N of them stopped serially reproduces the observed ~6s compounding
  stall. The offloaded variant moves the joins to one reaper thread.
* config flush -- a REAL ``JsonConfigManager.flush`` (json + fsync + os.replace)
  fired as an N-event storm inline vs coalesced+offloaded to one write.
* blocking connect -- a REAL ``socket.create_connection`` to an unreachable
  RFC-5737 address; inline it parks the loop for the connect timeout (None in
  production = unbounded), offloaded it does not.

Performance probes, not CI assertions. Numbers vary by host; compare relative
changes on the same machine. No network egress except the loopback worker IPC
and the (failing) connect to a documentation address.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import socket
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.worker.context import ExecutionContext
from simplyprint_ws_client.common.worker.pool import WorkerPool
from simplyprint_ws_client.core.config.json import JsonConfigManager
from simplyprint_ws_client.core.config import PrinterConfig

DEFAULT_BEAT_MS = 10.0
DEFAULT_CAMERAS = 3
DEFAULT_FLUSHES = 50
DEFAULT_CONFIGS = 20
DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_WORKER_BLOCK = 10.0  # how long the wedged producer ignores its stop event

# RFC 5737 TEST-NET-1: guaranteed-unrouteable, safe to (fail to) connect to.
BLACKHOLE = ("192.0.2.1", 81)


@dataclass(frozen=True)
class LoopMetric:
    name: str
    events: int  # blocking ops performed (workers / flushes / connects)
    beats: int  # heartbeats observed during the measured window
    interval_ms: float
    max_gap_ms: float
    p99_gap_ms: float
    mean_gap_ms: float
    blocked_ms: float  # total time the loop overran the interval (the stall)
    beats_per_second: float
    peak_bytes: int
    notes: str = ""


# --------------------------------------------------------------------------- #
# Heartbeat harness: run an async probe while a heartbeat samples loop latency.
# --------------------------------------------------------------------------- #


class Heartbeat:
    """Samples the gap between successive loop wakeups; the gap spikes to the
    block duration whenever the loop thread is held synchronously."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.gaps: List[float] = []
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            await asyncio.sleep(self.interval_s)
            now = time.perf_counter()
            self.gaps.append(now - last)
            last = now

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def measure(
    name: str,
    events: int,
    interval_s: float,
    body: Callable[[], Awaitable[None]],
    notes: str = "",
) -> LoopMetric:
    """Run ``body`` (which performs the blocking work) while a heartbeat samples
    loop latency, and turn the sampled gaps into a metric."""
    heartbeat = Heartbeat(interval_s)

    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()

    heartbeat.start()
    await asyncio.sleep(interval_s * 3)  # settle: a few clean beats first
    wall_start = time.perf_counter()
    await body()
    await asyncio.sleep(interval_s * 3)  # let the loop recover + record the spike
    wall = time.perf_counter() - wall_start
    await heartbeat.stop()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gaps_ms = [g * 1000.0 for g in heartbeat.gaps]
    interval_ms = interval_s * 1000.0
    # "blocked" = time spent beyond the expected interval (i.e. real stall).
    blocked_ms = sum(max(0.0, g - interval_ms) for g in gaps_ms)
    beats = len(gaps_ms)
    return LoopMetric(
        name=name,
        events=events,
        beats=beats,
        interval_ms=interval_ms,
        max_gap_ms=max(gaps_ms) if gaps_ms else 0.0,
        p99_gap_ms=_percentile(gaps_ms, 99.0),
        mean_gap_ms=(sum(gaps_ms) / beats) if beats else 0.0,
        blocked_ms=blocked_ms,
        beats_per_second=(beats / wall) if wall > 0 else 0.0,
        peak_bytes=peak,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Worker producers (module-level so PROCESS can pickle them).
# --------------------------------------------------------------------------- #


def wedged_producer(emit, is_stopped, block_seconds: float) -> None:
    """A producer that emits one frame then ignores its stop event for
    ``block_seconds`` -- models a camera worker stuck in a hung RTSP/FLV read.
    ``WorkerHandle.stop()`` will set the stop event and then block on join."""
    emit(b"frame", time.time())
    time.sleep(block_seconds)


# --------------------------------------------------------------------------- #
# Scenario 1: camera worker stop (the headline ~2s/~6s stall).
# --------------------------------------------------------------------------- #


def _allocate_wedged_workers(
    pool: WorkerPool, context: ExecutionContext, count: int, block_s: float
) -> list:
    handles = []
    for _ in range(count):
        handles.append(
            pool.allocate(
                context,
                wedged_producer,
                lambda _p, _ts: None,
                args=(block_s,),
                is_async=False,
            )
        )
    return handles


async def camera_stop_inline(
    context: ExecutionContext, count: int, interval_s: float, block_s: float
) -> LoopMetric:
    """Stop N wedged workers SYNCHRONOUSLY on the loop -- exactly what the
    camera controller URI setter does today. Joins add up serially."""
    provider = EventLoopProvider(loop=asyncio.get_running_loop())
    pool = WorkerPool(event_loop_provider=provider)
    handles = _allocate_wedged_workers(pool, context, count, block_s)
    await asyncio.sleep(0.05)  # let producers start

    async def body() -> None:
        for handle in handles:
            handle.stop()  # blocks the loop for JOIN_TIMEOUT each

    label = "PROCESS" if context is ExecutionContext.PROCESS else "THREAD"
    metric = await measure(
        f"camera stop INLINE x{count} ({label})",
        count,
        interval_s,
        body,
        notes=f"handle.stop() on the loop; {count} joins serialized",
    )
    pool.stop()
    return metric


async def camera_stop_offloaded(
    context: ExecutionContext, count: int, interval_s: float, block_s: float
) -> LoopMetric:
    """Proposed fix: stops dispatched to ONE background reaper thread; the loop
    only signals + enqueues, never joins."""
    provider = EventLoopProvider(loop=asyncio.get_running_loop())
    pool = WorkerPool(event_loop_provider=provider)
    handles = _allocate_wedged_workers(pool, context, count, block_s)
    await asyncio.sleep(0.05)

    reaper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench-reaper")

    async def body() -> None:
        loop = asyncio.get_running_loop()
        # The loop hands each stop to the reaper and never blocks on the join.
        futures = [loop.run_in_executor(reaper, h.stop) for h in handles]
        # We do NOT await the joins on the loop's critical path; in the real
        # design they finalize on the reaper. Await here only to keep teardown
        # clean, AFTER the measured window has captured loop latency.
        await asyncio.sleep(interval_s * 3)
        await asyncio.gather(*futures)

    label = "PROCESS" if context is ExecutionContext.PROCESS else "THREAD"
    metric = await measure(
        f"camera stop REAPER x{count} ({label})",
        count,
        interval_s,
        body,
        notes="stops offloaded to 1 reaper thread; loop only signals",
    )
    reaper.shutdown(wait=True)
    pool.stop()
    return metric


# --------------------------------------------------------------------------- #
# Scenario 2: config flush storm (real json + fsync + os.replace).
# --------------------------------------------------------------------------- #


def _make_config_manager(tmp: Path, n_configs: int) -> JsonConfigManager:
    manager = JsonConfigManager(name="bench", base_directory=str(tmp))
    for _ in range(n_configs):
        manager.persist(PrinterConfig.get_new())
    return manager


async def config_flush_inline(
    n_flushes: int, n_configs: int, interval_s: float
) -> LoopMetric:
    """Fire an N-event flush storm INLINE on the loop, as the current
    ClientConfigChangedEvent listener does (one full re-serialize + fsync each)."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = _make_config_manager(Path(tmp), n_configs)

        async def body() -> None:
            for _ in range(n_flushes):
                manager.flush()  # json.dump + fsync + os.replace, on the loop

        return await measure(
            f"config flush INLINE x{n_flushes}",
            n_flushes,
            interval_s,
            body,
            notes=f"{n_configs} configs re-serialized + fsync per event",
        )


async def config_flush_coalesced(
    n_flushes: int, n_configs: int, interval_s: float
) -> LoopMetric:
    """Proposed fix: N triggers coalesce to ONE flush, run on the io lane."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = _make_config_manager(Path(tmp), n_configs)
        io = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bench-io")

        async def body() -> None:
            loop = asyncio.get_running_loop()
            # N events arrive, but coalescing collapses them to a single
            # trailing flush, executed off the loop on the io lane.
            for _ in range(n_flushes):
                pass  # trigger() would just set a dirty flag (microseconds)
            await loop.run_in_executor(io, manager.flush)

        metric = await measure(
            f"config flush COALESCED x{n_flushes}->1",
            n_flushes,
            interval_s,
            body,
            notes=f"{n_configs} configs; coalesced to 1 offloaded write",
        )
        io.shutdown(wait=True)
        return metric


# --------------------------------------------------------------------------- #
# Scenario 3: blocking connect with no/large timeout (the unbounded-FTP shape).
# --------------------------------------------------------------------------- #


def _blocking_connect(timeout: float) -> None:
    try:
        socket.create_connection(BLACKHOLE, timeout=timeout).close()
    except OSError:
        pass  # expected: the address is unreachable


async def blocking_connect_inline(timeout: float, interval_s: float) -> LoopMetric:
    """A synchronous connect to an unreachable host on the loop -- the shape of
    the no-socket-timeout ftplib connect in ensure_file. Here bounded by
    ``timeout``; in production ftplib's timeout is None => UNBOUNDED."""

    async def body() -> None:
        _blocking_connect(timeout)

    return await measure(
        f"blocking connect INLINE (t={timeout:g}s)",
        1,
        interval_s,
        body,
        notes="no-timeout in prod = unbounded; here capped to bound the bench",
    )


async def blocking_connect_offloaded(timeout: float, interval_s: float) -> LoopMetric:
    """Proposed fix: the same connect on the transfer lane; the loop stays live."""
    transfer = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bench-transfer")

    async def body() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(transfer, _blocking_connect, timeout)

    metric = await measure(
        f"blocking connect TRANSFER LANE (t={timeout:g}s)",
        1,
        interval_s,
        body,
        notes="offloaded; loop keeps beating while the socket stalls",
    )
    transfer.shutdown(wait=True)
    return metric


# --------------------------------------------------------------------------- #
# Runner / reporting (mirrors bench_conn.py).
# --------------------------------------------------------------------------- #


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{value} B"


def print_table(metrics: List[LoopMetric]) -> None:
    rows = [
        (
            m.name,
            f"{m.events:,}",
            f"{m.beats:,}",
            f"{m.interval_ms:,.1f}",
            f"{m.max_gap_ms:,.1f}",
            f"{m.p99_gap_ms:,.1f}",
            f"{m.mean_gap_ms:,.1f}",
            f"{m.blocked_ms:,.1f}",
            f"{m.beats_per_second:,.0f}",
            human_bytes(m.peak_bytes),
            m.notes,
        )
        for m in metrics
    ]
    headers = (
        "scenario",
        "ops",
        "beats",
        "interval_ms",
        "max_gap_ms",
        "p99_ms",
        "mean_ms",
        "blocked_ms",
        "beats/s",
        "peak",
        "notes",
    )
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    def line(values: Tuple[str, ...]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    print(line(headers))
    print(line(tuple("-" * w for w in widths)))
    for row in rows:
        print(line(row))


async def run_benchmarks(args: argparse.Namespace) -> List[LoopMetric]:
    interval_s = args.beat_ms / 1000.0
    metrics: List[LoopMetric] = []

    # Scenario 1: camera worker stop (THREAD -- clean, reproducible ~2s join).
    metrics.append(
        await camera_stop_inline(
            ExecutionContext.THREAD, 1, interval_s, args.worker_block
        )
    )
    metrics.append(
        await camera_stop_inline(
            ExecutionContext.THREAD, args.cameras, interval_s, args.worker_block
        )
    )
    metrics.append(
        await camera_stop_offloaded(
            ExecutionContext.THREAD, args.cameras, interval_s, args.worker_block
        )
    )
    if args.process:
        metrics.append(
            await camera_stop_inline(
                ExecutionContext.PROCESS, 1, interval_s, args.worker_block
            )
        )
        metrics.append(
            await camera_stop_offloaded(
                ExecutionContext.PROCESS, args.cameras, interval_s, args.worker_block
            )
        )

    # Scenario 2: config flush storm.
    metrics.append(await config_flush_inline(args.flushes, args.configs, interval_s))
    metrics.append(await config_flush_coalesced(args.flushes, args.configs, interval_s))

    # Scenario 3: blocking connect.
    metrics.append(await blocking_connect_inline(args.connect_timeout, interval_s))
    metrics.append(await blocking_connect_offloaded(args.connect_timeout, interval_s))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beat-ms", type=float, default=DEFAULT_BEAT_MS)
    parser.add_argument("--cameras", type=int, default=DEFAULT_CAMERAS)
    parser.add_argument("--worker-block", type=float, default=DEFAULT_WORKER_BLOCK)
    parser.add_argument("--flushes", type=int, default=DEFAULT_FLUSHES)
    parser.add_argument("--configs", type=int, default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="also stress PROCESS workers (subprocess spawn; slower)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = asyncio.run(run_benchmarks(args))
    if args.json:
        print(json.dumps([asdict(m) for m in metrics], indent=2))
        return

    print(f"app-loop responsiveness benchmark (python {sys.version.split()[0]})")
    print("max_gap_ms = worst loop stall; blocked_ms = total stall; lower is better")
    print()
    print_table(metrics)


if __name__ == "__main__":
    main()
