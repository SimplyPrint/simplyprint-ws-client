"""Brand-free tests for the :class:`Courier` dispatch primitive.

The courier is the one seam that carries items from a producer world (a paho
thread, a camera reader) onto the SimplyPrint loop. These tests pin the
properties everything else relies on: coalescing (a burst is one loop wakeup),
strict FIFO delivery, every overflow policy, sync *and* async sinks, no
lost-wakeups under concurrent producers + draining, and safe shutdown.
"""

import asyncio
import threading
import time
from collections import defaultdict

import pytest

from simplyprint_ws_client.common.asyncio.courier import Courier, OverflowPolicy


@pytest.mark.asyncio
async def test_delivers_in_order_from_a_producer_thread():
    got = []
    courier = Courier(sink=got.append, policy=OverflowPolicy.UNBOUNDED)

    def producer():
        for i in range(300):
            courier.post(i)

    t = threading.Thread(target=producer)
    t.start()
    t.join()

    for _ in range(200):
        await asyncio.sleep(0.005)
        if len(got) == 300:
            break

    assert got == list(range(300))
    assert courier.dropped == 0


@pytest.mark.asyncio
async def test_coalesces_a_burst_into_a_single_drain():
    got = []
    courier = Courier(sink=got.append, policy=OverflowPolicy.UNBOUNDED)

    drains = []
    original = courier._drain_sync

    def counting_drain():
        drains.append(1)
        original()

    courier._drain_sync = counting_drain  # type: ignore[method-assign]

    # Post a burst from the loop thread without yielding: the first post
    # schedules one drain, the rest ride the same wakeup.
    for i in range(100):
        courier.post(i)

    assert courier.pending() == 100  # nothing drained yet
    await asyncio.sleep(0.01)

    assert got == list(range(100))
    assert len(drains) == 1  # the whole burst cost exactly one loop wakeup


@pytest.mark.asyncio
async def test_drop_oldest_keeps_the_newest_and_recycles_the_rest():
    got = []
    dropped = []
    courier = Courier(
        sink=got.append,
        policy=OverflowPolicy.DROP_OLDEST,
        maxsize=3,
        on_drop=dropped.append,
    )

    for i in range(5):
        courier.post(i)

    assert courier.pending() == 3
    await asyncio.sleep(0.01)

    assert got == [2, 3, 4]  # newest kept
    assert dropped == [0, 1]  # oldest evicted, handed to on_drop (slab recycle)
    assert courier.dropped == 2


@pytest.mark.asyncio
async def test_drop_oldest_preserves_lossless_items():
    got = []
    dropped = []
    courier = Courier(
        sink=got.append,
        policy=OverflowPolicy.DROP_OLDEST,
        maxsize=2,
        lossless=lambda item: item[0] == "lifecycle",
        on_drop=dropped.append,
    )

    courier.post(("message", 0))
    courier.post(("message", 1))
    courier.post(("lifecycle", "connected"))
    courier.post(("message", 2))
    courier.post(("message", 3))

    await asyncio.sleep(0.01)

    assert got == [("lifecycle", "connected"), ("message", 2), ("message", 3)]
    assert dropped == [("message", 0), ("message", 1)]


@pytest.mark.asyncio
async def test_drop_newest_rejects_incoming_and_reports_false():
    got = []
    dropped = []
    courier = Courier(
        sink=got.append,
        policy=OverflowPolicy.DROP_NEWEST,
        maxsize=3,
        on_drop=dropped.append,
    )

    results = [courier.post(i) for i in range(5)]

    assert results == [True, True, True, False, False]
    await asyncio.sleep(0.01)

    assert got == [0, 1, 2]
    assert dropped == [3, 4]
    assert courier.dropped == 2


@pytest.mark.asyncio
async def test_unbounded_never_drops():
    got = []
    courier = Courier(sink=got.append, policy=OverflowPolicy.UNBOUNDED, maxsize=1)

    for i in range(500):
        courier.post(i)

    await asyncio.sleep(0.02)
    assert got == list(range(500))
    assert courier.dropped == 0


def test_block_waits_for_space_then_unblocks():
    loop = asyncio.new_event_loop()
    try:
        got = []
        courier = Courier(
            sink=got.append, policy=OverflowPolicy.BLOCK, maxsize=2, loop=loop
        )

        assert courier.post(1) is True
        assert courier.post(2) is True  # buffer is now full

        done = threading.Event()

        def producer():
            courier.post(3)  # must block until the loop drains space
            done.set()

        t = threading.Thread(target=producer)
        t.start()

        assert not done.wait(0.2)  # still blocked

        courier._drain_sync()  # simulate the loop draining one batch

        assert done.wait(1.0)  # producer woke up and completed
        t.join(1.0)

        courier._drain_sync()
        assert got == [1, 2, 3]
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_block_on_the_loop_thread_is_rejected():
    courier = Courier(sink=lambda _: None, policy=OverflowPolicy.BLOCK, maxsize=1)

    assert courier.post(1) is True  # fills the buffer
    with pytest.raises(RuntimeError):
        courier.post(2)  # blocking here would deadlock the loop


@pytest.mark.asyncio
async def test_async_sink_is_awaited_in_strict_fifo_order():
    got = []

    async def sink(item):
        await asyncio.sleep(0)  # force a suspension between items
        got.append(item)

    courier = Courier(sink=sink, is_async_sink=True, policy=OverflowPolicy.UNBOUNDED)

    def producer():
        for i in range(200):
            courier.post(i)

    t = threading.Thread(target=producer)
    t.start()
    t.join()

    for _ in range(400):
        await asyncio.sleep(0.005)
        if len(got) == 200:
            break

    assert got == list(range(200))  # one serialized drain task -> no reordering


@pytest.mark.asyncio
async def test_no_loss_under_concurrent_producers_and_draining():
    got = []
    courier = Courier(sink=got.append, policy=OverflowPolicy.UNBOUNDED)

    n_threads, per = 8, 500
    total = n_threads * per

    def producer(base):
        for i in range(per):
            courier.post((base, i))

    threads = [threading.Thread(target=producer, args=(b,)) for b in range(n_threads)]
    for t in threads:
        t.start()

    # Yield repeatedly so drains run CONCURRENTLY with the producers -- this is
    # what exercises the empty->non-empty rescheduling for lost wakeups.
    for _ in range(4000):
        await asyncio.sleep(0.001)
        if len(got) == total and not any(t.is_alive() for t in threads):
            break

    for t in threads:
        t.join(1.0)

    assert len(got) == total  # not one item lost
    assert courier.dropped == 0

    per_producer = defaultdict(list)
    for base, i in got:
        per_producer[base].append(i)
    for base in range(n_threads):
        assert per_producer[base] == list(range(per))  # per-thread FIFO preserved


@pytest.mark.asyncio
async def test_a_crashing_sink_does_not_stop_delivery():
    got = []

    def sink(item):
        if item == 2:
            raise ValueError("boom")
        got.append(item)

    courier = Courier(sink=sink, policy=OverflowPolicy.UNBOUNDED)
    for i in range(5):
        courier.post(i)

    await asyncio.sleep(0.01)
    assert got == [0, 1, 3, 4]  # the crash on item 2 did not abort the batch


@pytest.mark.asyncio
async def test_close_drains_pending_then_rejects_posts():
    got = []
    courier = Courier(sink=got.append, policy=OverflowPolicy.UNBOUNDED)

    for i in range(10):
        courier.post(i)

    courier.close(drain=True)
    await asyncio.sleep(0.01)

    assert got == list(range(10))
    assert courier.post(99) is False
    assert 99 not in got


@pytest.mark.asyncio
async def test_close_without_drain_recycles_pending():
    got = []
    dropped = []
    courier = Courier(
        sink=got.append, policy=OverflowPolicy.UNBOUNDED, on_drop=dropped.append
    )

    for i in range(5):
        courier.post(i)

    courier.close(drain=False)
    await asyncio.sleep(0.01)

    assert got == []  # nothing delivered
    assert dropped == list(range(5))  # all recycled via on_drop


@pytest.mark.asyncio
async def test_close_without_drain_cancels_active_async_sink():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def sink(_item):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    courier = Courier(sink=sink, is_async_sink=True, policy=OverflowPolicy.UNBOUNDED)
    courier.post(1)
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.005)
    assert started.is_set()

    courier.close(drain=False)
    for _ in range(200):
        if cancelled.is_set():
            break
        await asyncio.sleep(0.005)
    assert cancelled.is_set()


def test_close_during_concurrent_posts_is_safe():
    loop = asyncio.new_event_loop()

    def run_loop():
        def selector_heartbeat():
            if loop.is_running():
                loop.call_later(0.01, selector_heartbeat)

        loop.call_soon(selector_heartbeat)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop)
    loop_thread.start()
    try:
        got = []
        courier = Courier(sink=got.append, policy=OverflowPolicy.UNBOUNDED, loop=loop)

        stop = threading.Event()
        errors = []

        def producer():
            try:
                while not stop.is_set():
                    courier.post(1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        producers = [threading.Thread(target=producer) for _ in range(4)]
        for p in producers:
            p.start()

        time.sleep(0.05)
        courier.close()  # close mid-flight while producers hammer post()
        stop.set()
        for p in producers:
            p.join(1.0)

        assert errors == []  # no producer ever saw an exception
        assert courier.post(1) is False  # posts after close are rejected
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        assert not loop_thread.is_alive()
        loop.close()


def test_a_bounded_policy_requires_a_positive_maxsize():
    with pytest.raises(ValueError):
        Courier(sink=lambda _: None, policy=OverflowPolicy.DROP_OLDEST, maxsize=0)
