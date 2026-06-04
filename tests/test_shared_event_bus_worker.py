"""The SharedEventBusWorker multiplexes many EventBuses onto one thread/queue.

It must (1) dispatch each bus's events to that bus's handlers only, and (2) let a
single bus detach via its handle's ``stop()`` without disturbing the others --
the property the per-client worker's ``stop()`` used to give each printer.
"""

import threading

from simplyprint_ws_client.events.event_bus import EventBus
from simplyprint_ws_client.events.event_bus_worker import SharedEventBusWorker


class EvA:
    pass


class EvB:
    pass


def test_shared_worker_dispatches_each_bus_to_its_own_handlers():
    worker = SharedEventBusWorker(daemon=True, name="test-shared-worker")
    worker.start()
    try:
        bus_a, bus_b = EventBus(), EventBus()
        got_a, got_b = [], []
        seen_b = threading.Event()

        def _on_a(value):
            got_a.append(value)

        def _on_b(value):
            got_b.append(value)
            seen_b.set()

        bus_a.on(EvA, _on_a)
        bus_b.on(EvB, _on_b)

        # FIFO across buses: A is enqueued before B, so by the time B's handler
        # signals, A's has already run on the one shared thread.
        worker.worker_for(bus_a).emit_sync(EvA, "a1")
        worker.worker_for(bus_b).emit_sync(EvB, "b1")

        assert seen_b.wait(timeout=2.0)
        assert got_a == ["a1"]
        assert got_b == ["b1"]
    finally:
        worker.stop()


def test_handle_stop_detaches_only_that_bus():
    worker = SharedEventBusWorker(daemon=True, name="test-shared-worker-detach")
    worker.start()
    try:
        bus_a, bus_b = EventBus(), EventBus()
        got_a, got_b = [], []
        seen_b = threading.Event()

        def _on_a(value):
            got_a.append(value)

        def _on_b(value):
            got_b.append(value)
            seen_b.set()

        bus_a.on(EvA, _on_a)
        bus_b.on(EvB, _on_b)

        handle_a = worker.worker_for(bus_a)
        handle_b = worker.worker_for(bus_b)

        handle_a.stop()  # detach bus A only
        handle_a.emit_sync(EvA, "dropped")
        handle_b.emit_sync(EvB, "kept")

        assert seen_b.wait(timeout=2.0)
        assert got_b == ["kept"]
        assert got_a == []  # A's emit was dropped after its handle stopped
        assert handle_a.is_stopped() and not handle_b.is_stopped()
    finally:
        worker.stop()
