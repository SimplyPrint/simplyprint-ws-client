"""Scope semantics: LIFO, isolation, loop hops, timeout, idempotence, children."""

import asyncio
import threading
import time

import pytest

from simplyprint_ws_client.core.scope import CloseReport, Scope, ScopeClosed


async def _cancel_others():
    current = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="scope-test-loop")
    thread.start()
    yield loop
    asyncio.run_coroutine_threadsafe(_cancel_others(), loop).result(5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def test_undos_run_last_in_first_out():
    order = []
    scope = Scope("s")
    for name in ("a", "b", "c"):
        scope.on_close(lambda name=name: order.append(name))

    report = scope.close("done")

    assert order == ["c", "b", "a"]
    assert report == CloseReport()
    assert report.ok
    assert scope.closed
    assert scope.close_reason == "done"


def test_a_raising_undo_is_isolated_and_reported():
    order = []
    boom = RuntimeError("boom")
    scope = Scope("s")
    scope.on_close(lambda: order.append("first"))

    def explode():
        raise boom

    scope.on_close(explode)
    scope.on_close(lambda: order.append("last"))

    report = scope.close("done")

    assert order == ["last", "first"]
    assert report.errors == (boom,)
    assert report.pending == ()
    assert not report.ok


def test_an_async_undo_raising_is_isolated_too(loop):
    order = []
    scope = Scope("s")
    scope.on_close(lambda: order.append("first"))

    async def explode():
        raise ValueError("async boom")

    scope.on_close(explode, loop=loop)
    scope.on_close(explode)  # no loop: driven on a private loop

    report = scope.close("done")

    assert order == ["first"]
    assert [type(error) for error in report.errors] == [ValueError, ValueError]


def test_on_close_hops_to_the_given_loop(loop):
    seen = {}
    scope = Scope("s")

    def sync_undo():
        seen["sync"] = (asyncio.get_running_loop(), threading.get_ident())

    async def async_undo():
        await asyncio.sleep(0)
        seen["async"] = (asyncio.get_running_loop(), threading.get_ident())

    def returns_awaitable():
        return async_undo_two()

    async def async_undo_two():
        seen["returned"] = asyncio.get_running_loop()

    scope.on_close(sync_undo, loop=loop)
    scope.on_close(async_undo, loop=loop)
    scope.on_close(returns_awaitable, loop=loop)

    report = scope.close("done")

    assert report.ok
    assert seen["sync"][0] is loop
    assert seen["async"][0] is loop
    assert seen["returned"] is loop
    assert seen["sync"][1] != threading.get_ident()


def test_an_undo_without_a_loop_runs_on_the_closing_thread():
    seen = []
    scope = Scope("s")
    scope.on_close(lambda: seen.append(threading.get_ident()))

    scope.close("done")

    assert seen == [threading.get_ident()]


def test_an_awaitable_from_a_loopless_undo_is_driven_to_completion():
    done = threading.Event()
    scope = Scope("s")

    async def finish():
        await asyncio.sleep(0.01)
        done.set()

    scope.on_close(finish)

    assert scope.close("done").ok
    assert done.is_set()


def test_a_slow_undo_is_reported_pending_and_the_rest_still_run(loop):
    order = []
    scope = Scope("s")
    scope.on_close(lambda: order.append("after-slow"))

    async def slow():
        await asyncio.sleep(10)

    scope.on_close(slow, loop=loop, name="slow")
    scope.on_close(lambda: order.append("before-slow"))

    started = time.monotonic()
    report = scope.close("done", timeout=0.2)
    elapsed = time.monotonic() - started

    assert report.pending == ("slow",)
    assert report.errors == ()
    assert order == ["before-slow", "after-slow"]
    assert elapsed < 5


def test_the_timeout_is_one_budget_for_the_whole_close(loop):
    scope = Scope("s")

    async def slow():
        await asyncio.sleep(10)

    for name in ("one", "two", "three"):
        scope.on_close(slow, loop=loop, name=name)

    started = time.monotonic()
    report = scope.close("done", timeout=0.3)

    assert time.monotonic() - started < 2
    assert set(report.pending) == {"one", "two", "three"}


def test_closing_twice_is_a_no_op():
    calls = []
    scope = Scope("s")
    scope.on_close(lambda: calls.append(1))

    first = scope.close("first")
    second = scope.close("second")

    assert calls == [1]
    assert first.ok
    assert second == CloseReport()
    assert scope.close_reason == "first"


def test_concurrent_closes_run_each_undo_once():
    calls = []
    gate = threading.Event()
    scope = Scope("s")

    def undo():
        gate.wait(1)
        calls.append(threading.get_ident())

    scope.on_close(undo)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(scope.close("race")))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(5)

    assert len(calls) == 1
    assert len(results) == 4


def test_an_undo_that_closes_its_own_scope_does_not_deadlock():
    scope = Scope("s")
    nested = []
    scope.on_close(lambda: nested.append(scope.close("again")))

    assert scope.close("done").ok
    assert nested == [CloseReport()]


def test_children_close_before_the_parents_own_undos_newest_first():
    order = []
    parent = Scope("parent")
    parent.on_close(lambda: order.append("parent-early"))
    first = parent.child("first")
    first.on_close(lambda: order.append("first"))
    parent.on_close(lambda: order.append("parent-late"))
    second = parent.child("second")
    second.on_close(lambda: order.append("second"))

    report = parent.close("down")

    assert order == ["second", "first", "parent-late", "parent-early"]
    assert report.ok
    assert first.closed and second.closed
    assert first.close_reason == "down"


def test_a_child_closed_on_its_own_is_not_closed_again():
    calls = []
    parent = Scope("parent")
    child = parent.child("child")
    child.on_close(lambda: calls.append("child"))

    child.close("unload")
    parent.close("down")

    assert calls == ["child"]


def test_child_errors_and_pending_merge_into_the_parent_report(loop):
    parent = Scope("parent")
    child = parent.child("child")

    def explode():
        raise KeyError("x")

    async def slow():
        await asyncio.sleep(10)

    child.on_close(explode)
    child.on_close(slow, loop=loop, name="slow")

    report = parent.close("down", timeout=0.2)

    assert [type(error) for error in report.errors] == [KeyError]
    assert report.pending == ("child/slow",)


def test_a_slot_closes_at_its_position_not_first():
    order = []
    parent = Scope("parent")
    parent.on_close(lambda: order.append("before"))
    slot = parent.slot("slot")
    parent.on_close(lambda: order.append("after"))
    slot.on_close(lambda: order.append("slot"))  # registered last, runs middle

    parent.close("down")

    assert order == ["after", "slot", "before"]


def test_registering_on_a_closed_scope_undoes_at_once_and_raises():
    calls = []
    scope = Scope("s")
    scope.close("done")

    with pytest.raises(ScopeClosed):
        scope.on_close(lambda: calls.append("late"))
    with pytest.raises(ScopeClosed):
        scope.child("late")

    assert calls == ["late"]


def test_spawned_tasks_are_cancelled_on_close(loop):
    started = threading.Event()
    cancelled = threading.Event()
    scope = Scope("s", loop=loop)

    async def forever():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = scope.spawn(forever(), name="forever")
    assert started.wait(5)
    assert task.get_name() == "forever"

    report = scope.close("done")

    assert report.ok
    assert cancelled.is_set()
    assert task.done()


def test_children_spawn_on_the_parents_loop(loop):
    ran_on = []
    scope = Scope("s", loop=loop)
    child = scope.child("c")

    async def record():
        ran_on.append(asyncio.get_running_loop())

    task = child.spawn(record(), name="record")
    deadline = time.monotonic() + 5
    while not task.done() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert ran_on == [loop]
    scope.close("done")


def test_spawn_on_a_closed_or_loopless_scope_is_refused(loop):
    async def noop():
        return None

    with pytest.raises(RuntimeError):
        Scope("no-loop").spawn(noop(), name="x")

    closed = Scope("s", loop=loop)
    closed.close("done")
    with pytest.raises(ScopeClosed):
        closed.spawn(noop(), name="x")


def test_a_finished_spawned_task_is_forgotten(loop):
    scope = Scope("s", loop=loop)

    async def quick():
        return 1

    task = scope.spawn(quick(), name="quick")
    deadline = time.monotonic() + 5
    while scope._entries and time.monotonic() < deadline:
        time.sleep(0.01)

    assert task.done()
    assert scope._entries == []
    scope.close("done")
