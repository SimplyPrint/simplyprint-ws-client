"""The host's loop-stall watchdog: one WARNING at the soft threshold, one ERROR
at the hard one, one recovery line, and silence on a healthy or idle loop.

The level logic is driven by hand (``check()`` with a fake clock and a fake
loop that runs callbacks only when told), so it is deterministic; one test
runs the real thread against a real loop with short thresholds, and the Host
tests prove the thread starts with the services and is joined at stop.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Callable, List, Tuple

import pytest

from simplyprint_ws_client import ClientSettings, PrinterConfig
from simplyprint_ws_client.core.host import Host
from simplyprint_ws_client.core.loop_watchdog import LoopWatchdog
from simplyprint_ws_client.core.registry import SpecRegistry
from simplyprint_ws_client.integration.spec import (
    IntegrationId,
    IntegrationSpec,
    ProductMetadata,
)

_LOGGER = "simplyprint.loop_watchdog"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Loop:
    """Runs scheduled callbacks only on :meth:`run_pending`."""

    def __init__(self) -> None:
        self.running = True
        self.closed = False
        self.pending: List[Tuple[Callable[..., Any], tuple]] = []

    def is_running(self) -> bool:
        return self.running

    def is_closed(self) -> bool:
        return self.closed

    def call_soon_threadsafe(self, callback, *args) -> None:
        if self.closed:
            raise RuntimeError("Event loop is closed")
        self.pending.append((callback, args))

    def run_pending(self) -> None:
        pending, self.pending = self.pending, []
        for callback, args in pending:
            callback(*args)


def _watchdog(**kwargs):
    clock, loop = _Clock(), _Loop()
    kwargs.setdefault("warn_after", 10.0)
    kwargs.setdefault("error_after", 300.0)
    return LoopWatchdog(loop, clock=clock, **kwargs), clock, loop


def _lines(caplog) -> List[Tuple[int, str]]:
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == _LOGGER]


def test_a_healthy_loop_logs_nothing(caplog):
    watchdog, clock, loop = _watchdog()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for _ in range(1000):
            assert watchdog.check() == 0.0
            loop.run_pending()
            clock.now += 1.0
    assert _lines(caplog) == []
    assert not watchdog.stalled


def test_one_heartbeat_outstanding_at_a_time():
    watchdog, clock, loop = _watchdog()
    watchdog.check()
    for _ in range(5):
        clock.now += 1.0
        watchdog.check()
    assert len(loop.pending) == 1


def test_a_stall_warns_once_errors_once_and_recovers_once(caplog):
    watchdog, clock, loop = _watchdog()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        watchdog.check()  # requests the heartbeat; the loop never runs it
        clock.now += 9.0
        assert watchdog.check() == 9.0
        assert _lines(caplog) == []

        clock.now += 2.0  # 11 s
        watchdog.check()
        clock.now += 5.0  # 16 s: still warned, nothing new
        watchdog.check()
        assert _lines(caplog) == [
            (logging.WARNING, "app event loop stalled: no callback has run for 11.0 s")
        ]
        assert watchdog.stalled

        clock.now += 290.0  # 306 s
        watchdog.check()
        clock.now += 100.0  # 406 s: still errored, nothing new
        watchdog.check()
        assert _lines(caplog)[1:] == [
            (
                logging.ERROR,
                "app event loop stalled: no callback has run for 306.0 s "
                "(still stalled)",
            )
        ]

        loop.run_pending()  # the loop runs again
        assert _lines(caplog)[2:] == [
            (
                logging.WARNING,
                "app event loop recovered: it ran callbacks again after a "
                "406.0 s stall",
            )
        ]
        assert not watchdog.stalled

        clock.now += 1.0
        assert watchdog.check() == 0.0
        loop.run_pending()
    assert len(_lines(caplog)) == 3


def test_a_jump_past_the_hard_threshold_logs_only_the_error(caplog):
    watchdog, clock, loop = _watchdog()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        watchdog.check()
        clock.now += 500.0
        watchdog.check()
    assert [level for level, _ in _lines(caplog)] == [logging.ERROR]


def test_each_stall_is_reported_on_its_own(caplog):
    watchdog, clock, loop = _watchdog()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for _ in range(2):
            watchdog.check()
            clock.now += 20.0
            watchdog.check()
            loop.run_pending()
            clock.now += 1.0
    assert [level for level, _ in _lines(caplog)] == [logging.WARNING] * 4


def test_without_the_soft_level_only_the_error_is_logged(caplog):
    watchdog, clock, loop = _watchdog(warn_after=None)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        watchdog.check()
        clock.now += 100.0
        watchdog.check()
        assert _lines(caplog) == []
        clock.now += 250.0
        watchdog.check()
    assert [level for level, _ in _lines(caplog)] == [logging.ERROR]


def test_a_loop_that_is_not_running_is_not_stalled(caplog):
    watchdog, clock, loop = _watchdog()
    loop.running = False
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for _ in range(5):
            clock.now += 400.0
            assert watchdog.check() == 0.0
        assert loop.pending == []

        # A stall cut short by the loop stopping is forgotten, not reported.
        loop.running = True
        watchdog.check()
        clock.now += 20.0
        watchdog.check()
        loop.running = False
        watchdog.check()
        assert not watchdog.stalled
        loop.running = True
        loop.run_pending()  # the stale heartbeat runs late: ignored
        clock.now += 1000.0
        watchdog.check()  # a fresh request, not a 1000 s stall
    assert [level for level, _ in _lines(caplog)] == [logging.WARNING]


def test_a_closed_loop_is_ignored():
    watchdog, clock, loop = _watchdog()
    loop.closed = True
    clock.now += 1000.0
    assert watchdog.check() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"error_after": 0},
        {"warn_after": 0},
        {"warn_after": 300.0, "error_after": 300.0},
        {"interval": 0},
    ],
)
def test_bad_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _watchdog(**kwargs)


def test_the_default_interval_is_a_fraction_of_the_first_threshold():
    assert _watchdog()[0].interval == 1.0
    assert _watchdog(warn_after=0.4, error_after=1.0)[0].interval == pytest.approx(0.1)
    assert _watchdog(warn_after=None, error_after=2.0)[0].interval == 0.5


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_the_thread_catches_a_real_blocked_loop(caplog):
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    watchdog = LoopWatchdog(loop, warn_after=0.05, error_after=0.1, interval=0.01)
    release = threading.Event()
    try:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            watchdog.start()
            assert watchdog.running
            loop.call_soon_threadsafe(release.wait, 10.0)  # block the loop
            assert _wait_for(
                lambda: any(level == logging.ERROR for level, _ in _lines(caplog))
            )
            release.set()
            assert _wait_for(lambda: any("recovered" in m for _, m in _lines(caplog)))
    finally:
        release.set()
        watchdog.stop()
        loop.call_soon_threadsafe(loop.stop)
        runner.join(5)
        loop.close()
    assert not watchdog.running
    watchdog.start()  # a stopped watchdog stays stopped
    assert not watchdog.running


def test_stop_before_start_is_safe():
    watchdog, _, _ = _watchdog()
    watchdog.stop()
    watchdog.stop()
    watchdog.start()
    assert not watchdog.running


# ---------------------------------------------------------------------------
# Owned by the Host: started with the services, joined when the root closes
# ---------------------------------------------------------------------------

_METADATA = ProductMetadata(
    display_name="Watch",
    image_url="/watch.png",
    supported_transports=(),
    capabilities=(),
)


def _spec() -> IntegrationSpec:
    def factory(config, *, context):
        raise AssertionError("no printers in this test")

    return IntegrationSpec(
        id=IntegrationId("watch"),
        client_factory=factory,
        config_factory=PrinterConfig,
        metadata=_METADATA,
    )


def _watchdog_threads() -> List[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "sp-loop-watchdog"]


def test_the_host_starts_its_watchdog_with_the_services_and_joins_it_at_stop():
    before = set(_watchdog_threads())
    host = Host(SpecRegistry.of(_spec()), ClientSettings(name="loop-watchdog"))
    assert not host.loop_watchdog.running
    assert host.loop_watchdog.loop is host.app._app_event_loop

    host.start_services()
    host.start_services()  # idempotent: one thread
    assert host.loop_watchdog.running
    assert len(set(_watchdog_threads()) - before) == 1

    assert host.stop().ok
    assert not host.loop_watchdog.running
    assert set(_watchdog_threads()) - before == set()


def test_a_host_stopped_before_its_services_never_starts_the_watchdog():
    host = Host(SpecRegistry.of(_spec()), ClientSettings(name="loop-watchdog-late"))
    host.stop()
    host.start_services()
    assert not host.loop_watchdog.running
