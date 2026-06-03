import threading
import time

from simplyprint_ws_client.contrib.runtime import stop_with_timeout


def test_blocking_stop_forces_exit_within_timeout():
    blocked = threading.Event()
    forced = []

    def stop():
        blocked.wait()

    started = time.monotonic()
    stop_with_timeout(stop, timeout=0.2, force_exit=lambda code: forced.append(code))
    elapsed = time.monotonic() - started

    blocked.set()

    assert forced == [1]
    assert elapsed < 2.0


def test_fast_stop_does_not_force_exit():
    forced = []
    called = []

    def stop():
        called.append(True)

    stop_with_timeout(stop, timeout=5.0, force_exit=lambda code: forced.append(code))

    assert called == [True]
    assert forced == []


def test_force_exit_receives_nonzero_code():
    forced = []

    def stop():
        time.sleep(1.0)

    stop_with_timeout(stop, timeout=0.05, force_exit=lambda code: forced.append(code))

    assert forced == [1]
    assert forced[0] != 0
