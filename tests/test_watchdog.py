"""``contrib.watchdog`` -- a brand-agnostic liveness timer.

A small dead-man's switch any polling integration can reuse: it records expiry
after ``timeout`` seconds without a reset, without killing anything. It lives at
the contrib root (not inside ``connection``) because it is plain infrastructure,
not part of the printer-connection pool.
"""

import time

from simplyprint_ws_client.common.utils.watchdog import Watchdog


def test_watchdog_expires_after_timeout():
    # The watchdog thread polls on a 1s granularity, so the observation window
    # must clear timeout + one poll tick.
    wd = Watchdog(0.3, name="test-wd")
    wd.start()
    try:
        assert not wd.expired
        time.sleep(1.6)
        assert wd.expired
    finally:
        wd.stop()


def test_watchdog_does_not_expire_before_timeout():
    wd = Watchdog(5.0, name="test-wd-long")
    wd.start()
    try:
        wd.reset_sync()
        time.sleep(1.6)
        assert not wd.expired
    finally:
        wd.stop()
