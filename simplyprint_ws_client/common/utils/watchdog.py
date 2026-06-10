"""A small liveness watchdog timer.

A brand-agnostic "dead-man's switch": something resets it periodically while the
connection is healthy; if too long passes without a reset it latches *expired*
(it records expiry, it does not kill anything -- the owner decides what to do).

Originally lived inside a polling integration; it is plain infrastructure, so it
belongs in the shared layer where any polling transport can reuse it.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class Watchdog:
    """A simple watchdog timer that records expiry without killing the process."""

    def __init__(self, timeout: float, *, name: str = "watchdog"):
        """Create a watchdog that expires ``timeout`` seconds after its last reset."""
        self.timeout = timeout
        self.name = name
        self._next_reset = time.monotonic() + self.timeout
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._expired_event = threading.Event()
        self._thread = threading.Thread(target=self._watchdog_thread, daemon=True)

    def start(self):
        """Start the watchdog thread."""
        self._thread.start()

    def stop(self):
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()

    @property
    def expired(self) -> bool:
        return self._expired_event.is_set()

    async def reset(self):
        """Reset the watchdog timer (taking the lock is instant; no executor hop)."""
        self.reset_sync()

    def reset_sync(self, offset: float = 0):
        """Reset the watchdog timer synchronously."""
        with self._lock:
            self._next_reset = time.monotonic() + self.timeout + offset

    def _watchdog_thread(self):
        """Thread that checks the watchdog timer.

        Sleeps until the current deadline on the stop event (instead of a 1 s
        poll), so stopping is immediate and the thread wakes only when due.
        """
        while not self._stop_event.is_set():
            with self._lock:
                remaining = self._next_reset - time.monotonic()
            if remaining <= 0:
                self._expired_event.set()
                logger.error("%s expired", self.name)
                break
            self._stop_event.wait(remaining)
