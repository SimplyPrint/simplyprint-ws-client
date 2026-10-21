"""The host's loop-stall watchdog: notice when the app event loop stops running
callbacks, and say so once.

Everything a printer does runs on the app event loop, so a callback that blocks
it (a sync socket read, a CPU-bound parse, a deadlock) silently freezes the
whole fleet. This watchdog is the host's one detector for that, whatever the
brand: brands own no threads and no liveness timers of their own.

How it works. One daemon thread (``sp-loop-watchdog``) wakes every
``interval`` seconds and asks the loop to run a tiny heartbeat callback
(``call_soon_threadsafe``). At most one heartbeat is outstanding; how long the
outstanding one has waited is the *stall*. On a healthy loop it runs within
milliseconds and the stall is zero at every check.

Two levels, each logged once per stall:

* ``warn_after`` (default 10 s): a WARNING. Long enough that a busy-but-healthy
  loop never trips it, short enough to point at the offending code while it is
  still happening. ``None`` turns this level off.
* ``error_after`` (default 300 s): an ERROR -- the old Duet watchdog's
  threshold, kept so long freezes still read as errors in the logs.

When the heartbeat finally runs, one WARNING says the loop recovered and how
long it was stalled. A loop that is not running (not started yet, stopped) is
not a stall: the watchdog waits quietly and starts measuring when it runs.

The thread is started by :meth:`LoopWatchdog.start` and joined by
:meth:`LoopWatchdog.stop`; the :class:`~simplyprint_ws_client.core.host.Host`
starts it with its services and stops it when its root scope closes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Optional

__all__ = ["LoopWatchdog"]

logger = logging.getLogger("simplyprint.loop_watchdog")

#: Default thresholds, in seconds (see the module doc).
DEFAULT_WARN_AFTER = 10.0
DEFAULT_ERROR_AFTER = 300.0

_OK = 0
_WARNED = 1
_ERRORED = 2


class LoopWatchdog:
    """Watches one event loop for stalls from a single daemon thread.

    ``clock`` is injectable so tests can drive :meth:`check` by hand with a fake
    clock and never start the thread.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        warn_after: Optional[float] = DEFAULT_WARN_AFTER,
        error_after: float = DEFAULT_ERROR_AFTER,
        interval: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        name: str = "app event loop",
    ) -> None:
        if error_after <= 0:
            raise ValueError("error_after must be positive")
        if warn_after is not None and not 0 < warn_after < error_after:
            raise ValueError("warn_after must be positive and below error_after")
        first = warn_after if warn_after is not None else error_after
        if interval is None:
            interval = min(1.0, first / 4)
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.loop = loop
        self.warn_after = warn_after
        self.error_after = error_after
        self.interval = interval
        self.name = name
        self._clock = clock
        self._lock = threading.Lock()
        #: When the outstanding heartbeat was requested, or ``None``.
        self._requested_at: Optional[float] = None
        self._level = _OK
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the watching thread. Idempotent; a stopped watchdog stays
        stopped."""
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            self._thread = threading.Thread(
                target=self._run, name="sp-loop-watchdog", daemon=True
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Stop and join the thread. Idempotent and safe before :meth:`start`."""
        self._stop.set()
        with self._lock:
            thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join()

    @property
    def running(self) -> bool:
        """Whether the watching thread is alive."""
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def stalled(self) -> bool:
        """Whether a stall past a threshold is in progress (until it recovers)."""
        with self._lock:
            return self._level != _OK

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except Exception:  # never let the detector die quietly
                logger.exception("loop watchdog check failed")

    # ------------------------------------------------------------------ checking

    def check(self) -> float:
        """One check: request a heartbeat if none is outstanding, else report
        how long the outstanding one has waited. Returns the stall in seconds.

        Called by the thread every ``interval``; tests call it directly.
        """
        now = self._clock()
        loop = self.loop
        if loop.is_closed() or not loop.is_running():
            # Not running is not stalled: forget any outstanding beat (it will
            # never run, or runs on a later start and is ignored as stale).
            with self._lock:
                self._requested_at = None
                self._level = _OK
            return 0.0
        with self._lock:
            requested = self._requested_at
            if requested is None:
                self._requested_at = now
        if requested is None:
            try:
                loop.call_soon_threadsafe(self._beat, now)
            except RuntimeError:  # closed between the check and the call
                with self._lock:
                    if self._requested_at == now:
                        self._requested_at = None
            return 0.0
        stalled = now - requested
        self._escalate(stalled)
        return stalled

    def _escalate(self, stalled: float) -> None:
        with self._lock:
            level = self._level
            if stalled >= self.error_after and level < _ERRORED:
                self._level = _ERRORED
            elif (
                self.warn_after is not None
                and stalled >= self.warn_after
                and level < _WARNED
            ):
                self._level = _WARNED
            else:
                return
            new_level = self._level
        if new_level == _ERRORED:
            logger.error(
                "%s stalled: no callback has run for %.1f s (still stalled)",
                self.name,
                stalled,
            )
        else:
            logger.warning(
                "%s stalled: no callback has run for %.1f s",
                self.name,
                stalled,
            )

    def _beat(self, requested_at: float) -> None:
        """Runs on the watched loop: the heartbeat came through."""
        now = self._clock()
        with self._lock:
            if self._requested_at != requested_at:
                return  # stale: a reset forgot it
            self._requested_at = None
            level = self._level
            self._level = _OK
        if level != _OK:
            logger.warning(
                "%s recovered: it ran callbacks again after a %.1f s stall",
                self.name,
                now - requested_at,
            )
