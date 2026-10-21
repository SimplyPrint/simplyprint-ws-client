"""Structural logging policy shared by file, console and live sinks."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Final, Literal, Mapping, Optional, Tuple

from simplyprint_ws_client.common.logging.naming import (
    PRINTER_ROOT,
    decode_uid,
    encode_uid,
    is_printer_logger,
)

DEFAULT_NOISY_LOGGERS: Tuple[str, ...] = (
    "aiohttp",
    "asyncio",
    "charset_normalizer",
    "httpcore",
    "httpx",
    "multipart",
    "PIL",
    "tzlocal",
    "urllib3",
    "websocket",
    "websockets",
    "apscheduler",
)

LOG_TARGET_FILE: Final = "file"
LOG_TARGET_QUEUE: Final = "queue"
LOG_TARGET_STREAM: Final = "stream"
LOG_TARGET_LIVE: Final = "live"

#: The sinks a record can be policy-checked against (the LOG_TARGET_* values).
LogTarget = Literal["file", "queue", "stream", "live"]

#: The :class:`DebugOverrides` key that turns DEBUG on for every logger. A
#: printer ``unique_id`` can't collide with it: ids are stored URL-encoded.
DEBUG_ALL: Final = "*"

_PRINTER_PREFIX = PRINTER_ROOT + "."


def _printer_segment(logger_name: str) -> Optional[str]:
    """The encoded uid segment of a printer logger name, else ``None``."""
    if not logger_name.startswith(_PRINTER_PREFIX):
        return None
    return logger_name[len(_PRINTER_PREFIX) :].partition(".")[0]


class DebugOverrides:
    """DEBUG turned on at runtime, for the whole process or for single printers.

    Keys are :data:`DEBUG_ALL` or a printer ``unique_id``; a value is the
    wall-clock expiry (``time.time()`` seconds), or ``None`` for "until turned
    off". Wall clock, not monotonic, because an app may persist the entries
    (the client keeps them in its settings file) and re-apply them on start.

    An override lowers the *file* and *live* thresholds to DEBUG for the loggers
    it covers. It never touches the console stream, so journald/stderr don't get
    a printer's raw traffic. Noisy third-party loggers stay clamped.

    ``covers`` runs for every record below a sink's level, on any thread, so it
    takes no lock: writers swap in a whole new dict, and readers only ever see
    one complete dict. An expired entry stops matching at once; :meth:`expire`
    only tidies up (and tells the caller to raise logger levels back).
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._write_lock = threading.Lock()
        #: ``DEBUG_ALL`` or an *encoded* uid -> expiry. Encoded, so ``covers``
        #: compares the logger-name segment as is, without decoding per record.
        self._entries: Dict[str, Optional[float]] = {}

    def set(self, entries: Mapping[str, Optional[float]]) -> None:
        """Replace every override with ``entries`` (already expired ones are
        dropped). An empty mapping turns runtime debugging off."""
        now = self._clock()
        fresh = {
            (key if key == DEBUG_ALL else encode_uid(key)): until
            for key, until in entries.items()
            if until is None or until > now
        }
        with self._write_lock:
            self._entries = fresh

    def active(self) -> Dict[str, Optional[float]]:
        """The unexpired overrides, keyed like :meth:`set` takes them."""
        now = self._clock()
        return {
            (key if key == DEBUG_ALL else decode_uid(key)): until
            for key, until in self._entries.items()
            if until is None or until > now
        }

    def expire(self) -> bool:
        """Drop expired overrides. Returns whether any were dropped."""
        with self._write_lock:
            now = self._clock()
            kept = {
                key: until
                for key, until in self._entries.items()
                if until is None or until > now
            }
            if len(kept) == len(self._entries):
                return False
            self._entries = kept
            return True

    def covers_all(self) -> bool:
        return self._live(self._entries, DEBUG_ALL)

    def covers(self, logger_name: str) -> bool:
        """Whether DEBUG is on, right now, for the logger ``logger_name``."""
        entries = self._entries
        if not entries:  # the common case: nobody is debugging
            return False
        if self._live(entries, DEBUG_ALL):
            return True
        segment = _printer_segment(logger_name)
        return segment is not None and self._live(entries, segment)

    def _live(self, entries: Dict[str, Optional[float]], key: str) -> bool:
        if key not in entries:
            return False
        until = entries[key]
        return until is None or until > self._clock()


@dataclass(frozen=True)
class LoggingPolicy:
    """Level/filter policy for every sink the logging facility owns.

    Production defaults: every file gets INFO, so DEBUG (raw device traffic) never
    reaches disk unless someone asks for it through ``development``
    (``_resolve_config`` lowers the file levels). The live printer view stays
    DEBUG: it only formats records for a scope someone is watching. Noisy
    third-party prefixes are clamped to warnings for every target, including
    development mode.
    """

    system_file_level: int = logging.INFO
    printer_file_level: int = logging.INFO
    stream_system_level: int = logging.INFO
    stream_printer_level: int = logging.WARNING
    live_system_level: int = logging.INFO
    live_printer_level: int = logging.DEBUG
    noisy_level: int = logging.WARNING
    noisy_loggers: Tuple[str, ...] = DEFAULT_NOISY_LOGGERS
    #: Runtime DEBUG switches. Mutable, shared by every copy ``replace`` makes of
    #: this policy, and left out of equality/hash so the frozen policy stays a
    #: valid ``lru_cache`` key.
    debug: DebugOverrides = field(
        default_factory=DebugOverrides, compare=False, repr=False
    )

    def allows(self, record: logging.LogRecord, target: LogTarget) -> bool:
        """Whether ``record`` should reach ``target``.

        ``target`` is one of ``queue``, ``file``, ``stream`` or ``live``. Unknown
        targets are rejected; accepting silently would make a typo disable policy
        enforcement.
        """
        if self.is_noisy(record.name) and record.levelno < self.noisy_level:
            return False

        if target == LOG_TARGET_QUEUE:
            return self.allows(record, LOG_TARGET_FILE) or self.allows(
                record, LOG_TARGET_STREAM
            )

        if target == LOG_TARGET_FILE:
            level = (
                self.printer_file_level
                if self.is_printer(record.name)
                else self.system_file_level
            )
        elif target == LOG_TARGET_STREAM:
            level = (
                self.stream_printer_level
                if self.is_printer(record.name)
                else self.stream_system_level
            )
        elif target == LOG_TARGET_LIVE:
            level = (
                self.live_printer_level
                if self.is_printer(record.name)
                else self.live_system_level
            )
        else:
            return False

        if record.levelno >= level:
            return True
        # The console is never opened up at runtime (see DebugOverrides).
        return target != LOG_TARGET_STREAM and self.debug.covers(record.name)

    @lru_cache(maxsize=512)
    def is_printer(self, logger_name: str) -> bool:
        return is_printer_logger(logger_name)

    @lru_cache(maxsize=512)
    def is_noisy(self, logger_name: str) -> bool:
        return any(
            logger_name == prefix or logger_name.startswith(prefix + ".")
            for prefix in self.noisy_loggers
        )

    def system_logger_level(self) -> int:
        if self.debug.covers_all():
            return logging.DEBUG
        return min(
            self.system_file_level,
            self.stream_system_level,
            self.live_system_level,
        )

    def printer_logger_level(self) -> int:
        # Any override may name a printer, so the printer root opens up for all
        # of them; the per-record ``allows`` check keeps the others out.
        if self.debug.active():
            return logging.DEBUG
        return min(
            self.printer_file_level,
            self.stream_printer_level,
            self.live_printer_level,
        )

    def apply_logger_levels(self) -> Callable[[], None]:
        """Set logger thresholds before unwanted records are allocated.

        Returns a restore callback for tests/reconfiguration. The root level gates
        app/system records, while the printer root stays debug-capable so raw
        printer logs still reach their scoped files. Noisy third-party roots are
        clamped even in development mode.
        """
        levels = {
            "": logging.getLogger().level,
            PRINTER_ROOT: logging.getLogger(PRINTER_ROOT).level,
            **{name: logging.getLogger(name).level for name in self.noisy_loggers},
        }

        self.refresh_logger_levels()
        for name in self.noisy_loggers:
            logging.getLogger(name).setLevel(self.noisy_level)

        def restore() -> None:
            for name, level in levels.items():
                logging.getLogger(name).setLevel(level)

        return restore

    def refresh_logger_levels(self) -> None:
        """Re-apply the root and printer-root thresholds in place, after a
        :class:`DebugOverrides` change. Takes effect for the next record, with
        no restart and no handler rebuilt."""
        logging.getLogger().setLevel(self.system_logger_level())
        logging.getLogger(PRINTER_ROOT).setLevel(self.printer_logger_level())


class LoggingPolicyFilter(logging.Filter):
    """A ``logging.Filter`` adapter for :class:`LoggingPolicy`."""

    def __init__(self, policy: LoggingPolicy, target: LogTarget) -> None:
        super().__init__()
        self._policy = policy
        self._target = target

    def filter(self, record: logging.LogRecord) -> bool:
        return self._policy.allows(record, self._target)
