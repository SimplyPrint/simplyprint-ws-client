"""Structural logging policy shared by file, console and live sinks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Tuple

from simplyprint_ws_client.contrib.logging.naming import PRINTER_ROOT, is_printer_logger

DEFAULT_NOISY_LOGGERS: Tuple[str, ...] = (
    "aiohttp",
    "asyncio",
    "httpcore",
    "httpx",
    "PIL",
    "tzlocal",
    "urllib3",
    "websocket",
    "websockets",
    "apscheduler",
)

LOG_TARGET_FILE = "file"
LOG_TARGET_QUEUE = "queue"
LOG_TARGET_STREAM = "stream"
LOG_TARGET_LIVE = "live"


@dataclass(frozen=True)
class LoggingPolicy:
    """Level/filter policy for every sink the logging facility owns.

    Printer loggers keep raw detail in their scoped files/live printer view, while
    app/system channels stay operationally useful by default. Noisy third-party
    prefixes are clamped to warnings for every target, including development mode.
    """

    system_file_level: int = logging.INFO
    printer_file_level: int = logging.DEBUG
    stream_system_level: int = logging.INFO
    stream_printer_level: int = logging.WARNING
    live_system_level: int = logging.INFO
    live_printer_level: int = logging.DEBUG
    noisy_level: int = logging.WARNING
    noisy_loggers: Tuple[str, ...] = DEFAULT_NOISY_LOGGERS

    def allows(self, record: logging.LogRecord, target: str) -> bool:
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

        return record.levelno >= level

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
        return min(
            self.system_file_level,
            self.stream_system_level,
            self.live_system_level,
        )

    def printer_logger_level(self) -> int:
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

        logging.getLogger().setLevel(self.system_logger_level())
        logging.getLogger(PRINTER_ROOT).setLevel(self.printer_logger_level())
        for name in self.noisy_loggers:
            logging.getLogger(name).setLevel(self.noisy_level)

        def restore() -> None:
            for name, level in levels.items():
                logging.getLogger(name).setLevel(level)

        return restore


class LoggingPolicyFilter(logging.Filter):
    """A ``logging.Filter`` adapter for :class:`LoggingPolicy`."""

    def __init__(self, policy: LoggingPolicy, target: str) -> None:
        super().__init__()
        self._policy = policy
        self._target = target

    def filter(self, record: logging.LogRecord) -> bool:
        return self._policy.allows(record, self._target)
