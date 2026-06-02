"""Library-owned logging: name-based routing, per-printer + system scopes.

Records route purely by their (plain, dotted) logger name -- no ``ClientName``
str-subclass, no custom Logger class. Per-printer loggers
(``simplyprint.printer.<uid>[.<sub>]``, made via :func:`printer_logger`) land in
``<log_dir>/<uid>/<sub>.log``; everything else in one root-level
``<log_dir>/system.log``.
Output is plain text or one-line JSON, chosen per routing rule, so a record can be
rendered differently per destination.

Two entry points: :func:`setup_logging` (the low-level primitive; returns the
listener ``stop``) and :func:`configure_logging` (adds a :class:`LogStore` for
browsing/retention, returned together as a :class:`LoggingFacility`).
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from .config import LoggingConfig, RoutingRule
from .naming import (
    PRINTER_ROOT,
    ROOT,
    printer_log_dir,
    printer_logger,
    printer_logger_name,
    scope_of,
)
from .routing import JsonLogFormatter, PassthroughQueueHandler, RoutingHandler
from .store import LogFileInfo, LogNotFound, LogScopeInfo, LogStore

if TYPE_CHECKING:
    from ...core.app import ClientSettings

__all__ = [
    "LoggingConfig",
    "RoutingRule",
    "LoggingFacility",
    "LogStore",
    "LogNotFound",
    "LogFileInfo",
    "LogScopeInfo",
    "RoutingHandler",
    "JsonLogFormatter",
    "scope_of",
    "printer_logger",
    "printer_logger_name",
    "printer_log_dir",
    "PRINTER_ROOT",
    "ROOT",
    "setup_logging",
    "configure_logging",
]


@dataclass
class LoggingFacility:
    """The configured facility: the on-disk :class:`LogStore` plus the listener's
    ``stop`` callable, returned by :func:`configure_logging`."""

    store: LogStore
    stop: Callable[[], None]
    config: LoggingConfig


def setup_logging(
    settings: "ClientSettings", config: Optional[LoggingConfig] = None
) -> Callable[[], None]:
    """Set up logging and return the listener's ``stop`` callable.

    Records fan out (non-blocking, via a queue) to a stream handler and the
    name-based :class:`RoutingHandler`. ``config`` tunes destinations, rotation
    and rendering.
    """
    config = config or LoggingConfig()

    logging_queue: queue.SimpleQueue = queue.SimpleQueue()
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[PassthroughQueueHandler(logging_queue)],
        force=True,
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        config.formatter_for("json" if config.json_output else "text")
    )
    stream_handler.setLevel(logging.DEBUG if settings.development else logging.INFO)

    router = RoutingHandler(config)
    listener = logging.handlers.QueueListener(
        logging_queue, stream_handler, router, respect_handler_level=True
    )
    listener.start()

    def stop() -> None:
        listener.stop()
        router.close()

    return stop


def configure_logging(
    settings: "ClientSettings", config: Optional[LoggingConfig] = None
) -> LoggingFacility:
    """Set logging up (via :func:`setup_logging`) and return a
    :class:`LoggingFacility` whose :class:`LogStore` browses/retains the result."""
    config = config or LoggingConfig()
    stop = setup_logging(settings, config)
    return LoggingFacility(store=LogStore(config), stop=stop, config=config)
