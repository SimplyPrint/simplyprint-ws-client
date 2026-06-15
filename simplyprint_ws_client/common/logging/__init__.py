"""Library-owned logging: name-based routing, per-printer + system scopes.

Records route purely by their (plain, dotted) logger name -- no ``ClientName``
str-subclass, no custom Logger class. Per-printer loggers
(``simplyprint.printer.<uid>[.<sub>]``, made via :func:`printer_logger`) land in
``<log_dir>/<uid>/<sub>.log``; camera and worker-pool loggers get their own scope
directories (``<log_dir>/camera/`` and ``<log_dir>/workers/``); everything else
goes to one root-level app log (``<log_dir>/system.log``).
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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from simplyprint_ws_client.common.logging.config import LoggingConfig, RoutingRule
from simplyprint_ws_client.common.logging.naming import (
    PRINTER_ROOT,
    ROOT,
    printer_log_dir,
    printer_logger,
    printer_logger_name,
    scope_of,
)
from simplyprint_ws_client.common.logging.policy import (
    LOG_TARGET_FILE,
    LOG_TARGET_LIVE,
    LOG_TARGET_QUEUE,
    LOG_TARGET_STREAM,
    LoggingPolicy,
    LoggingPolicyFilter,
)
from simplyprint_ws_client.common.logging.routing import (
    JsonLogFormatter,
    PassthroughQueueHandler,
    RoutingHandler,
)
from simplyprint_ws_client.common.logging.store import (
    LogFileInfo,
    LogNotFound,
    LogScopeInfo,
    LogStore,
)

if TYPE_CHECKING:
    from simplyprint_ws_client.core.app import ClientSettings

__all__ = [
    "LoggingConfig",
    "LoggingPolicy",
    "LoggingPolicyFilter",
    "LOG_TARGET_FILE",
    "LOG_TARGET_LIVE",
    "LOG_TARGET_QUEUE",
    "LOG_TARGET_STREAM",
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


def _resolve_config(settings: "ClientSettings", config: LoggingConfig) -> LoggingConfig:
    policy = config.policy
    if settings.development:
        policy = replace(
            policy,
            system_file_level=logging.DEBUG,
            stream_system_level=logging.DEBUG,
            live_system_level=logging.DEBUG,
        )

    # The system log file is the fixed, brand-free ``system.log`` (see
    # ``LoggingConfig.system_log_stem``); the filename is never derived from
    # ``settings.name``. ``RoutingHandler`` falls back to ``system_scope`` if a
    # caller explicitly clears the stem, which still yields ``system.log``.
    return replace(config, policy=policy)


def setup_logging(
    settings: "ClientSettings", config: Optional[LoggingConfig] = None
) -> Callable[[], None]:
    """Set up logging and return the listener's ``stop`` callable.

    Records fan out (non-blocking, via a queue) to a stream handler and the
    name-based :class:`RoutingHandler`. ``config`` tunes destinations, rotation
    and rendering.
    """
    stop, _router = _setup_logging(settings, config)
    return stop


def _setup_logging(
    settings: "ClientSettings", config: Optional[LoggingConfig] = None
) -> "Tuple[Callable[[], None], RoutingHandler]":
    config = _resolve_config(settings, config or LoggingConfig())
    restore_levels = config.policy.apply_logger_levels()

    logging_queue: queue.SimpleQueue = queue.SimpleQueue()
    queue_handler = PassthroughQueueHandler(logging_queue)
    queue_handler.addFilter(LoggingPolicyFilter(config.policy, LOG_TARGET_QUEUE))
    logging.basicConfig(
        level=config.policy.system_logger_level(),
        handlers=[queue_handler],
        force=True,
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        config.formatter_for("json" if config.json_output else "text")
    )
    stream_handler.setLevel(
        min(config.policy.stream_system_level, config.policy.stream_printer_level)
    )
    stream_handler.addFilter(LoggingPolicyFilter(config.policy, LOG_TARGET_STREAM))

    router = RoutingHandler(config)
    listener = logging.handlers.QueueListener(
        logging_queue, stream_handler, router, respect_handler_level=True
    )
    listener.start()

    def stop() -> None:
        listener.stop()
        router.close()
        restore_levels()

    return stop, router


def configure_logging(
    settings: "ClientSettings", config: Optional[LoggingConfig] = None
) -> LoggingFacility:
    """Set logging up (via :func:`setup_logging`) and return a
    :class:`LoggingFacility` whose :class:`LogStore` browses/retains the result.

    The store's prune path closes the router's open per-scope file handles
    first, so pruning a printer actually releases its files and the handler
    table cannot grow with printer churn."""
    config = _resolve_config(settings, config or LoggingConfig())
    stop, router = _setup_logging(settings, config)
    store = LogStore(config, on_scope_pruned=router.close_scope)
    return LoggingFacility(store=store, stop=stop, config=config)
