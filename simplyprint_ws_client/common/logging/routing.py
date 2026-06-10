"""The delegating routing handler + formatters.

A single :class:`RoutingHandler` (instance state only -- no ClassVar globals)
dispatches each record to a per-destination rotating file handler chosen by
config rules matched against the plain logger name. Unmatched records fall to the
system rule. Each destination gets its own formatter, so text and JSON
destinations can coexist in one process.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
from typing import TYPE_CHECKING, Dict

from simplyprint_ws_client.common.logging.naming import scope_of
from simplyprint_ws_client.common.logging.policy import LOG_TARGET_FILE, LoggingPolicyFilter

if TYPE_CHECKING:
    from simplyprint_ws_client.common.logging.config import LoggingConfig

__all__ = ["RoutingHandler", "JsonLogFormatter", "PassthroughQueueHandler"]


class JsonLogFormatter(logging.Formatter):
    """Render a record as one JSON line, tagged with its log scope."""

    def __init__(self, system_scope: str = "system") -> None:
        super().__init__()
        self._system_scope = system_scope

    def format(self, record: logging.LogRecord) -> str:
        scope, unique_id = scope_of(record, self._system_scope)
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "scope": scope,
            "message": record.getMessage(),
        }
        if unique_id is not None:
            payload["unique_id"] = unique_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class PassthroughQueueHandler(logging.handlers.QueueHandler):
    """A QueueHandler that enqueues the *raw* record (no pre-formatting).

    The stock QueueHandler formats in ``prepare()``, which would bake one format
    into the record before it reaches the listener -- defeating per-destination
    text/JSON formatters. We run a single in-process SimpleQueue, so passing the
    live record through is safe.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


class RoutingHandler(logging.Handler):
    """Routes each record to a per-destination file handler by name-based rules."""

    def __init__(self, config: "LoggingConfig") -> None:
        super().__init__()
        self._config = config
        self._rules = config.compiled_rules()  # ordered; catch-all system rule last
        self._handlers: Dict[str, logging.Handler] = {}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self._config.policy.allows(record, LOG_TARGET_FILE):
                return

            handler = self._handler_for(record)
            handler.handle(record)
        except Exception:
            self.handleError(record)

    def _handler_for(self, record: logging.LogRecord) -> logging.Handler:
        scope, stem, formatter_kind = self._route(record.name)
        root = self._config.resolve_log_dir()
        if scope == self._config.system_scope:
            path = root / (stem + ".log")
        else:
            path = root / scope / (stem + ".log")
        key = str(path)
        handler = self._handlers.get(key)
        if handler is not None:
            return handler
        with self._lock:
            handler = self._handlers.get(key)
            if handler is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = self._config.make_file_handler(path)
                handler.setFormatter(self._config.formatter_for(formatter_kind))
                handler.addFilter(
                    LoggingPolicyFilter(self._config.policy, LOG_TARGET_FILE)
                )
                self._handlers[key] = handler
            return handler

    def _route(self, logger_name: str):
        for rule in self._rules:
            if rule.matches(logger_name):
                if rule.resolver is not None:
                    scope, stem = rule.resolver(logger_name)
                else:
                    scope, stem = (
                        self._config.system_scope,
                        self._config.system_log_stem or self._config.system_scope,
                    )
                return scope, stem, rule.formatter
        # _rules always ends with a catch-all, so this is unreachable.
        return (
            self._config.system_scope,
            self._config.system_log_stem or self._config.system_scope,
            "text",
        )

    def close(self) -> None:
        with self._lock:
            for handler in self._handlers.values():
                handler.close()
            self._handlers.clear()
        super().close()
