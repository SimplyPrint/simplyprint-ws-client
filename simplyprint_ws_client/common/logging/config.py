"""Configuration for the library-owned logging facility.

The library owns where records land and how they're rendered; an integration
tunes it with a :class:`LoggingConfig` (or accepts the defaults, which reproduce
the historical ``<uid>/<sub>.log`` + root-level app log layout). Routing is a list
of :class:`RoutingRule` matched against the plain logger name, so an app can add
destinations (e.g. send ``discovery`` to its own JSON file) without touching the
handler.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

from simplyprint_ws_client.common.logging.policy import LoggingPolicy

DEFAULT_TEXT_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 30 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3

#: How a destination renders records: plain text lines or JSON objects.
FormatterKind = Literal["text", "json"]

#: Resolves a matched logger name to ``(scope, file_stem)``.
NameResolver = Callable[[str], Tuple[str, str]]

#: Logger-name roots for shared (non-printer) subsystems, and the on-disk scopes
#: they route to. These are generic infrastructure names (no brand): camera
#: capture and the generic worker pool each get their own scope directory so they
#: never spam the system log.
CAMERA_LOGGER_ROOT = "camera"
WORKER_LOGGER_ROOT = "worker"
CAMERA_SCOPE = "camera"
WORKER_SCOPE = "workers"


def _camera_resolver(_logger_name: str) -> Tuple[str, str]:
    return CAMERA_SCOPE, CAMERA_SCOPE


def _worker_resolver(_logger_name: str) -> Tuple[str, str]:
    return WORKER_SCOPE, WORKER_SCOPE


@dataclass(frozen=True)
class RoutingRule:
    """A logger-name predicate -> on-disk destination.

    ``prefix`` is matched against the dotted logger name (``None`` = catch-all).
    ``resolver`` maps the matched name to ``(scope, file_stem)``; when ``None``
    the rule writes to the single system file. ``formatter`` is ``"text"`` or
    ``"json"``. Rules are brand-free: they name a logger-name prefix, never a brand.
    """

    name: str
    prefix: Optional[str]
    formatter: FormatterKind = "text"
    resolver: Optional[NameResolver] = None

    def matches(self, logger_name: str) -> bool:
        if self.prefix is None:
            return True
        return logger_name == self.prefix or logger_name.startswith(self.prefix + ".")


@dataclass(frozen=True)
class LoggingConfig:
    """Where logs are stored and how they're rendered."""

    log_dir: Optional[Path] = None
    system_scope: str = "system"
    # The system log file stem -- fixed and brand-free, so the app/system log is
    # always ``<log_dir>/system.log`` rather than ``<ClientSettings.name>.log``.
    # Kept as a separate axis from ``system_scope`` (scope = UI grouping; stem =
    # filename). ``RoutingHandler`` alone also falls back to ``system_scope``.
    system_log_stem: Optional[str] = "system"
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT
    text_format: str = DEFAULT_TEXT_FORMAT
    date_format: str = DEFAULT_DATE_FORMAT
    json_output: bool = False
    policy: LoggingPolicy = field(default_factory=LoggingPolicy)
    #: Custom routing rules; ``None`` -> the default per-printer + camera + worker
    #: + system rules.
    routes: Optional[List[RoutingRule]] = None
    #: Non-printer scope directories the default routing produces. Prune keeps
    #: these (alongside the system scope and active printers), so a printer-churn
    #: prune never deletes shared-subsystem logs. Override when passing custom
    #: ``routes`` that resolve to other fixed scopes.
    reserved_scopes: Tuple[str, ...] = (CAMERA_SCOPE, WORKER_SCOPE)

    def resolve_log_dir(self) -> Path:
        if self.log_dir is not None:
            return Path(self.log_dir)
        from simplyprint_ws_client.const import APP_DIRS

        return APP_DIRS.user_log_path

    def make_file_handler(self, path: Path) -> logging.Handler:
        return logging.handlers.RotatingFileHandler(
            path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            delay=True,
        )

    def formatter_for(self, kind: FormatterKind) -> logging.Formatter:
        from simplyprint_ws_client.common.logging.routing import JsonLogFormatter

        if kind == "json":
            return JsonLogFormatter(self.system_scope)
        return logging.Formatter(self.text_format, self.date_format)

    def compiled_rules(self) -> List[RoutingRule]:
        """The routing rules, always ending in a catch-all system rule."""
        from simplyprint_ws_client.common.logging.naming import (
            PRINTER_ROOT,
            printer_resolver,
        )

        kind: FormatterKind = "json" if self.json_output else "text"
        if self.routes is not None:
            rules = list(self.routes)
        else:
            rules = [
                RoutingRule("printer", PRINTER_ROOT, kind, printer_resolver),
                RoutingRule("camera", CAMERA_LOGGER_ROOT, kind, _camera_resolver),
                RoutingRule("workers", WORKER_LOGGER_ROOT, kind, _worker_resolver),
            ]
        rules.append(RoutingRule("system", None, kind))
        return rules
