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
from typing import Callable, List, Optional, Tuple

from .policy import LoggingPolicy

DEFAULT_TEXT_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 30 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3

#: Resolves a matched logger name to ``(scope, file_stem)``.
NameResolver = Callable[[str], Tuple[str, str]]


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
    formatter: str = "text"
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
    # None -> setup_logging uses ClientSettings.name; RoutingHandler alone falls
    # back to <log_dir>/<system_scope>.log.
    system_log_stem: Optional[str] = None
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT
    text_format: str = DEFAULT_TEXT_FORMAT
    date_format: str = DEFAULT_DATE_FORMAT
    json_output: bool = False
    policy: LoggingPolicy = field(default_factory=LoggingPolicy)
    #: Custom routing rules; ``None`` -> the default per-printer + system rules.
    routes: Optional[List[RoutingRule]] = None

    def resolve_log_dir(self) -> Path:
        if self.log_dir is not None:
            return Path(self.log_dir)
        from ...const import APP_DIRS

        return APP_DIRS.user_log_path

    def make_file_handler(self, path: Path) -> logging.Handler:
        return logging.handlers.RotatingFileHandler(
            path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            delay=True,
        )

    def formatter_for(self, kind: str) -> logging.Formatter:
        from .routing import JsonLogFormatter

        if kind == "json":
            return JsonLogFormatter(self.system_scope)
        return logging.Formatter(self.text_format, self.date_format)

    def compiled_rules(self) -> List[RoutingRule]:
        """The routing rules, always ending in a catch-all system rule."""
        from .naming import PRINTER_ROOT, printer_resolver

        kind = "json" if self.json_output else "text"
        if self.routes is not None:
            rules = list(self.routes)
        else:
            rules = [RoutingRule("printer", PRINTER_ROOT, kind, printer_resolver)]
        rules.append(RoutingRule("system", None, kind))
        return rules
