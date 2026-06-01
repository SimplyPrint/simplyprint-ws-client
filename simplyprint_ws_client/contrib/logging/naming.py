"""Logger naming and scope derivation for the routing logging facility.

Routing is driven entirely by the (plain, dotted) logger name -- there is no
``ClientName`` str-subclass and no custom Logger class. Per-printer loggers are
named ``simplyprint.printer.<encoded-uid>[.<sub>...]``; the ``unique_id`` is
URL-encoded into a single opaque segment so it can never be confused with the
sub-logger separator (uids may contain dots, spaces, slashes, ...). Everything
without that prefix is the system scope.
"""

from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple

#: Library root logger; per-printer loggers live under ``<ROOT>.printer``.
ROOT = "simplyprint"
PRINTER_ROOT = ROOT + ".printer"


def encode_uid(unique_id: str) -> str:
    """A single dot/slash/space-free name segment for an opaque unique_id.

    ``quote`` leaves ``.`` untouched (it's always-safe), but the dot is our
    logger-name separator, so encode it too. Reversible via :func:`decode_uid`.
    """
    return urllib.parse.quote(str(unique_id), safe="").replace(".", "%2E")


def decode_uid(segment: str) -> str:
    """Inverse of :func:`encode_uid`."""
    return urllib.parse.unquote(segment)


def printer_logger_name(unique_id: str, sub: Optional[str] = None) -> str:
    """The dotted logger name for a printer (and optional sub-logger)."""
    name = PRINTER_ROOT + "." + encode_uid(unique_id)
    return name + "." + sub if sub else name


def printer_logger(unique_id: str, sub: Optional[str] = None) -> logging.Logger:
    """The one front door for per-printer loggers -- a plain ``logging.Logger``.

    ``printer_logger(uid)`` is the printer's base logger; ``printer_logger(uid,
    "mqtt")`` (or ``base.getChild("mqtt")``) is its mqtt sub-logger, written to
    ``<log_dir>/<uid>/mqtt.log`` by the routing handler.
    """
    return logging.getLogger(printer_logger_name(unique_id, sub))


def is_printer_logger(logger_name: str) -> bool:
    return logger_name == PRINTER_ROOT or logger_name.startswith(PRINTER_ROOT + ".")


def scope_of(
    record: logging.LogRecord, system_scope: str = "system"
) -> Tuple[str, Optional[str]]:
    """``(scope, unique_id)`` derived purely from the record's logger name."""
    scope, uid, _ = _parse(record.name, system_scope)
    return scope, uid


def printer_resolver(logger_name: str) -> Tuple[str, str]:
    """``'simplyprint.printer.<enc>.mqtt' -> ('<uid>', 'mqtt')``; base -> ('<uid>', 'main')."""
    from ...shared.utils.slugify import slugify

    rest = logger_name[len(PRINTER_ROOT) + 1 :]
    segment, _, sub = rest.partition(".")
    uid = decode_uid(segment)
    stem = slugify(sub.replace(".", "-")) if sub else "main"
    return uid, stem


def _parse(logger_name: str, system_scope: str) -> Tuple[str, Optional[str], str]:
    prefix = PRINTER_ROOT + "."
    if logger_name.startswith(prefix):
        uid, stem = printer_resolver(logger_name)
        return uid, uid, stem
    return system_scope, None, system_scope


def printer_log_dir(unique_id: str, log_dir: Optional[Path] = None) -> Path:
    """The on-disk directory for a printer's logs (created on demand)."""
    if log_dir is not None:
        root = Path(log_dir)
    else:
        from ...const import APP_DIRS

        root = APP_DIRS.user_log_path
    folder = root / str(unique_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
