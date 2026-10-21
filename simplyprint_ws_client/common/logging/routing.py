"""The delegating routing handler + formatters.

A single :class:`RoutingHandler` (instance state only -- no ClassVar globals)
dispatches each record to a per-destination rotating file handler chosen by
config rules matched against the plain logger name. Unmatched records fall to the
system rule. Each destination gets its own formatter, so text and JSON
destinations can coexist in one process.
"""

from __future__ import annotations

import gzip
import json
import logging
import logging.handlers
import os
import queue
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING, Dict

from simplyprint_ws_client.common.logging.naming import scope_of
from simplyprint_ws_client.common.logging.policy import (
    LOG_TARGET_FILE,
    LoggingPolicyFilter,
)

if TYPE_CHECKING:
    from simplyprint_ws_client.common.logging.config import LoggingConfig

__all__ = [
    "RoutingHandler",
    "JsonLogFormatter",
    "PassthroughQueueHandler",
    "BatchedRotatingFileHandler",
    "gzip_rotator",
]

logger = logging.getLogger(__name__)


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
    text/JSON formatters. When the bounded queue is full, the oldest pending
    record is discarded so application threads never block and recent evidence
    is preserved.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
            return
        except queue.Full:
            pass

        try:
            self.queue.get_nowait()
            self.queue.task_done()
        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass


#: zlib's default: text logs shrink about 8-10x. Level 1 is faster but leaves
#: noticeably more on disk, and disk wear is what this is for.
_ROTATE_COMPRESSLEVEL = 6


def gzip_rotator(source: str, dest: str) -> None:
    """``RotatingFileHandler.rotator``: move ``source`` to ``dest`` gzipped.

    ``dest`` keeps its ``<stem>.log.N`` name; only the content is gzip. That is
    the layout :class:`~simplyprint_ws_client.common.logging.store.LogStore`
    already reads (it sniffs the gzip magic) and that its
    ``compress_rotated_files`` produces. A ``.gz`` namer would orphan the
    backups written before it, which the handler would then never rotate out.

    Runs inside ``emit``, so on the queue-listener thread, never the event loop.
    The listener waits while a full file compresses (estimated: well under a
    second for 30 MB on x86, several seconds on a Pi Zero 2W); records queue up
    meanwhile. The active file stays plain text.

    Compress first, then delete, so a crash or a full disk never loses the
    backup: on any error it falls back to the plain rename the stock handler
    does, and the periodic ``compress_rotated_files`` compresses it later.
    """
    if not os.path.exists(source):
        return
    directory = os.path.dirname(dest) or "."
    # The temporary name has no ".log" in it, so LogStore never lists it.
    fd, tmp = tempfile.mkstemp(prefix=".rotate-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=_ROTATE_COMPRESSLEVEL,
                mtime=0,
            ) as compressed:
                with open(source, "rb") as plain:
                    shutil.copyfileobj(plain, compressed, 1024 * 1024)
        os.replace(tmp, dest)
        os.remove(source)
    except OSError:
        logger.debug("could not gzip %s; keeping it plain", source, exc_info=True)
        try:
            os.remove(tmp)
        except OSError:
            pass
        if os.path.exists(source):
            os.replace(source, dest)


class BatchedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A rotating file handler flushed by the queue listener after each batch.

    ``StreamHandler.emit`` flushes every line. That defeats the text stream's own
    buffering during raw MQTT/WebSocket bursts, even though all file work already
    runs on one queue-listener thread. This variant writes without flushing; the
    listener flushes as soon as the current queue batch is drained (at most 256
    records), and ``close`` still flushes through the standard handler lifecycle.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.shouldRollover(record):
                self.doRollover()
            if self.stream is None:
                self.stream = self._open()
            if self.stream is not None:
                self.stream.write(self.format(record) + self.terminator)
        except Exception:
            self.handleError(record)


class RoutingHandler(logging.Handler):
    """Routes each record to a per-destination file handler by name-based rules."""

    def __init__(self, config: "LoggingConfig") -> None:
        super().__init__()
        self._config = config
        self._root = config.resolve_log_dir()
        self._rules = config.compiled_rules()  # ordered; catch-all system rule last
        self._handlers: Dict[str, logging.handlers.RotatingFileHandler] = {}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not self._config.policy.allows(record, LOG_TARGET_FILE):
                return

            handler = self._handler_for(record)
            handler.handle(record)
        except Exception:
            self.handleError(record)

    def _handler_for(
        self, record: logging.LogRecord
    ) -> logging.handlers.RotatingFileHandler:
        scope, stem, formatter_kind = self._route(record.name)
        if scope == self._config.system_scope:
            path = self._root / (stem + ".log")
        else:
            path = self._root / scope / (stem + ".log")
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

    def close_scope(self, scope: str) -> None:
        """Close and drop every file handler under ``scope``'s directory.

        Called when a printer's logs are pruned: the open handles must go
        first, both to release the files (Windows cannot delete open files)
        and to keep this handler table from growing with printer churn.
        """
        scope_dir = self._root / scope
        prefix = str(scope_dir) + os.sep

        with self._lock:
            stale = [key for key in self._handlers if key.startswith(prefix)]
            handlers = [self._handlers.pop(key) for key in stale]

        for handler in handlers:
            handler.close()

    def clear_file(self, path: str) -> bool:
        """Truncate a live destination's file to zero *in place*.

        Used when the user "deletes" a log that the logging facility is actively
        writing: unlinking an open file fails on Windows and orphans the live
        handler's inode on POSIX, so we reset it to empty instead. The handler's
        own write position is reset too, so logging continues seamlessly without
        leaving a sparse file.

        Returns ``True`` when ``path`` is a destination this handler manages
        (whether or not its stream is open yet -- handlers open lazily), so the
        caller can fall back to unlinking an inactive rotated backup that no
        handler holds.
        """
        with self._lock:
            handler = self._handlers.get(path)
        if handler is None:
            return False
        # Hold the handler's own lock so we don't race an in-flight emit on the
        # same stream.
        handler.acquire()
        try:
            stream = handler.stream
            if stream is not None:
                stream.seek(0)
                stream.truncate()
                stream.flush()
            else:
                # delay=True handler that hasn't opened yet: truncate on disk so
                # the next emit appends to an empty file.
                try:
                    open(path, "w").close()
                except OSError:
                    logger.debug(
                        "clear_file: could not truncate %s", path, exc_info=True
                    )
        finally:
            handler.release()
        return True

    def flush(self) -> None:
        with self._lock:
            handlers = list(self._handlers.values())
        for handler in handlers:
            handler.flush()

    def close(self) -> None:
        with self._lock:
            handlers = list(self._handlers.values())
            self._handlers.clear()
        for handler in handlers:
            handler.close()
        super().close()

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
