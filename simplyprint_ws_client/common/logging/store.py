"""The on-disk log store -- owns layout, retention, browsing and bundling.

Two scopes:

* the *system* scope (``config.system_scope``, default ``system``) -- app-wide
  logs. On disk these are the flat ``*.log`` files at the log-dir root.
* one per printer, keyed by its ``unique_id`` -- the ``<unique_id>/`` subdirectory
  the logging facility writes per-printer files into.

This class is the single owner of that policy: enumeration, gz-aware reading,
zip bundling for download, and the retention helpers (strip oversized raw logs,
prune directories for deleted printers, compress rotated backups). It is
brand-agnostic and depends only on stdlib + the library's ``LoggingConfig``, so
an integration gets sane log storage for free. Construct it with an explicit
``root`` in tests.
"""

from __future__ import annotations

import gzip
import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, List, Optional, Tuple

from simplyprint_ws_client.common.logging.config import LoggingConfig

#: gzip magic number -- rotated backups are compressed in place but keep their
#: ``.log.N`` name, so content has to be sniffed.
_GZIP_MAGIC = b"\x1f\x8b"


class LogNotFound(Exception):
    """A requested scope or log file does not exist (or escaped its scope)."""


@dataclass(frozen=True)
class LogFileInfo:
    name: str  # bare filename, e.g. "main.log" or "mqtt.log.2"
    size: int  # bytes on disk
    modified_at: float  # mtime, epoch seconds
    rotation_index: int  # 0 = live file, N = the ".log.N" backup
    compressed: bool  # gzip content (a compressed rotated backup)


@dataclass(frozen=True)
class LogScopeInfo:
    scope: str  # the system scope or a printer unique_id
    is_system: bool
    file_count: int
    total_size: int


def _is_log_file(path: Path) -> bool:
    """A live ``*.log`` or a rotated ``*.log.N`` backup."""
    if not path.is_file():
        return False
    suffixes = path.suffixes
    return ".log" in suffixes


def _rotation_index(path: Path) -> int:
    """0 for ``foo.log``; N for ``foo.log.N`` (and ``foo.log.N.gz``)."""
    for suffix in reversed(path.suffixes):
        token = suffix.lstrip(".")
        if token.isdigit():
            return int(token)
    return 0


def _is_compressed(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == _GZIP_MAGIC
    except OSError:
        return False


class LogStore:
    """Owns the log directory: layout, retention, browsing and bundling."""

    def __init__(
        self,
        config: Optional[LoggingConfig] = None,
        *,
        root: Optional[Path] = None,
    ) -> None:
        self._config = config or LoggingConfig()
        self._root = Path(root) if root is not None else self._config.resolve_log_dir()
        self._system_scope = self._config.system_scope

    @property
    def root(self) -> Path:
        return self._root

    @property
    def system_scope(self) -> str:
        return self._system_scope

    def system_dir(self) -> Path:
        """The directory that contains root-level system logs."""
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def _scope_token_ok(self, scope: str) -> bool:
        return (
            bool(scope)
            and "/" not in scope
            and "\\" not in scope
            and scope not in {".", ".."}
        )

    def _scope_files(self, scope: str) -> List[Path]:
        """Every log file backing ``scope`` (does not require the scope to exist)."""
        if not self._scope_token_ok(scope):
            raise LogNotFound(scope)

        if not self._root.exists():
            return []

        if scope == self._system_scope:
            return [p for p in self._root.iterdir() if _is_log_file(p)]

        scope_dir = self._root / scope
        if not scope_dir.is_dir():
            return []
        return [p for p in scope_dir.iterdir() if _is_log_file(p)]

    def resolve_file(self, scope: str, name: str) -> Path:
        """The path of ``name`` within ``scope``, or raise ``LogNotFound``.

        Guards against path traversal: ``name`` must be a bare filename that
        resolves to a real log file actually backing the scope.
        """
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise LogNotFound(name)

        for candidate in self._scope_files(scope):
            if candidate.name == name:
                return candidate

        raise LogNotFound(name)

    def list_scopes(self) -> List[LogScopeInfo]:
        """System scope first, then one entry per printer ``unique_id`` directory."""
        scopes: List[LogScopeInfo] = [self._scope_info(self._system_scope)]

        if self._root.exists():
            for child in sorted(self._root.iterdir()):
                if child.is_dir() and child.name != self._system_scope:
                    scopes.append(self._scope_info(child.name))

        return scopes

    def _scope_info(self, scope: str) -> LogScopeInfo:
        files = self._scope_files(scope)
        total = sum(self._safe_size(p) for p in files)
        return LogScopeInfo(
            scope=scope,
            is_system=(scope == self._system_scope),
            file_count=len(files),
            total_size=total,
        )

    def list_files(self, scope: str) -> List[LogFileInfo]:
        """Files in ``scope``, newest first (live file before its backups)."""
        infos = [self._file_info(p) for p in self._scope_files(scope)]
        infos.sort(key=lambda info: (info.rotation_index, -info.modified_at))
        return infos

    def _file_info(self, path: Path) -> LogFileInfo:
        return LogFileInfo(
            name=path.name,
            size=self._safe_size(path),
            modified_at=self._safe_mtime(path),
            rotation_index=_rotation_index(path),
            compressed=_is_compressed(path),
        )

    @staticmethod
    def _safe_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def read_text(
        self,
        scope: str,
        name: str,
        *,
        tail_lines: Optional[int] = None,
        max_bytes: int = 5 * 1024 * 1024,
    ) -> str:
        """Decoded text of a log file, transparently decompressing a gzipped
        backup. Returns at most the last ``tail_lines`` lines (when given) and
        never more than ``max_bytes`` of decoded text (truncated from the front)."""
        path = self.resolve_file(scope, name)
        opener = gzip.open if _is_compressed(path) else open

        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            if tail_lines is not None:
                lines = handle.readlines()[-tail_lines:]
                text = "".join(lines)
            else:
                text = handle.read()

        if len(text) > max_bytes:
            text = text[-max_bytes:]

        return text

    def read_tail_lines(
        self,
        scope: str,
        name: str,
        max_lines: int,
        *,
        max_bytes: int = 5 * 1024 * 1024,
    ) -> Tuple[str, bool]:
        """The last ``max_lines`` complete lines of a log file, read cheaply.

        For an uncompressed file this seeks from the end and reads fixed-size
        blocks backwards until it has gathered ``max_lines`` complete lines (or
        reached the start of the file), so a 30 MB live log is never read in full
        just to show its tail. A gzipped backup cannot be seeked, so it falls back
        to the whole-file tail in :meth:`read_text` (only hit when paging deep
        history). Never reads more than ``max_bytes`` from the end.

        Returns ``(text, truncated)``: ``text`` is in file order (oldest line
        first), the same orientation as :meth:`read_text`; ``truncated`` is True
        when the start of the file was cut off (by the line cap, the byte ceiling
        or a mid-line seek), so the caller knows the first logical line may be a
        fragment of an earlier entry (e.g. a continued traceback) to drop.
        """
        if max_lines <= 0:
            return "", False

        path = self.resolve_file(scope, name)
        if _is_compressed(path):
            text = self.read_text(scope, name, max_bytes=max_bytes)
            lines = text.splitlines()
            truncated = len(lines) > max_lines or len(text) >= max_bytes
            lines = lines[-max_lines:]
            return ("\n".join(lines) + "\n" if lines else ""), truncated

        size = self._safe_size(path)
        if size == 0:
            return "", False

        # Read blocks backwards until we have one more newline than lines wanted
        # (the extra one lets us drop the partial leading line), reach the start of
        # the file, or hit the byte ceiling.
        block = 64 * 1024
        want_newlines = max_lines + 1
        data = b""
        pos = size
        with open(path, "rb") as handle:
            while pos > 0 and len(data) < max_bytes:
                read_size = min(block, pos)
                pos -= read_size
                handle.seek(pos)
                data = handle.read(read_size) + data
                if data.count(b"\n") >= want_newlines:
                    break

        # We read the whole file only if we walked all the way back to byte 0 (so
        # line 0 is real); otherwise the seek landed mid-line and line 0 is a
        # fragment to drop.
        reached_start = pos == 0
        if len(data) > max_bytes:
            data = data[-max_bytes:]
            reached_start = False

        lines = data.decode("utf-8", errors="replace").splitlines()
        # If we stopped before the start of the file the first line is a fragment of
        # an earlier line (the seek landed mid-line) -- drop it.
        if not reached_start and lines:
            lines = lines[1:]
        truncated = not reached_start or len(lines) > max_lines
        lines = lines[-max_lines:]
        return ("\n".join(lines) + "\n" if lines else ""), truncated

    def open_bytes(self, scope: str, name: str) -> BinaryIO:
        """Raw bytes of a log file, for streaming a download."""
        return open(self.resolve_file(scope, name), "rb")

    def bundle_zip(self, scopes: Optional[Iterable[str]] = None) -> io.BytesIO:
        """A zip of the given scopes (all of them by default), each file stored
        under ``<scope>/<name>``."""
        wanted = (
            list(scopes)
            if scopes is not None
            else [s.scope for s in self.list_scopes()]
        )

        buffer = io.BytesIO()
        with zipfile.ZipFile(
            buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        ) as archive:
            for scope in wanted:
                for path in self._scope_files(scope):
                    archive.write(path, f"{scope}/{path.name}")

        buffer.seek(0)
        return buffer

    def delete_file(self, scope: str, name: str) -> None:
        self.resolve_file(scope, name).unlink(missing_ok=True)

    def delete_scope(self, scope: str) -> None:
        """Delete a printer scope's directory. The system scope can't be deleted."""
        if scope == self._system_scope:
            raise ValueError("The system log scope cannot be deleted")
        if not self._scope_token_ok(scope):
            raise LogNotFound(scope)

        scope_dir = self._root / scope
        if not scope_dir.is_dir():
            raise LogNotFound(scope)

        shutil.rmtree(scope_dir, ignore_errors=True)

    def strip_system_raw_logs(self, max_size: int = 50 * 1024 * 1024) -> None:
        """Tail-truncate the unbounded macOS raw capture logs to ``max_size``."""
        from simplyprint_ws_client.core.files.file_backup import FileBackup

        for name in ("stderr.log", "stdout.log"):
            FileBackup.strip_log_file(self._root / name, max_size=max_size)

    def prune_unused_scopes(self, active_unique_ids: Iterable[str]) -> None:
        """Remove per-printer directories for printers that no longer exist.

        The system scope and any active ``unique_id`` are always kept.
        """
        if not self._root.exists():
            return

        keep = {self._system_scope, *active_unique_ids}

        for child in self._root.iterdir():
            if not child.is_dir() or child.name in keep:
                continue
            shutil.rmtree(child, ignore_errors=True)

    def compress_rotated_files(self) -> None:
        """Gzip every uncompressed ``*.log.N`` backup in place (idempotent),
        keeping the name so the rotator's backup count still holds."""
        if not self._root.exists():
            return

        for path in self._iter_all_log_files():
            # Only rotated backups (".log.N"), and only if not already gzipped.
            if _rotation_index(path) == 0 or _is_compressed(path):
                continue

            tmp = path.with_suffix(f"{path.suffix}.gz")
            try:
                # mtime=0 -> deterministic output, independent of the wall clock.
                with (
                    open(path, "rb") as source,
                    gzip.GzipFile(tmp, "wb", mtime=0) as dest,
                ):
                    shutil.copyfileobj(source, dest)
                shutil.move(tmp, path)
            except FileNotFoundError:
                pass

    def _iter_all_log_files(self):
        def walk(directory: Path):
            for item in directory.iterdir():
                if item.is_dir():
                    yield from walk(item)
                elif _is_log_file(item):
                    yield item

        yield from walk(self._root)
