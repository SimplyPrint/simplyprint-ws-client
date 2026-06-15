from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

Pathish = Union[str, os.PathLike]

_TEMP_ENV_VARS: Tuple[str, ...] = ("TMPDIR", "TEMP", "TMP")
_SYSTEM_TEMP_ROOTS: Tuple[Path, ...] = (Path("/tmp"), Path("/private/tmp"))


@dataclass(frozen=True)
class TempPruneResult:
    scanned: int = 0
    deleted: int = 0
    errors: int = 0


def app_cache_path(*parts: str) -> Path:
    """Return a path under the app cache directory."""
    from simplyprint_ws_client.const import APP_DIRS

    path = Path(APP_DIRS.user_cache_path)
    for part in parts:
        path /= part
    return path


def cache_temporary_directory(
    prefix: Optional[str] = None,
    *,
    root: Optional[Pathish] = None,
):
    """Create a temporary directory under app-controlled cache storage.

    This deliberately avoids ``tempfile``'s system default, which is often a
    tmpfs on Linux/Raspberry Pi and can be exhausted by OTA/file-transfer work.
    """
    directory = Path(root) if root is not None else app_cache_path("tmp")
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=str(directory))


def configure_process_tempdir(
    root: Optional[Pathish] = None,
    *,
    respect_existing: bool = True,
) -> Path:
    """Point process tempfile defaults at cache storage.

    Existing non-system temp env vars are preserved by default. Missing env vars,
    and vars pointing at ``/tmp`` or macOS' ``/private/tmp`` alias, are replaced.
    The stdlib cache is updated too so later ``tempfile`` callers follow the same
    root even if ``tempfile.gettempdir()`` was evaluated earlier.
    """
    directory = None
    if respect_existing:
        directory = _existing_non_system_tempdir()
    if directory is None:
        directory = Path(root) if root is not None else app_cache_path("tmp")

    directory.mkdir(parents=True, exist_ok=True)
    for name in _TEMP_ENV_VARS:
        current = os.environ.get(name)
        if not current or _is_system_temp(current):
            os.environ[name] = str(directory)
    tempfile.tempdir = str(directory)
    return directory


def prune_stale_temporary_entries(
    root: Pathish,
    *,
    older_than_seconds: float,
    prefixes: Optional[Iterable[str]] = None,
    now: Optional[float] = None,
) -> TempPruneResult:
    """Delete stale temp entries inside one owned temp root.

    The function is intentionally narrow: it only inspects direct children of
    ``root`` and, when ``prefixes`` is given, only entries whose names start with
    one of those known temp prefixes. Age is based on ``mtime`` rather than
    ``atime`` because many Linux/RPi filesystems mount with ``noatime`` or
    ``relatime``.
    """
    directory = Path(root)
    try:
        children = list(directory.iterdir())
    except FileNotFoundError:
        return TempPruneResult()

    allowed_prefixes = tuple(prefixes or ())
    cutoff = (time.time() if now is None else now) - older_than_seconds
    scanned = 0
    deleted = 0
    errors = 0

    for child in children:
        if allowed_prefixes and not child.name.startswith(allowed_prefixes):
            continue
        scanned += 1
        try:
            stat = child.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            errors += 1
            continue
        if stat.st_mtime > cutoff:
            continue
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except OSError:
            errors += 1

    return TempPruneResult(scanned=scanned, deleted=deleted, errors=errors)


def _existing_non_system_tempdir() -> Optional[Path]:
    for name in _TEMP_ENV_VARS:
        value = os.environ.get(name)
        if value and not _is_system_temp(value):
            return Path(value)
    return None


def _is_system_temp(value: str) -> bool:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()

    for root in _SYSTEM_TEMP_ROOTS:
        if resolved == root or root in resolved.parents:
            return True
    return False
