"""Atomic, collision-proof file writes.

Write to a *unique* temp file in the same directory, ``fsync`` it, then
``os.replace`` it onto the target. ``os.replace`` is atomic within a filesystem,
so a crash mid-write can never leave a torn or empty file.

The temp name is unique (``tempfile.mkstemp``), never a fixed ``<name>.tmp``.
A fixed temp name is what makes two concurrent writers (two managers in one
process, or a CLI invocation racing the running service) unsafe: the first
writer's ``os.replace`` consumes the shared temp out from under the second,
which then fails with ``FileNotFoundError: <tmp> -> <target>``. A per-write
unique temp removes that race entirely.

The parent directory is (re)created right before the write, so a target whose
directory was removed at runtime is healed rather than failing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

__all__ = ["atomic_write_bytes", "atomic_write_text"]


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Atomically write ``data`` to ``path`` (see module docstring)."""
    target = Path(path)
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, target)
    except BaseException:
        # The replace never happened (or failed); drop the orphan temp so a
        # failed write does not litter the directory.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(
    path: Union[str, Path], text: str, encoding: str = "utf-8"
) -> None:
    """Atomically write ``text`` to ``path`` using ``encoding``."""
    atomic_write_bytes(path, text.encode(encoding))
