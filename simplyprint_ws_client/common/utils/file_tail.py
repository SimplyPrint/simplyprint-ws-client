"""Generic file tail-truncation primitive."""

from __future__ import annotations

from pathlib import Path

__all__ = ["strip_log_file"]


def strip_log_file(file: Path, max_size: int = 100 * 1024 * 1024) -> None:
    """Strip a log file to a maximum size, keeping the tail."""

    if not file.exists():
        return

    if file.stat().st_size <= max_size:
        return

    # Use the size to start seeking from the end of the file
    # and then read the file in chunks of 1024 bytes until we have read the last size
    # then overwrite the file with the new content
    with open(file, "rb+") as f:
        f.seek(-max_size, 2)
        data = f.read()
        f.seek(0)
        f.write(data)
        f.truncate()
