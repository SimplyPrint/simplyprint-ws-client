"""Checksum helpers shared by file-transfer code."""

from __future__ import annotations

import hashlib
from pathlib import Path


def fast_md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the uppercase hex MD5 of ``path``, read in large chunks."""
    md5 = hashlib.md5()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest().upper()
