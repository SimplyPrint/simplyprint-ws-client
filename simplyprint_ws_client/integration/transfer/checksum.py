"""Checksum helpers shared by file-transfer code.

One place to answer "what is this transfer file's MD5", however we can cheapest
get it: from an S3 (CDN) ETag without touching the file, or by hashing it off
the event loop. :func:`fast_md5sum` is the sync streaming leaf; everything else
is the async/ETag-aware surface callers should prefer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from simplyprint_ws_client.common.asyncio.concurrent import run_in_thread


def fast_md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the uppercase hex MD5 of ``path``, read in large chunks."""
    md5 = hashlib.md5()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest().upper()


async def file_md5(path: Path) -> str:
    """Uppercase hex MD5 of ``path``, streamed and hashed off the event loop.

    The whole read+hash runs on a worker thread, so a large file never blocks
    the loop and the file is never slurped into memory.
    """
    return await run_in_thread(fast_md5sum, path, thread_name="file-md5")


def parse_s3_etag(etag: Optional[str]) -> Optional[str]:
    """Return the uppercase MD5 an S3 ETag encodes, or ``None`` if it doesn't.

    SimplyPrint's CDN is S3-backed, so a single-part object's ETag *is* the
    file's MD5 (quoted, optionally with a weak-validator ``W/`` prefix). A
    multipart upload's ETag is ``"<hash>-<partcount>"`` -- not a whole-file MD5
    -- so we return ``None`` and let the caller fall back to computing one.
    """
    if not etag:
        return None

    value = etag.strip()
    if value.startswith("W/"):
        value = value[2:]
    value = value.strip('"')

    if not value or "-" in value:  # empty or multipart
        return None

    return value.upper()
