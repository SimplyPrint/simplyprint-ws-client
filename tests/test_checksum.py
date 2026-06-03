"""Tests for the shared transfer checksum helpers."""

import asyncio
import hashlib

from simplyprint_ws_client.contrib.transfer.checksum import (
    file_md5,
    parse_s3_etag,
)


def test_parse_s3_etag():
    md5 = "d41d8cd98f00b204e9800998ecf8427e"
    # A single-part ETag is the file's md5 (quoted), normalised to uppercase.
    assert parse_s3_etag(f'"{md5}"') == md5.upper()
    # Weak-validator prefix is stripped.
    assert parse_s3_etag(f'W/"{md5}"') == md5.upper()
    # A multipart ETag ("<hash>-<parts>") is NOT a whole-file md5 -> None.
    assert parse_s3_etag(f'"{md5}-3"') is None
    # Absent / empty -> None.
    assert parse_s3_etag(None) is None
    assert parse_s3_etag("") is None


def test_file_md5(tmp_path):
    data = b"simplyprint" * 100_000  # a few MB, exercises chunked streaming
    path = tmp_path / "blob.gcode"
    path.write_bytes(data)
    expected = hashlib.md5(data).hexdigest().upper()

    # file_md5 streams + hashes off-loop, matching hashlib (uppercase).
    assert asyncio.run(file_md5(path)) == expected
