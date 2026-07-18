"""MJPEG camera protocols and frame parsing.

The protocol mechanics are printer-agnostic: a configured camera URL either
returns one JPEG snapshot or a continuous multipart/raw JPEG stream. Integrations
choose which protocols a printer type exposes; the parser and transport live here
once.
"""

from __future__ import annotations

import re
import ssl
from collections.abc import Iterator
from typing import Optional, Union

import aiohttp
from yarl import URL

from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolConnectionError,
    CameraProtocolPollingMode,
)

__all__ = [
    "MJPEGFrameParser",
    "MJPEGSnapshotCamera",
    "MJPEGStreamCamera",
    "extract_jpeg_frame",
]

_JPEG_START = b"\xff\xd8"
_JPEG_END = b"\xff\xd9"
_HEADER_SEPARATOR = b"\r\n\r\n"
_TIMEOUT = 10
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_CHUNK_SIZE = 2**16


def _to_http_url(uri: URL) -> str:
    scheme = uri.scheme

    if scheme in ("mjpeg", "mjpeg-stream"):
        return str(uri.with_scheme("http"))

    return str(uri)


def _create_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _extract_boundary(content_type: str) -> Optional[str]:
    match = re.search(r'boundary=(?:"([^"]+)"|([^\s;]+))', content_type, re.IGNORECASE)
    if match:
        boundary = match.group(1) or match.group(2)
        return boundary[2:] if boundary.startswith("--") else boundary
    return None


def extract_jpeg_frame(data: Union[bytes, bytearray, memoryview]) -> Optional[bytes]:
    raw = bytes(data)
    start = raw.find(_JPEG_START)
    end = raw.find(_JPEG_END, start)

    if start == -1 or end == -1:
        return None

    return bytes(raw[start : end + len(_JPEG_END)])


def _parse_content_length(header_block: bytes) -> Optional[int]:
    for header_line in header_block.split(b"\r\n"):
        if b":" not in header_line:
            continue

        header_key, header_value = header_line.split(b":", 1)
        if header_key.lower() != b"content-length":
            continue

        try:
            return int(header_value.strip())
        except ValueError:
            return None

    return None


class MJPEGFrameParser:
    def __init__(self, content_type: str = "") -> None:
        boundary = _extract_boundary(content_type)
        self.boundary = b"--" + boundary.encode() if boundary is not None else None
        self.buffer = bytearray()
        self.in_raw_frame = False

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        self.buffer.extend(chunk)

        if self.boundary is None:
            yield from self._feed_raw()
            return

        yield from self._feed_multipart()

    def _feed_multipart(self) -> Iterator[bytes]:
        while True:
            idx = self.buffer.find(self.boundary)
            if idx == -1:
                break

            next_idx = self.buffer.find(self.boundary, idx + len(self.boundary))

            if next_idx != -1:
                part = self.buffer[idx + len(self.boundary) : next_idx]
                frame = extract_jpeg_frame(part)
                if frame is not None:
                    yield frame

                del self.buffer[:next_idx]
                continue

            frame = self._extract_content_length_frame(idx + len(self.boundary))
            if frame is None:
                break

            frame_data, frame_end = frame
            jpeg_frame = extract_jpeg_frame(frame_data)
            if jpeg_frame is not None:
                yield jpeg_frame

            del self.buffer[:frame_end]

    def _extract_content_length_frame(
        self, part_start: int
    ) -> Optional[tuple[bytes, int]]:
        while self.buffer[part_start : part_start + 2] == b"\r\n":
            part_start += 2

        header_end = self.buffer.find(_HEADER_SEPARATOR, part_start)
        if header_end == -1:
            return None

        header_block = bytes(self.buffer[part_start:header_end])
        content_length = _parse_content_length(header_block)
        if content_length is None:
            return None

        content_start = header_end + len(_HEADER_SEPARATOR)
        content_end = content_start + content_length
        if len(self.buffer) < content_end:
            return None

        return bytes(self.buffer[content_start:content_end]), content_end

    def _feed_raw(self) -> Iterator[bytes]:
        while True:
            if not self.in_raw_frame:
                start = self.buffer.find(_JPEG_START)

                if start == -1:
                    self.buffer = self.buffer[-1:]
                    break

                del self.buffer[:start]
                self.in_raw_frame = True

            end = self.buffer.find(_JPEG_END)

            if end == -1:
                break

            end += len(_JPEG_END)
            yield bytes(self.buffer[:end])
            del self.buffer[:end]
            self.in_raw_frame = False


class MJPEGSnapshotCamera(BaseCameraProtocol):
    """Captures a single JPEG frame from an HTTP(S) or mjpeg:// snapshot URL."""

    polling_mode = CameraProtocolPollingMode.ON_DEMAND
    is_async = True

    @staticmethod
    def test(uri: URL) -> bool:
        return uri.scheme in ("http", "https", "mjpeg")

    async def read(self):
        url = _to_http_url(self.uri)
        ctx = _create_ssl_context() if self.uri.scheme == "https" else None
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
        raw_data = bytearray()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, ssl=ctx) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if (
                    "multipart" in content_type.lower()
                    and _extract_boundary(content_type) is None
                ):
                    raise CameraProtocolConnectionError(
                        "Multipart response without boundary."
                    )
                async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                    raw_data.extend(chunk)
                    frame = extract_jpeg_frame(raw_data)
                    if frame is not None:
                        yield frame
                        return
                    if len(raw_data) > _MAX_FRAME_BYTES:
                        raise CameraProtocolConnectionError(
                            "Camera frame exceeded the 8 MiB limit."
                        )

        # Preserve support for non-JPEG snapshot endpoints. SimplyPrint usually
        # receives JPEG, but the old implementation forwarded any non-empty body.
        if raw_data:
            yield bytes(raw_data)
            return
        raise CameraProtocolConnectionError("No image data received from the camera.")


class MJPEGStreamCamera(BaseCameraProtocol):
    """Reads a continuous MJPEG stream from an mjpeg-stream:// URL."""

    polling_mode = CameraProtocolPollingMode.CONTINUOUS
    is_async = True

    @staticmethod
    def test(uri: URL) -> bool:
        return uri.scheme == "mjpeg-stream"

    async def read(self):
        url = _to_http_url(self.uri)
        timeout = aiohttp.ClientTimeout(
            total=None, connect=_TIMEOUT, sock_read=_TIMEOUT
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                parser = MJPEGFrameParser(resp.headers.get("Content-Type", ""))
                async for chunk in resp.content.iter_chunked(_CHUNK_SIZE):
                    for frame in parser.feed(chunk):
                        yield frame
                    if len(parser.buffer) > _MAX_FRAME_BYTES:
                        raise CameraProtocolConnectionError(
                            "MJPEG frame exceeded the 8 MiB limit."
                        )

        raise CameraProtocolConnectionError("MJPEG stream ended.")
