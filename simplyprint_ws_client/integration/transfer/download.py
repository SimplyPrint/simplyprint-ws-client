"""Pure HTTP source-file download used by the print preparation controller."""

from __future__ import annotations

import asyncio
from pathlib import Path
from ssl import SSLError
from typing import Callable, Optional

import aiohttp
from aiohttp import ClientError

from simplyprint_ws_client.core.protocol.messages import FileDemandData


class FileDownloadError(Exception):
    """The requested source could not be downloaded and validated."""


DEFAULT_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(
    total=None,
    connect=120,
    sock_connect=120,
    # Operation-level stall detection belongs to FileTransfer's watchdog.
    sock_read=None,
)


async def download_file(
    data: FileDemandData,
    destination: Path,
    progress: Callable[[float], None],
    *,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> Path:
    """Download one validated source artifact into ``destination``.

    The primary CDN URL and fallback URL are attempted in that order. Each
    attempt owns/truncates the destination, so a partially delivered primary
    can never be concatenated with its fallback. This function has no client or
    printer-state dependency; callers own progress and terminal reporting.
    """

    urls = tuple(dict.fromkeys(url for url in (data.cdn_url, data.url) if url))
    if not urls:
        raise FileDownloadError("No file URL provided")

    if session is None:
        async with aiohttp.ClientSession(
            timeout=timeout or DEFAULT_DOWNLOAD_TIMEOUT
        ) as owned_session:
            return await _download_with_session(
                data, destination, progress, urls, owned_session
            )

    return await _download_with_session(data, destination, progress, urls, session)


async def _download_with_session(
    data: FileDemandData,
    destination: Path,
    progress: Callable[[float], None],
    urls: tuple[str, ...],
    session: aiohttp.ClientSession,
) -> Path:
    last_error: Optional[BaseException] = None

    for url in urls:
        downloaded = 0
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                expected = data.file_size or int(
                    response.headers.get("content-length") or 0
                )

                with destination.open("wb") as output:
                    async for chunk in response.content.iter_any():
                        if not chunk:
                            continue
                        await asyncio.to_thread(output.write, chunk)
                        downloaded += len(chunk)
                        if expected:
                            progress(min(downloaded / expected * 100.0, 100.0))
                    await asyncio.to_thread(output.flush)

            if downloaded == 0:
                raise FileDownloadError(f"Downloaded file from {url} was empty")
            if data.file_size and downloaded != data.file_size:
                raise FileDownloadError(
                    "Downloaded file size mismatch: "
                    f"expected {data.file_size}, got {downloaded}"
                )

            progress(100.0)
            return destination
        except (
            OSError,
            SSLError,
            ClientError,
            asyncio.TimeoutError,
            FileDownloadError,
        ) as error:
            last_error = error

    if isinstance(last_error, FileDownloadError):
        raise last_error
    if last_error is not None:
        raise FileDownloadError(
            f"Failed to download file: {last_error}"
        ) from last_error
    raise FileDownloadError("Failed to download file")


__all__ = ["FileDownloadError", "download_file"]
