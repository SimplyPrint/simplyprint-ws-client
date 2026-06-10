"""Progress-reporting HTTP downloads to file-like objects."""

from __future__ import annotations

import asyncio
import logging
from typing import BinaryIO, Optional

import aiohttp

from simplyprint_ws_client import FileProgressState, FileProgressStateEnum


async def download_to_file(
    *,
    url: str,
    file: BinaryIO,
    file_progress: FileProgressState,
    logger: logging.Logger,
    file_name: Optional[str] = None,
    file_size: Optional[int] = None,
    progress_end: float = 50.0,
    chunk_size: int = 8192,
    session=None,
) -> int:
    """Download ``url`` into ``file`` while updating ``file_progress``.

    ``session`` is injectable for tests or callers that already own an
    ``aiohttp.ClientSession``. The function owns all file writes/flush/rewind and
    leaves the handle positioned at the beginning on success.
    """

    if session is None:
        async with aiohttp.ClientSession() as owned_session:
            return await _download_to_file_with_session(
                url=url,
                file=file,
                file_progress=file_progress,
                logger=logger,
                file_name=file_name,
                file_size=file_size,
                progress_end=progress_end,
                chunk_size=chunk_size,
                session=owned_session,
            )

    return await _download_to_file_with_session(
        url=url,
        file=file,
        file_progress=file_progress,
        logger=logger,
        file_name=file_name,
        file_size=file_size,
        progress_end=progress_end,
        chunk_size=chunk_size,
        session=session,
    )


async def _download_to_file_with_session(
    *,
    url: str,
    file: BinaryIO,
    file_progress: FileProgressState,
    logger: logging.Logger,
    file_name: Optional[str],
    file_size: Optional[int],
    progress_end: float,
    chunk_size: int,
    session,
) -> int:
    logger.info("Downloading file file_name: %s", file_name)
    logger.info("cdn_url: %s", url)
    logger.info("file_size: %s  progress: %s", file_size, file_progress.state)

    file_progress.state = FileProgressStateEnum.DOWNLOADING
    file_progress.percent = 0.0

    downloaded = 0

    try:
        async with session.get(url) as response:
            response.raise_for_status()

            async for chunk in response.content.iter_chunked(chunk_size):
                if not chunk:
                    continue

                await asyncio.to_thread(file.write, chunk)
                downloaded += len(chunk)

                if file_size:
                    file_progress.percent = min(
                        (downloaded / file_size) * progress_end,
                        progress_end,
                    )

        if downloaded == 0:
            raise ValueError("Downloaded file is empty")

        if file_size and downloaded != file_size:
            raise ValueError(
                f"Downloaded file size mismatch: expected {file_size}, got {downloaded}"
            )

        await asyncio.to_thread(file.flush)
        await asyncio.to_thread(file.seek, 0)
        file_progress.percent = progress_end

        return downloaded
    except Exception as exc:
        file_progress.state = FileProgressStateEnum.ERROR
        file_progress.message = f"Download failed: {exc}"
        logger.exception("Download failed")
        raise
