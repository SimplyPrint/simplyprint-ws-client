import asyncio
from pathlib import Path
from ssl import SSLError
from typing import Callable, Optional, AsyncIterable

import aiohttp
from aiohttp import ClientError

from simplyprint_ws_client.core.client import Client
from simplyprint_ws_client.core.state import FileProgressState, FileProgressStateEnum
from simplyprint_ws_client.core.protocol.messages import FileDemandData


class FileDownload:
    state: FileProgressState
    client: Client
    timeout: aiohttp.ClientTimeout

    def __init__(
        self, client: Client, timeout: Optional[aiohttp.ClientTimeout] = None
    ) -> None:
        self.client = client
        self.state = client.printer.file_progress

        self.timeout = (
            timeout
            or aiohttp.ClientTimeout(
                # default is total = 5 minutes, which is too short for large files
                total=None,  # Total number of seconds for the whole request
                connect=5,  # Maximal number of seconds for acquiring a connection from pool
                sock_connect=10,  # Maximal number of seconds for connecting to a peer for a new connection
                sock_read=60
                * 30,  # seconds for consecutive reads - 30 minutes as we do not control the block size
            )
        )

    async def download(
        self, data: FileDemandData, clamp_progress: Optional[Callable] = None
    ) -> AsyncIterable:
        """
        Download a file with file progress.
        """

        # Support fallback urls in case the primary one fails
        valid_urls = [data.cdn_url, data.url]

        # Chunk the download so we can get progress
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            while valid_urls:
                url = valid_urls.pop(0)

                if url is None:
                    continue

                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            self.state.message = (
                                f"Failed to download file: {resp.status}"
                            )
                            continue

                        self.state.state = FileProgressStateEnum.STARTED

                        size = int(resp.headers.get("content-length", 0))
                        downloaded = 0

                        self.state.state = FileProgressStateEnum.DOWNLOADING

                        # Download chunk by chunk
                        async for chunk in resp.content.iter_any():
                            yield chunk

                            downloaded += len(chunk)

                            total_percentage = int((downloaded / size) * 100)

                            self.state.percent = (
                                clamp_progress(total_percentage)
                                if clamp_progress
                                else total_percentage
                            )

                        break
                except (OSError, SSLError, ClientError, asyncio.TimeoutError) as e:
                    self.state.message = f"Failed to download file from {url}: {e}"
                    continue
            else:
                # If we exhausted all URLs and none worked, set the state to error.
                self.state.state = FileProgressStateEnum.ERROR

    async def download_as_bytes(
        self, data: FileDemandData, clamp_progress: Optional[Callable] = None
    ) -> bytes:
        # Bytes object to store the downloaded data
        content = b""

        async for chunk in self.download(data, clamp_progress):
            content += chunk

        # Return the data as a BytesIO object
        return content

    async def download_as_file(
        self,
        data: FileDemandData,
        dest: Path,
        clamp_progress: Optional[Callable] = None,
    ) -> Path:
        """Download a file with progress and write it to ``dest``.

        Each chunk is written on a worker thread so a large or slow disk never
        blocks the event loop between network reads.
        """
        with open(dest, "wb") as f:
            async for chunk in self.download(data, clamp_progress):
                await asyncio.to_thread(f.write, chunk)

        return dest
