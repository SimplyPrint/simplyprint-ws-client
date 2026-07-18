"""Behavioral tests for the pure source downloader."""

import aiohttp
import pytest

from simplyprint_ws_client import FileDemandData
from simplyprint_ws_client.integration.transfer import FileDownloadError, download_file
from simplyprint_ws_client.integration.transfer import download as download_module


@pytest.fixture(autouse=True)
def _inline_file_io(monkeypatch):
    async def immediate(function, *args):
        return function(*args)

    monkeypatch.setattr(download_module.asyncio, "to_thread", immediate)


class Content:
    def __init__(self, chunks, error=None):
        self.chunks = list(chunks)
        self.error = error

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class Response:
    def __init__(self, chunks=(), *, status=200, headers=None, error=None):
        self.content = Content(chunks, error)
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None,
                history=(),
                status=self.status,
            )


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_missing_urls_are_rejected(tmp_path):
    with pytest.raises(FileDownloadError, match="No file URL"):
        await download_file(
            FileDemandData(file_name="job.gcode"),
            tmp_path / "job.gcode",
            lambda _percent: None,
        )


@pytest.mark.asyncio
async def test_download_writes_and_reports_byte_progress(tmp_path):
    progress = []
    session = Session([Response([b"ab", b"cd"], headers={})])
    destination = tmp_path / "job.gcode"
    result = await download_file(
        FileDemandData(
            file_name="job.gcode",
            file_size=4,
            cdn_url="https://cdn.test/job.gcode",
        ),
        destination,
        progress.append,
        session=session,
    )
    assert result == destination
    assert destination.read_bytes() == b"abcd"
    assert progress == [50.0, 100.0, 100.0]


@pytest.mark.asyncio
async def test_empty_primary_falls_back_without_concatenation(tmp_path):
    session = Session(
        [
            Response([], headers={"content-length": "0"}),
            Response([b"fallback"], headers={"content-length": "8"}),
        ]
    )
    destination = tmp_path / "job.gcode"
    await download_file(
        FileDemandData(
            file_name="job.gcode",
            cdn_url="https://cdn.test/job.gcode",
            url="https://fallback.test/job.gcode",
        ),
        destination,
        lambda _percent: None,
        session=session,
    )
    assert destination.read_bytes() == b"fallback"
    assert session.urls == [
        "https://cdn.test/job.gcode",
        "https://fallback.test/job.gcode",
    ]


@pytest.mark.asyncio
async def test_partial_primary_is_truncated_before_fallback(tmp_path):
    session = Session(
        [
            Response([b"partial"], error=OSError("link lost")),
            Response([b"good"]),
        ]
    )
    destination = tmp_path / "job.gcode"
    await download_file(
        FileDemandData(
            file_name="job.gcode",
            cdn_url="https://cdn.test/job.gcode",
            url="https://fallback.test/job.gcode",
        ),
        destination,
        lambda _percent: None,
        session=session,
    )
    assert destination.read_bytes() == b"good"


@pytest.mark.asyncio
async def test_expected_size_mismatch_is_terminal(tmp_path):
    with pytest.raises(FileDownloadError, match="size mismatch"):
        await download_file(
            FileDemandData(
                file_name="job.gcode",
                file_size=5,
                cdn_url="https://cdn.test/job.gcode",
            ),
            tmp_path / "job.gcode",
            lambda _percent: None,
            session=Session([Response([b"four"])]),
        )
