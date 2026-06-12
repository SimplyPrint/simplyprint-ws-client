import logging
from io import BytesIO
from types import SimpleNamespace

import pytest

from simplyprint_ws_client import FileDemandData, FileProgressStateEnum
from simplyprint_ws_client.core.files import file_download as file_download_module
from simplyprint_ws_client.core.files.file_download import FileDownload, FileDownloadError
from simplyprint_ws_client.integration.transfer import download_to_file


class FakeContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_chunked(self, _chunk_size):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    def __init__(self, chunks, *, error=None):
        self.content = FakeContent(chunks)
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def raise_for_status(self):
        if self._error is not None:
            raise self._error


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        return self.response


class FakeAioContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class FakeAioResponse:
    def __init__(self, chunks=(), *, status=200, headers=None):
        self.content = FakeAioContent(chunks)
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None


class FakeAioSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def get(self, url):
        self.urls.append(url)
        return self.responses.pop(0)


def patch_aio_session(monkeypatch, session):
    monkeypatch.setattr(
        file_download_module.aiohttp,
        "ClientSession",
        lambda timeout: session,
    )


@pytest.mark.asyncio
async def test_download_to_file_writes_flushes_and_rewinds():
    session = FakeSession(FakeResponse([b"abc"]))
    file = BytesIO()
    progress = SimpleNamespace(state=None, percent=None, message=None)

    downloaded = await download_to_file(
        url="https://example.test/file.gcode",
        file=file,
        file_progress=progress,
        logger=logging.getLogger(__name__),
        file_name="file.gcode",
        file_size=3,
        session=session,
    )

    assert downloaded == 3
    assert file.tell() == 0
    assert file.read() == b"abc"
    assert progress.state == FileProgressStateEnum.DOWNLOADING
    assert progress.percent == 50.0
    assert session.urls == ["https://example.test/file.gcode"]


@pytest.mark.asyncio
async def test_download_to_file_marks_error_on_size_mismatch():
    session = FakeSession(FakeResponse([b"abc"]))
    progress = SimpleNamespace(state=None, percent=None, message=None)

    with pytest.raises(ValueError, match="size mismatch"):
        await download_to_file(
            url="https://example.test/file.gcode",
            file=BytesIO(),
            file_progress=progress,
            logger=logging.getLogger(__name__),
            file_name="file.gcode",
            file_size=4,
            session=session,
        )

    assert progress.state == FileProgressStateEnum.ERROR
    assert "size mismatch" in progress.message


@pytest.mark.asyncio
async def test_file_download_rejects_missing_urls(client):
    downloader = FileDownload(client)

    with pytest.raises(FileDownloadError, match="No file URL provided"):
        await downloader.download_as_bytes(FileDemandData(file_name="file.gcode"))

    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR
    assert client.printer.file_progress.message == "No file URL provided"


@pytest.mark.asyncio
async def test_file_download_rejects_empty_success_response(client, tmp_path, monkeypatch):
    session = FakeAioSession([FakeAioResponse([], headers={"content-length": "0"})])
    patch_aio_session(monkeypatch, session)
    downloader = FileDownload(client)

    with pytest.raises(FileDownloadError, match="was empty"):
        await downloader.download_as_file(
            FileDemandData(file_name="file.gcode", cdn_url="https://cdn.test/file.gcode"),
            tmp_path / "file.gcode",
        )

    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR
    assert client.printer.file_progress.message == (
        "Downloaded file from https://cdn.test/file.gcode was empty"
    )
    assert session.urls == ["https://cdn.test/file.gcode"]


@pytest.mark.asyncio
async def test_file_download_falls_back_after_empty_primary(client, tmp_path, monkeypatch):
    session = FakeAioSession(
        [
            FakeAioResponse([], headers={"content-length": "0"}),
            FakeAioResponse([b"abc"], headers={"content-length": "3"}),
        ]
    )
    patch_aio_session(monkeypatch, session)
    downloader = FileDownload(client)

    dest = await downloader.download_as_file(
        FileDemandData(
            file_name="file.gcode",
            cdn_url="https://cdn.test/file.gcode",
            url="https://fallback.test/file.gcode",
        ),
        tmp_path / "file.gcode",
    )

    assert dest.read_bytes() == b"abc"
    assert client.printer.file_progress.state == FileProgressStateEnum.DOWNLOADING
    assert client.printer.file_progress.percent == 100
    assert client.printer.file_progress.message is None
    assert session.urls == [
        "https://cdn.test/file.gcode",
        "https://fallback.test/file.gcode",
    ]
