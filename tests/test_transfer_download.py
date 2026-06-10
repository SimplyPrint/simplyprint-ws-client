import logging
from io import BytesIO
from types import SimpleNamespace

import pytest

from simplyprint_ws_client import FileProgressStateEnum
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
