import pytest
from yarl import URL

from simplyprint_ws_client.integration.camera import mjpeg
from simplyprint_ws_client.integration.camera.mjpeg import (
    MJPEGFrameParser,
    MJPEGSnapshotCamera,
    MJPEGStreamCamera,
)
from simplyprint_ws_client.integration.camera.base import CameraProtocolConnectionError


JPEG_ONE = b"\xff\xd8one\xff\xd9"
JPEG_TWO = b"\xff\xd8two\xff\xd9"


class FakeResponse:
    def __init__(self, *, headers=None, body=b"", chunks=None):
        self.headers = headers or {}
        self._body = body
        self._chunks = list(chunks or [])

        if not self._chunks and body:
            self._chunks = [body]
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk


def _patch_session(monkeypatch, response):
    calls = []

    class FakeSession:
        def __init__(self, *args, **kwargs):
            calls.append(("session", args, kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def get(self, url, *args, **kwargs):
            calls.append((url, args, kwargs))
            return response

    monkeypatch.setattr(mjpeg.aiohttp, "ClientSession", FakeSession)
    return calls


@pytest.mark.asyncio
async def test_snapshot_camera_extracts_multipart_frame_with_case_sensitive_boundary(
    monkeypatch,
):
    body = (
        b"--FrameBoundary\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        b"prefix" + JPEG_ONE + b"suffix\r\n"
        b"--FrameBoundary--\r\n"
    )
    response = FakeResponse(
        headers={"Content-Type": 'multipart/x-mixed-replace; boundary="FrameBoundary"'},
        body=body,
    )
    calls = _patch_session(monkeypatch, response)

    frames = [
        frame
        async for frame in MJPEGSnapshotCamera(
            URL("mjpeg://printer/snapshot")
        ).read()
    ]

    assert frames == [JPEG_ONE]
    assert calls[1][0] == "http://printer/snapshot"


@pytest.mark.asyncio
async def test_snapshot_camera_rejects_multipart_without_boundary(monkeypatch):
    response = FakeResponse(headers={"Content-Type": "multipart/x-mixed-replace"})
    _patch_session(monkeypatch, response)

    with pytest.raises(CameraProtocolConnectionError, match="without boundary"):
        async for _ in MJPEGSnapshotCamera(URL("http://printer/snapshot")).read():
            pass


@pytest.mark.asyncio
async def test_stream_camera_extracts_multipart_frames_across_chunks(monkeypatch):
    response = FakeResponse(
        headers={"Content-Type": "multipart/x-mixed-replace; boundary=FrameBoundary"},
        chunks=[
            b"--FrameBoundary\r\nContent-Type: image/jpeg\r\n\r\n" + JPEG_ONE[:4],
            JPEG_ONE[4:]
            + b"\r\n--FrameBoundary\r\nContent-Type: image/jpeg\r\n\r\n"
            + JPEG_TWO,
            b"\r\n--FrameBoundary\r\n",
        ],
    )
    calls = _patch_session(monkeypatch, response)

    frames = []
    with pytest.raises(CameraProtocolConnectionError, match="stream ended"):
        async for frame in MJPEGStreamCamera(
            URL("mjpeg-stream://printer/stream")
        ).read():
            frames.append(frame)

    assert frames == [JPEG_ONE, JPEG_TWO]
    assert calls[1][0] == "http://printer/stream"


def test_frame_parser_uses_content_length_before_next_boundary():
    parser = MJPEGFrameParser("multipart/x-mixed-replace; boundary=FrameBoundary")

    frames = list(
        parser.feed(
            b"--FrameBoundary\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: 7\r\n"
            b"\r\n" + JPEG_ONE
        )
    )

    assert frames == [JPEG_ONE]


@pytest.mark.asyncio
async def test_stream_camera_extracts_raw_jpeg_frames_without_boundary(monkeypatch):
    response = FakeResponse(
        headers={"Content-Type": "image/jpeg"},
        chunks=[b"noise" + JPEG_ONE + b"middle" + JPEG_TWO],
    )
    _patch_session(monkeypatch, response)

    frames = []
    with pytest.raises(CameraProtocolConnectionError, match="stream ended"):
        async for frame in MJPEGStreamCamera(
            URL("mjpeg-stream://printer/stream")
        ).read():
            frames.append(frame)

    assert frames == [JPEG_ONE, JPEG_TWO]
