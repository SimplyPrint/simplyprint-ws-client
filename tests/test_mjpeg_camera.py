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

    def read(self, _size=-1):
        if self._chunks:
            return self._chunks.pop(0)
        return self._body


def _patch_urlopen(monkeypatch, response):
    calls = []

    def urlopen(request, *args, **kwargs):
        calls.append((request, args, kwargs))
        return response

    monkeypatch.setattr(mjpeg.urllib.request, "urlopen", urlopen)
    return calls


def test_snapshot_camera_extracts_multipart_frame_with_case_sensitive_boundary(
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
    calls = _patch_urlopen(monkeypatch, response)

    frames = list(MJPEGSnapshotCamera(URL("mjpeg://printer/snapshot")).read())

    assert frames == [JPEG_ONE]
    assert calls[0][0].full_url == "http://printer/snapshot"


def test_snapshot_camera_rejects_multipart_without_boundary(monkeypatch):
    response = FakeResponse(headers={"Content-Type": "multipart/x-mixed-replace"})
    _patch_urlopen(monkeypatch, response)

    with pytest.raises(CameraProtocolConnectionError, match="without boundary"):
        list(MJPEGSnapshotCamera(URL("http://printer/snapshot")).read())


def test_stream_camera_extracts_multipart_frames_across_chunks(monkeypatch):
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
    calls = _patch_urlopen(monkeypatch, response)

    frames = MJPEGStreamCamera(URL("mjpeg-stream://printer/stream")).read()

    assert next(frames) == JPEG_ONE
    assert next(frames) == JPEG_TWO
    assert calls[0][0].full_url == "http://printer/stream"


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


def test_stream_camera_extracts_raw_jpeg_frames_without_boundary(monkeypatch):
    response = FakeResponse(
        headers={"Content-Type": "image/jpeg"},
        chunks=[b"noise" + JPEG_ONE + b"middle" + JPEG_TWO],
    )
    _patch_urlopen(monkeypatch, response)

    frames = MJPEGStreamCamera(URL("mjpeg-stream://printer/stream")).read()

    assert next(frames) == JPEG_ONE
    assert next(frames) == JPEG_TWO
