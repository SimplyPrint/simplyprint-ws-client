"""Reusable camera protocol implementations."""

from simplyprint_ws_client.contrib.camera.mjpeg import (
    MJPEGFrameParser,
    MJPEGSnapshotCamera,
    MJPEGStreamCamera,
    extract_jpeg_frame,
)

__all__ = [
    "MJPEGFrameParser",
    "MJPEGSnapshotCamera",
    "MJPEGStreamCamera",
    "extract_jpeg_frame",
]
