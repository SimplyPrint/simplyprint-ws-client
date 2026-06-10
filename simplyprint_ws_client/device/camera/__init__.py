"""Camera machinery (pool, mixin, protocols) + reusable protocol implementations."""

from simplyprint_ws_client.device.camera.mjpeg import (
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
