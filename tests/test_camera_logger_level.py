"""A printer's camera logger: its level set without clearing every logger's
cache, and no warning per report when the app runs without cameras."""

from __future__ import annotations

import logging
import uuid

from yarl import URL

from simplyprint_ws_client import ClientContext, PrinterClient, PrinterConfig
from simplyprint_ws_client.integration.camera.controller import _child_with_level


def _name() -> str:
    return f"camera-level-test-{uuid.uuid4().hex}"


def test_a_new_logger_gets_its_level_without_clearing_the_process_caches(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        logging.Logger.manager, "_clear_cache", lambda: cleared.append(True)
    )
    parent = logging.getLogger(_name())

    child = _child_with_level(parent, "camera", logging.INFO)

    assert child is parent.getChild("camera")
    assert child.level == logging.INFO
    assert cleared == []


def test_an_existing_logger_goes_through_set_level():
    parent = logging.getLogger(_name())
    child = parent.getChild("camera")
    grandchild = child.getChild("frames")
    child.setLevel(logging.DEBUG)
    assert grandchild.isEnabledFor(logging.DEBUG)  # cached now

    assert _child_with_level(parent, "camera", logging.INFO) is child

    assert child.level == logging.INFO
    assert not grandchild.isEnabledFor(logging.DEBUG)  # the cache was cleared


def test_a_placeholder_with_children_goes_through_set_level():
    parent = logging.getLogger(_name())
    grandchild = logging.getLogger(f"{parent.name}.camera.frames")
    parent.setLevel(logging.DEBUG)
    assert grandchild.isEnabledFor(logging.DEBUG)  # cached through the placeholder

    child = _child_with_level(parent, "camera", logging.INFO)

    assert child.level == logging.INFO
    assert not grandchild.isEnabledFor(logging.DEBUG)


def test_an_unchanged_level_is_left_alone(monkeypatch):
    parent = logging.getLogger(_name())
    parent.getChild("camera").setLevel(logging.INFO)
    cleared = []
    monkeypatch.setattr(
        logging.Logger.manager, "_clear_cache", lambda: cleared.append(True)
    )

    _child_with_level(parent, "camera", logging.INFO)

    assert cleared == []


class _Printer(PrinterClient[PrinterConfig]):
    def __init__(self, camera: str) -> None:
        config = PrinterConfig.get_new()
        config.id = 1
        super().__init__(config, context=ClientContext())
        self._camera = camera

    def resolve_camera_uri(self):
        return URL(self._camera)


def test_no_camera_pool_is_not_a_failure_per_report(caplog):
    """Without a camera pool (cameras off for the app) a printer's camera is
    refused by design: no warning every time its address is projected."""
    printer = _Printer("http://camera.local/stream")
    with caplog.at_level(logging.DEBUG):
        printer.update_camera_uri()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert printer.printer.webcam_info.connected is False
