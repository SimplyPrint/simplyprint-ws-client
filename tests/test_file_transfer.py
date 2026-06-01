"""Tests for the FileTransfer prepare lifecycle."""

from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from simplyprint_ws_client import Client, FileDemandData
from simplyprint_ws_client.contrib.transfer import (
    FileTransfer,
    FirmwareStartOutcome,
)


class _StubFileTransfer(FileTransfer):
    """Minimal concrete FileTransfer that fills the abstract hooks."""

    async def _upload(
        self, local_path: Path, on_progress: Callable[[float], None]
    ) -> PurePosixPath:
        return PurePosixPath(local_path.name)

    async def _send_start(
        self,
        path: PurePosixPath,
        data: FileDemandData,
        md5checksum: Optional[str],
    ) -> None:
        pass

    def _firmware_outcome(self, changes) -> FirmwareStartOutcome:
        return FirmwareStartOutcome.PENDING


def test_set_active_job_for_prepare_sets_job_state(client: Client):
    """_set_active_job_for_prepare sets job_id, action_token, clears bed flag."""
    printer = client.printer
    printer.have_cleared_bed = True  # pre-set to non-default

    transfer = _StubFileTransfer(client=client)
    transfer._set_active_job_for_prepare(job_id=42, action_token="test_token")

    assert printer.current_job_id == 42
    assert printer.file_action_token == "test_token"
    assert printer.have_cleared_bed is False


def test_set_active_job_for_prepare_accepts_none(client: Client):
    """_set_active_job_for_prepare accepts a cleared job (None) and still resets bed."""
    printer = client.printer
    printer.have_cleared_bed = True

    transfer = _StubFileTransfer(client=client)
    transfer._set_active_job_for_prepare(job_id=None, action_token=None)

    assert printer.current_job_id is None
    assert printer.file_action_token is None
    assert printer.have_cleared_bed is False
