"""Tests for the FileTransfer prepare lifecycle."""

import asyncio
import threading
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import pytest

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


class _CompletingTransfer(_StubFileTransfer):
    def __init__(self, client: Client, done: threading.Event):
        super().__init__(client)
        self.done = done
        self.loop = None

    async def ensure_file(self, data: FileDemandData):
        self.loop = asyncio.get_running_loop()
        self.done.set()
        return PurePosixPath("/ready.gcode"), "md5"


def test_ensure_file_and_start_task_marshals_to_client_loop(client: Client):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="client-loop")
    client.event_loop = loop
    done = threading.Event()
    transfer = _CompletingTransfer(client, done)

    thread.start()
    try:
        transfer.ensure_file_and_start_task(
            FileDemandData(job_id=1, file_name="ready.gcode", auto_start=False)
        )

        assert done.wait(timeout=3.0)
        transfer._download_dispatch.result(timeout=3.0)
        assert transfer.loop is loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=3.0)
        loop.close()


class _BlockingTransfer(_StubFileTransfer):
    def __init__(self, client: Client):
        super().__init__(client)
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()
        self._never = asyncio.Event()

    async def ensure_file(self, data: FileDemandData):
        if data.job_id == 1:
            self.first_started.set()
            await self._never.wait()
        self.second_started.set()
        return PurePosixPath("/ready.gcode"), "md5"


@pytest.mark.asyncio
async def test_new_transfer_cancels_stuck_previous_transfer(client: Client):
    client.event_loop = asyncio.get_running_loop()
    transfer = _BlockingTransfer(client)

    first = asyncio.create_task(
        transfer.ensure_file_and_start(
            FileDemandData(job_id=1, file_name="first.gcode", auto_start=False)
        )
    )
    await asyncio.wait_for(transfer.first_started.wait(), timeout=1.0)

    second = asyncio.create_task(
        transfer.ensure_file_and_start(
            FileDemandData(job_id=2, file_name="second.gcode", auto_start=False)
        )
    )

    await asyncio.wait_for(transfer.second_started.wait(), timeout=1.0)
    await second

    assert first.cancelled()
    assert client.printer.current_job_id == 2
    assert transfer.is_preparing_to_print is False
