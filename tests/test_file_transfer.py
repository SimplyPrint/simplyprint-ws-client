"""Tests for the FileTransfer prepare lifecycle."""

import asyncio
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional
from unittest.mock import patch

import pytest

import simplyprint_ws_client.const as ws_const
from simplyprint_ws_client import Client, FileDemandData, FileProgressStateEnum
from simplyprint_ws_client.integration.transfer import (
    FileTransfer,
    FirmwareStartOutcome,
)
from simplyprint_ws_client.integration.transfer import file_transfer as ft_mod

from tests._loop_heartbeat import LoopHeartbeat


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


# -- async hooks: locate skips download; offload runs off the loop ----------- #


@pytest.mark.asyncio
async def test_locate_existing_hit_skips_the_download(client: Client):
    located = (PurePosixPath("/remote/job.gcode"), "md5existing")
    downloads = {"n": 0}

    class LocatingTransfer(_StubFileTransfer):
        async def _locate_existing(self, data):
            return located

        async def download_file_and_upload(self, data):
            downloads["n"] += 1
            return PurePosixPath("never"), "never"

    transfer = LocatingTransfer(client)
    result = await transfer.ensure_file(FileDemandData(file_name="job.gcode"))

    assert result == located
    assert downloads["n"] == 0  # the located file short-circuits the download


@pytest.mark.asyncio
async def test_offload_falls_back_to_a_thread_without_an_app(client: Client):
    assert client.offload is None  # a client built outside an app has no lanes
    transfer = _StubFileTransfer(client=client)

    ran_on = await transfer._offload(threading.get_ident)

    assert ran_on != threading.get_ident()  # ran on a worker thread, value returned


@pytest.mark.asyncio
async def test_slow_prepare_does_not_stall_the_loop(client: Client, monkeypatch):
    async def fake_download_as_file(self, data, dest, clamp_progress):
        dest.write_bytes(b"gcode")
        clamp_progress(100)
        return dest

    monkeypatch.setattr(ft_mod.FileDownload, "download_as_file", fake_download_as_file)

    class SlowPrepareTransfer(_StubFileTransfer):
        async def _prepare_local_file(self, data, local_path):
            # A blocking transform offloaded off the loop (no app => a thread).
            return await self._offload(lambda: (time.sleep(2.0), local_path)[1])

        async def _upload(self, local_path, on_progress):
            return PurePosixPath("/") / local_path.name  # absolute (relative_to "/")

    transfer = SlowPrepareTransfer(client)
    transfer.begin_prepare()

    async with LoopHeartbeat(interval=0.01) as hb:
        task = asyncio.create_task(
            transfer.download_file_and_upload(FileDemandData(file_name="job.gcode"))
        )
        await asyncio.sleep(0.3)  # the 2s prepare is running off the loop
        assert not task.done()
        result = await task

    assert hb.max_gap_ms < 200  # the loop kept beating through the slow prepare
    assert result is not None
    transfer.end_prepare("done")


@pytest.mark.asyncio
async def test_download_file_and_upload_uses_app_cache_tempdir(
    client: Client, monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        type(ws_const.APP_DIRS),
        "user_cache_path",
        property(lambda self: cache),
    )

    async def fake_download_as_file(self, data, dest, clamp_progress):
        dest.write_bytes(b"gcode")
        clamp_progress(100)
        return dest

    monkeypatch.setattr(ft_mod.FileDownload, "download_as_file", fake_download_as_file)
    seen = {}

    class InspectingTransfer(_StubFileTransfer):
        async def _upload(self, local_path, on_progress):
            seen["temp_dir"] = local_path.parent
            on_progress(100)
            return PurePosixPath("/") / local_path.name

    transfer = InspectingTransfer(client)
    transfer.begin_prepare()

    result = await transfer.download_file_and_upload(
        FileDemandData(file_name="job.gcode")
    )

    assert result is not None
    assert seen["temp_dir"].parent == cache / "transfers"
    assert not seen["temp_dir"].exists()
    transfer.end_prepare("test done")


def test_connect_change_mid_prepare_keeps_download_pending(client: Client):
    transfer = _StubFileTransfer(client=client)
    transfer.begin_prepare()
    client.printer.file_progress.percent = 73
    client.printer.file_progress.message = "old"

    transfer._on_client_connect_change("disconnected")

    assert transfer.is_preparing_to_print is True
    assert client.printer.file_progress.state == FileProgressStateEnum.DOWNLOADING
    assert client.printer.file_progress.percent == 0
    assert client.printer.file_progress.message is None


def test_no_progress_timeout_lands_terminal_error(client: Client):
    transfer = _StubFileTransfer(client=client)
    with patch.object(ft_mod.time, "monotonic", return_value=1000.0):
        transfer.begin_prepare()

    with patch.object(ft_mod.time, "monotonic", return_value=1601.0):
        assert transfer._fail_if_prepare_stalled() is True

    assert transfer.is_preparing_to_print is False
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR
    assert "600s" in client.printer.file_progress.message


@pytest.mark.asyncio
async def test_upload_retries_keep_download_state(client: Client, monkeypatch):
    async def fake_download_as_file(self, data, dest, clamp_progress):
        dest.write_bytes(b"gcode")
        clamp_progress(100)
        return dest

    monkeypatch.setattr(ft_mod.FileDownload, "download_as_file", fake_download_as_file)

    class FlakyUploadTransfer(_StubFileTransfer):
        retry_backoff_seconds = 0

        def __init__(self, client):
            super().__init__(client)
            self.attempts = 0

        async def _upload(self, local_path, on_progress):
            self.attempts += 1
            if self.attempts < 3:
                raise OSError("temporary offline")
            on_progress(100)
            return PurePosixPath("/") / local_path.name

    transfer = FlakyUploadTransfer(client)
    transfer.begin_prepare()

    result = await transfer.download_file_and_upload(
        FileDemandData(file_name="job.gcode")
    )

    assert result is not None
    assert result[0] == PurePosixPath("job.gcode")
    assert transfer.attempts == 3
    assert client.printer.file_progress.state == FileProgressStateEnum.DOWNLOADING
    assert client.printer.file_progress.percent == 100
    transfer.end_prepare("test done")
