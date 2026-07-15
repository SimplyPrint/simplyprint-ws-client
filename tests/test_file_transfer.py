"""Behavioral contract for the composed print-file controller."""

import asyncio
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock

import pytest

from simplyprint_ws_client import (
    Client,
    ClientContext,
    FileDemandData,
    FileProgressStateEnum,
)
from simplyprint_ws_client.common.utils.slugify import slugify
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.integration.client import PrinterClient
from simplyprint_ws_client.integration.transfer import (
    FileOperationError,
    FileTransfer,
    PreparationKind,
    PrintFileDriver,
    RetryableFileError,
    StartDisposition,
    UploadedFile,
)


async def _download(data, destination: Path, progress):
    destination.write_bytes(b"G28\n")
    progress(100)
    return destination


class Driver(PrintFileDriver):
    retry_backoff_seconds = 0

    def __init__(
        self,
        *,
        route=PreparationKind.UPLOAD,
        disposition=StartDisposition.COMPLETE,
    ):
        self.selected_route = route
        self.disposition = disposition
        self.uploads = 0
        self.starts = 0
        self.urls = 0
        self.atomic = 0
        self.closed = 0
        self.requests = []
        self.upload_errors = []

    def route(self, data):
        return self.selected_route

    async def upload(self, data, source, progress):
        self.uploads += 1
        self.requests.append(data)
        if self.upload_errors:
            raise self.upload_errors.pop(0)
        progress(100)
        return UploadedFile(PurePosixPath("/") / source.name)

    async def start_uploaded(self, data, uploaded):
        self.starts += 1
        self.requests.append(data)
        return self.disposition

    async def upload_and_start(self, data, source, progress):
        self.atomic += 1
        self.requests.append(data)
        progress(100)
        return self.disposition

    async def start_url(self, data, progress):
        self.urls += 1
        self.requests.append(data)
        return self.disposition

    async def close(self):
        self.closed += 1


def _transfer(
    client: Client,
    driver: PrintFileDriver,
    downloader=_download,
    report_job_error=None,
):
    return FileTransfer(
        driver,
        client.printer.file_progress,
        client.logger,
        report_job_error,
        cache_root=Path("/tmp"),
        downloader=downloader,
    )


def test_file_operation_error_keeps_user_and_diagnostic_messages():
    error = FileOperationError("visible", "diagnostic")
    assert error.user_message == "visible"
    assert error.reason == "diagnostic"


def test_print_identity_tracks_accepted_start_and_same_file_reprint(client: Client):
    transfer = _transfer(client, Driver())
    transfer.record_start_sent("model.gcode")
    assert transfer.observe_job_start("firmware.gcode") == ("model.gcode", False)
    assert transfer.observe_job_start("/local/model.gcode") == (
        "/local/model.gcode",
        True,
    )


@pytest.mark.asyncio
async def test_request_is_normalized_by_copy_not_mutated(client: Client):
    driver = Driver()
    transfer = _transfer(client, driver)
    request = FileDemandData(file_name="My awkward file.gcode", auto_start=False)

    assert transfer.submit(request)
    await transfer.wait()

    assert request.file_name == "My awkward file.gcode"
    assert driver.requests[0].file_name == slugify(request.file_name)
    assert client.printer.file_progress.state == FileProgressStateEnum.READY


@pytest.mark.asyncio
async def test_upload_can_stage_then_start(client: Client):
    driver = Driver()
    transfer = _transfer(client, driver)
    assert transfer.submit(
        FileDemandData(job_id=1, file_name="job.gcode", auto_start=False)
    )
    await transfer.wait()

    assert driver.uploads == 1
    assert driver.starts == 0
    assert transfer.start_staged()
    await transfer.wait()

    assert driver.starts == 1
    assert not transfer.start_staged()
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR


@pytest.mark.asyncio
async def test_failed_staged_start_retains_file_for_retry(client: Client):
    class FailsOnce(Driver):
        async def start_uploaded(self, data, uploaded):
            self.starts += 1
            if self.starts == 1:
                raise FileOperationError("start failed")
            return StartDisposition.COMPLETE

    driver = FailsOnce()
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=False))
    await transfer.wait()

    assert transfer.start_staged()
    await transfer.wait()
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR
    assert transfer.start_staged()
    await transfer.wait()
    assert driver.starts == 2
    assert client.printer.file_progress.state == FileProgressStateEnum.READY


@pytest.mark.asyncio
async def test_new_file_admission_invalidates_an_older_staged_upload(client: Client):
    driver = Driver()
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(job_id=1, file_name="old.gcode", auto_start=False))
    await transfer.wait()

    driver.upload_errors = [FileOperationError("new upload failed")]
    assert transfer.submit(
        FileDemandData(job_id=2, file_name="new.gcode", auto_start=False)
    )
    await transfer.wait()

    assert not transfer.start_staged()
    assert client.printer.file_progress.message == "No file is ready to start"


@pytest.mark.asyncio
async def test_url_route_skips_connector_download_and_waits_for_device(client: Client):
    downloads = 0

    async def forbidden(*args):
        nonlocal downloads
        downloads += 1
        raise AssertionError("URL route downloaded locally")

    driver = Driver(
        route=PreparationKind.URL,
        disposition=StartDisposition.AWAIT_DEVICE,
    )
    transfer = _transfer(client, driver, forbidden)
    transfer.submit(
        FileDemandData(
            file_name="job.3mf",
            cdn_url="https://cdn.test/job.3mf",
            auto_start=True,
        )
    )
    await transfer.wait()

    assert downloads == 0
    assert driver.urls == 1
    assert transfer.awaiting_device_start
    assert transfer.progress(63)
    assert client.printer.file_progress.percent == 63
    assert transfer.started()
    assert client.printer.file_progress.state == FileProgressStateEnum.READY


@pytest.mark.asyncio
async def test_url_route_cannot_be_staged(client: Client):
    driver = Driver(route=PreparationKind.URL)
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.3mf", auto_start=False))
    await transfer.wait()
    assert driver.urls == 0
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR


@pytest.mark.asyncio
async def test_atomic_route_rejects_stage_before_download(client: Client):
    downloaded = False

    async def downloader(*args):
        nonlocal downloaded
        downloaded = True

    driver = Driver(route=PreparationKind.UPLOAD_AND_START)
    transfer = _transfer(client, driver, downloader)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=False))
    await transfer.wait()
    assert not downloaded
    assert driver.atomic == 0
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR


@pytest.mark.asyncio
async def test_atomic_route_completes_on_accepted_request(client: Client):
    driver = Driver(route=PreparationKind.UPLOAD_AND_START)
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=True))
    await transfer.wait()
    assert driver.atomic == 1
    assert client.printer.file_progress.state == FileProgressStateEnum.READY


@pytest.mark.asyncio
async def test_only_retryable_uploads_use_the_single_attempt_budget(client: Client):
    driver = Driver()
    driver.max_upload_attempts = 3
    driver.upload_errors = [
        RetryableFileError("temporary one"),
        RetryableFileError("temporary two"),
    ]
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=False))
    await transfer.wait()
    assert driver.uploads == 3
    assert client.printer.file_progress.state == FileProgressStateEnum.READY


@pytest.mark.asyncio
async def test_permanent_upload_error_is_not_retried(client: Client):
    driver = Driver()
    driver.upload_errors = [FileOperationError("permanent")]
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=False))
    await transfer.wait()
    assert driver.uploads == 1
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR


@pytest.mark.asyncio
async def test_device_start_can_win_response_race(client: Client):
    class RacingDriver(Driver):
        def __init__(self):
            super().__init__(disposition=StartDisposition.AWAIT_DEVICE)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def start_uploaded(self, data, uploaded):
            self.entered.set()
            await self.release.wait()
            return self.disposition

    driver = RacingDriver()
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=True))
    await driver.entered.wait()

    assert transfer.started()
    driver.release.set()
    await transfer.wait()
    assert client.printer.file_progress.state == FileProgressStateEnum.READY


@pytest.mark.asyncio
async def test_device_rejection_is_terminal(client: Client):
    driver = Driver(disposition=StartDisposition.AWAIT_DEVICE)
    transfer = _transfer(client, driver)
    transfer.submit(FileDemandData(file_name="job.gcode", auto_start=True))
    await transfer.wait()
    assert transfer.rejected("bad file")
    assert client.printer.file_progress.state == FileProgressStateEnum.ERROR
    assert client.printer.file_progress.message == "bad file"


@pytest.mark.asyncio
async def test_different_job_preempts_and_reports_previous_job(client: Client):
    class BlockingDriver(Driver):
        def __init__(self):
            super().__init__()
            self.first = asyncio.Event()
            self.release = asyncio.Event()

        async def upload(self, data, source, progress):
            if data.job_id == 1:
                self.first.set()
                await self.release.wait()
            return await super().upload(data, source, progress)

    report_job_error = AsyncMock()
    driver = BlockingDriver()
    transfer = _transfer(client, driver, report_job_error=report_job_error)
    transfer.submit(FileDemandData(job_id=1, file_name="one.gcode"))
    await driver.first.wait()
    transfer.submit(FileDemandData(job_id=2, file_name="two.gcode"))
    await transfer.wait()

    report_job_error.assert_awaited_once_with(
        1, "File preparation was replaced by a newer job"
    )


@pytest.mark.asyncio
async def test_duplicate_active_job_is_idempotent(client: Client):
    class BlockingDriver(Driver):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def upload(self, data, source, progress):
            self.entered.set()
            await self.release.wait()
            return await super().upload(data, source, progress)

    driver = BlockingDriver()
    transfer = _transfer(client, driver)
    assert transfer.submit(FileDemandData(job_id=1, file_name="one.gcode"))
    await driver.entered.wait()
    assert not transfer.submit(FileDemandData(job_id=1, file_name="again.gcode"))
    driver.release.set()
    await transfer.wait()
    assert driver.uploads == 1


@pytest.mark.asyncio
async def test_close_cancels_work_and_closes_driver(client: Client):
    driver = Driver()
    transfer = _transfer(client, driver)
    await transfer.close()
    assert driver.closed == 1


@pytest.mark.asyncio
async def test_printer_client_builds_the_superseded_job_message():
    client = PrinterClient(PrinterConfig.get_new(), context=ClientContext())
    client.send = AsyncMock()

    await client.report_job_error(7, "superseded")

    message = client.send.await_args.args[0]
    assert message.data == {
        "state": FileProgressStateEnum.ERROR,
        "job_id": 7,
        "message": "superseded",
    }
    assert client.send.await_args.kwargs == {"skip_dispatch": True}
