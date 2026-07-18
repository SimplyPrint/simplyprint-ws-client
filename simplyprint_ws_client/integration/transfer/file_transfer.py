"""One print-preparation lifecycle composed with one printer file driver.

``FileTransfer`` owns admission, cancellation, source download, upload retry,
staging, firmware-confirmation timeouts and teardown. Integrations implement
``PrintFileDriver`` and explicitly project device reports back through
``started()``, ``rejected()`` and ``progress()``. Neither side reaches through
the other with lifecycle hooks.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable, final, Optional, Tuple

from simplyprint_ws_client.common.utils.slugify import slugify
from simplyprint_ws_client.common.utils.temp import (
    app_cache_path,
    cache_temporary_directory,
)
from simplyprint_ws_client.core.protocol.messages import (
    FileDemandData,
)
from simplyprint_ws_client.core.state import FileProgressState, FileProgressStateEnum
from simplyprint_ws_client.integration.transfer.download import (
    FileDownloadError,
    download_file,
)

ProgressCallback = Callable[[float], None]
JobErrorReporter = Callable[[int, str], Awaitable[None]]

DEFAULT_GRACE_SECONDS = 60.0
DEFAULT_MAX_UPLOAD_ATTEMPTS = 4
DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS = 10 * 60.0
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_WATCHDOG_INTERVAL_SECONDS = 5.0


class PreparationKind(Enum):
    """The physical route a FILE demand takes to the printer."""

    UPLOAD = "upload"
    UPLOAD_AND_START = "upload_and_start"
    URL = "url"


class StartDisposition(Enum):
    """How an accepted start operation reaches its terminal state."""

    COMPLETE = "complete"
    AWAIT_DEVICE = "await_device"
    # The device still has to fetch the file. Its start grace begins at 100 and
    # is disarmed if the device reports that transfer restarted below 100.
    AWAIT_DEVICE_TRANSFER = "await_device_transfer"


@dataclass(frozen=True)
class UploadedFile:
    """A driver-owned file now addressable by a printer start command."""

    path: PurePosixPath
    md5: Optional[str] = None


class FileOperationError(Exception):
    """A permanent, user-reportable print-file operation failure."""

    def __init__(self, user_message: str, reason: Optional[str] = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.reason = reason or user_message


class RetryableFileError(FileOperationError):
    """An upload failed before acceptance and may safely be attempted again."""


class UnsupportedFileOperation(FileOperationError):
    """The selected printer route is not implemented by this driver."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"This printer does not support {operation}",
            f"unsupported print-file operation: {operation}",
        )


class PrintFileDriver:
    """Printer-specific print-file operations, without lifecycle ownership.

    A driver method either returns after the device transport accepted the
    operation or raises. Device-side completion is expressed by the returned
    :class:`StartDisposition`; later reports are projected explicitly into the
    composed :class:`FileTransfer` by the owning printer.
    """

    max_upload_attempts: int = DEFAULT_MAX_UPLOAD_ATTEMPTS
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS

    def route(self, _data: FileDemandData) -> PreparationKind:
        return PreparationKind.UPLOAD

    async def upload(
        self,
        data: FileDemandData,
        source: Path,
        progress: ProgressCallback,
    ) -> UploadedFile:
        raise UnsupportedFileOperation("local file upload")

    async def start_uploaded(
        self, data: FileDemandData, uploaded: UploadedFile
    ) -> StartDisposition:
        raise UnsupportedFileOperation("starting a stored file")

    async def upload_and_start(
        self,
        data: FileDemandData,
        source: Path,
        progress: ProgressCallback,
    ) -> StartDisposition:
        raise UnsupportedFileOperation("atomic upload and start")

    async def start_url(
        self, data: FileDemandData, progress: ProgressCallback
    ) -> StartDisposition:
        raise UnsupportedFileOperation("starting a file by URL")

    async def cancel_start(self) -> None:
        """Best-effort cancellation after a device-confirmation timeout."""

    async def close(self) -> None:
        """Close resources owned by the driver."""


@dataclass(frozen=True)
class _StagedPrint:
    uploaded: UploadedFile
    data: FileDemandData


@dataclass
class _TransferLifecycle:
    route: PreparationKind
    data: FileDemandData
    staged_start: bool = False
    task: Optional[asyncio.Task] = field(default=None, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    terminal: Optional[FileProgressStateEnum] = None
    activity_at: Optional[float] = None
    awaiting_since: Optional[float] = None
    start_in_flight: bool = False
    start_disposition: Optional[StartDisposition] = None
    watchdog: Optional[asyncio.Task] = field(default=None, repr=False)

    @property
    def job_id(self) -> Optional[int]:
        return self.data.job_id

    @property
    def is_preparing(self) -> bool:
        return self.activity_at is not None


_executing_lifecycle: contextvars.ContextVar[Optional[_TransferLifecycle]] = (
    contextvars.ContextVar("file_transfer_lifecycle", default=None)
)


@final
class FileTransfer:
    """Final controller for one printer's FILE and staged START operations."""

    grace_seconds: float = DEFAULT_GRACE_SECONDS
    no_progress_timeout_seconds: float = DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    watchdog_interval_seconds: float = DEFAULT_WATCHDOG_INTERVAL_SECONDS

    def __init__(
        self,
        driver: PrintFileDriver,
        progress: FileProgressState,
        logger: logging.Logger,
        report_job_error: Optional[JobErrorReporter] = None,
        *,
        cache_root: Optional[Path] = None,
        downloader: Callable[
            [FileDemandData, Path, ProgressCallback], Awaitable[Path]
        ] = download_file,
    ) -> None:
        self.driver = driver
        self._progress_state = progress
        self._logger = logger
        self._report_job_error_callback = report_job_error
        self._cache_root = cache_root
        self._downloader = downloader
        self._staged: Optional[_StagedPrint] = None
        self._sent_start_filename: Optional[str] = None
        self._previous_print_filename: Optional[str] = None
        self._lifecycle: Optional[_TransferLifecycle] = None

    @property
    def is_preparing_to_print(self) -> bool:
        lifecycle = self._lifecycle
        return lifecycle is not None and lifecycle.is_preparing

    @property
    def awaiting_device_start(self) -> bool:
        lifecycle = self._lifecycle
        return lifecycle is not None and lifecycle.awaiting_since is not None

    @property
    def cancelled(self) -> bool:
        lifecycle = _executing_lifecycle.get()
        return lifecycle is not None and (
            lifecycle.cancelled.is_set() or self._lifecycle is not lifecycle
        )

    def record_start_sent(self, filename: str) -> None:
        """Remember the accepted start identity for the next device job edge."""
        self._sent_start_filename = filename

    def observe_job_start(
        self,
        device_filename: Optional[str],
        *,
        reprint_if_filename_missing: bool = False,
    ) -> Tuple[Optional[str], bool]:
        """Resolve a device job edge against the last accepted start command."""
        sent = self._sent_start_filename
        filename = sent or device_filename
        previous = self._previous_print_filename
        reprint = (
            sent is None
            and previous is not None
            and (
                (not device_filename and reprint_if_filename_missing)
                or (
                    bool(device_filename)
                    and PurePosixPath(device_filename).name
                    == PurePosixPath(previous).name
                )
            )
        )
        self._previous_print_filename = sent
        self._sent_start_filename = None
        return filename, reprint

    def submit(self, data: FileDemandData) -> bool:
        """Accept and schedule a FILE demand on the current event loop."""
        try:
            normalized = self._normalize_request(data)
            route = self.driver.route(normalized)
        except FileOperationError as error:
            self._report_start_error(error.user_message)
            return False
        except Exception as error:
            self._report_start_error(f"Failed to prepare file: {error}")
            return False
        return self._submit(normalized, route, self._prepare)

    def start_staged(self) -> bool:
        """Accept START for the most recently staged uploaded file."""
        if self._lifecycle is not None:
            return False
        staged = self._staged
        if staged is None:
            self._report_start_error("No file is ready to start")
            return False
        return self._submit(
            staged.data,
            PreparationKind.UPLOAD,
            self._start_prepared,
            staged_start=True,
        )

    def _submit(
        self,
        data: FileDemandData,
        route: PreparationKind,
        runner: Callable[[FileDemandData], Awaitable[None]],
        *,
        staged_start: bool = False,
    ) -> bool:
        loop = asyncio.get_running_loop()
        previous = self._lifecycle
        if (
            data.job_id is not None
            and previous is not None
            and previous.job_id == data.job_id
        ):
            self._logger.info("Ignoring duplicate file demand for job %s", data.job_id)
            return False

        # A newly accepted FILE supersedes both in-flight work and any older
        # staged upload.  Invalidate synchronously at admission so a failure
        # in the new operation can never expose the stale file to START.
        if not staged_start:
            self._staged = None

        if previous is not None:
            previous.cancelled.set()
            if previous.task is not None and not previous.task.done():
                previous.task.cancel()
            if previous.watchdog is not None and not previous.watchdog.done():
                previous.watchdog.cancel()

        lifecycle = _TransferLifecycle(
            route=route,
            data=data,
            staged_start=staged_start,
        )
        self._lifecycle = lifecycle
        self._progress_state.message = None
        self._progress_state.percent = 0
        self._progress_state.state = FileProgressStateEnum.DOWNLOADING
        lifecycle.task = loop.create_task(
            self._run_lifecycle(lifecycle, previous, runner)
        )
        lifecycle.task.add_done_callback(
            lambda task, current=lifecycle: self._on_lifecycle_done(current, task)
        )
        return True

    async def _run_lifecycle(
        self,
        lifecycle: _TransferLifecycle,
        previous: Optional[_TransferLifecycle],
        runner: Callable[[FileDemandData], Awaitable[None]],
    ) -> None:
        token = _executing_lifecycle.set(lifecycle)
        data = lifecycle.data
        try:
            if previous is not None and previous.task is not None:
                await asyncio.gather(previous.task, return_exceptions=True)
            if self._lifecycle is not lifecycle:
                return

            if previous is not None:
                self._end_lifecycle(previous, "superseded")
                if (
                    previous.terminal is None
                    and previous.job_id is not None
                    and previous.job_id != lifecycle.job_id
                ):
                    await self._report_job_error(
                        previous.job_id,
                        "File preparation was replaced by a newer job",
                    )
                    previous.terminal = FileProgressStateEnum.ERROR
            if self._lifecycle is not lifecycle:
                return

            self._begin_lifecycle(lifecycle)
            await runner(data)
        except asyncio.CancelledError:
            if self._lifecycle is lifecycle and not lifecycle.cancelled.is_set():
                self._fail("File transfer was cancelled", "task cancelled")
            raise
        except FileOperationError as error:
            if self._lifecycle is lifecycle:
                self._logger.warning(
                    'Preparing file "%s" failed', data.file_name, exc_info=error
                )
                self._fail(error.user_message, error.reason)
        except Exception as error:
            if self._lifecycle is lifecycle:
                self._logger.warning(
                    'Preparing file "%s" failed', data.file_name, exc_info=error
                )
                action = "start print" if lifecycle.staged_start else "prepare file"
                self._fail(f"Failed to {action}: {error}", f"exception: {error}")
        finally:
            _executing_lifecycle.reset(token)
            if self._lifecycle is lifecycle and not lifecycle.is_preparing:
                self._lifecycle = None

    def _on_lifecycle_done(
        self, lifecycle: _TransferLifecycle, task: asyncio.Task
    ) -> None:
        if self._lifecycle is lifecycle and not lifecycle.is_preparing:
            self._lifecycle = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._logger.error("File operation failed", exc_info=error)

    async def wait(self) -> None:
        """Wait until the active operation coroutine returns."""
        lifecycle = self._lifecycle
        if lifecycle is not None and lifecycle.task is not None:
            await asyncio.gather(lifecycle.task, return_exceptions=True)

    async def close(self) -> None:
        """Cancel owned work, then close the composed driver."""
        lifecycle, self._lifecycle = self._lifecycle, None
        if lifecycle is not None:
            lifecycle.cancelled.set()
        tasks = [
            task
            for task in (
                lifecycle.task if lifecycle is not None else None,
                lifecycle.watchdog if lifecycle is not None else None,
            )
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if lifecycle is not None:
            self._end_lifecycle(lifecycle, "closed")
        self._staged = None
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.driver.close()

    async def _prepare(self, data: FileDemandData) -> None:
        lifecycle = self._executing()
        route = lifecycle.route

        if route is PreparationKind.URL:
            if not data.auto_start:
                raise FileOperationError(
                    "This printer cannot store a URL file without starting it",
                    "URL route is start-only",
                )
            await self._dispatch_start(
                data.file_name or "print-file",
                self.driver.start_url(data, self._device_progress),
            )
            return

        if route is PreparationKind.UPLOAD_AND_START and not data.auto_start:
            raise FileOperationError(
                "This printer cannot store a file without starting it",
                "device upload endpoint starts the print",
            )

        async with self._source_file(data) as source:
            if route is PreparationKind.UPLOAD_AND_START:
                await self._dispatch_start(
                    data.file_name or source.name,
                    self.driver.upload_and_start(data, source, self._upload_progress),
                )
                return
            uploaded = await self._upload_with_retry(data, source)
        if not data.auto_start:
            self._staged = _StagedPrint(uploaded=uploaded, data=data)
            self._mark_ready("file ready (no auto_start)")
            return

        await self._dispatch_start(
            uploaded.path.name,
            self.driver.start_uploaded(data, uploaded),
        )

    async def _start_prepared(self, data: FileDemandData) -> None:
        staged = self._staged
        if staged is None or staged.data is not data:
            raise FileOperationError("No file is ready to start", "staged file missing")
        await self._dispatch_start(
            staged.uploaded.path.name,
            self.driver.start_uploaded(data, staged.uploaded),
        )
        # Clear only after the driver accepted the START operation.
        self._staged = None

    async def _dispatch_start(
        self, filename: str, operation: Awaitable[StartDisposition]
    ) -> None:
        lifecycle = self._executing()
        lifecycle.start_in_flight = True
        self.record_start_sent(filename)
        try:
            disposition = await operation
        except BaseException:
            if self._sent_start_filename == filename:
                self._sent_start_filename = None
            lifecycle.start_in_flight = False
            raise
        lifecycle.start_in_flight = False

        # A device edge may have completed this operation while its command
        # coroutine was still returning (Moonraker's response/event race).
        if self._lifecycle is not lifecycle or not lifecycle.is_preparing:
            return
        if disposition is StartDisposition.COMPLETE:
            self._mark_ready("device accepted start")
            return
        if disposition is StartDisposition.AWAIT_DEVICE:
            lifecycle.start_disposition = disposition
            lifecycle.awaiting_since = lifecycle.activity_at = time.monotonic()
            self._progress_state.percent = 100
            return
        if disposition is StartDisposition.AWAIT_DEVICE_TRANSFER:
            lifecycle.start_disposition = disposition
            if self._progress_state.percent >= 100:
                lifecycle.awaiting_since = lifecycle.activity_at
            return
        raise FileOperationError(
            "Printer returned an invalid start result",
            f"invalid start disposition: {disposition!r}",
        )

    @asynccontextmanager
    async def _source_file(self, data: FileDemandData):
        root = self._cache_root or app_cache_path("transfers")
        with cache_temporary_directory("sp-transfer-", root=root) as local_folder:
            destination = Path(local_folder) / (data.file_name or "print-file")
            try:
                await self._downloader(data, destination, self._download_progress)
            except _TransferCancelled:
                raise asyncio.CancelledError
            except FileDownloadError as error:
                raise FileOperationError(
                    str(error), f"download failed: {error}"
                ) from error
            except Exception as error:
                if self.cancelled:
                    raise asyncio.CancelledError
                raise FileOperationError(
                    f"Failed to download file: {error}", f"download failed: {error}"
                ) from error
            self._progress_state.percent = 50
            self._touch_activity()
            yield destination

    async def _upload_with_retry(
        self, data: FileDemandData, source: Path
    ) -> UploadedFile:
        attempts = max(1, int(self.driver.max_upload_attempts))
        for attempt in range(1, attempts + 1):
            try:
                uploaded = await self.driver.upload(data, source, self._upload_progress)
                path = uploaded.path
                if path.is_absolute():
                    path = path.relative_to("/")
                return replace(uploaded, path=path)
            except _TransferCancelled:
                raise asyncio.CancelledError
            except RetryableFileError as error:
                if self.cancelled:
                    raise asyncio.CancelledError
                if self._has_stalled():
                    raise FileOperationError(
                        "No file transfer progress for "
                        f"{int(self.no_progress_timeout_seconds)}s",
                        "transfer stalled",
                    ) from error
                if attempt >= attempts:
                    raise FileOperationError(
                        error.user_message,
                        f"upload failed after {attempt} attempts: {error.reason}",
                    ) from error
                self._logger.warning(
                    "File upload attempt %s/%s failed; retrying",
                    attempt,
                    attempts,
                    exc_info=error,
                )
                self._progress_state.message = None
                self._progress_state.percent = 50
                self._progress_state.state = FileProgressStateEnum.DOWNLOADING
                await asyncio.sleep(self.driver.retry_backoff_seconds)
        raise AssertionError("upload attempt loop exhausted")

    def started(self) -> bool:
        """Project a genuine device-start edge into the active operation."""
        lifecycle = self._lifecycle
        if (
            lifecycle is None
            or not lifecycle.is_preparing
            or not (
                lifecycle.start_in_flight
                or lifecycle.start_disposition is not None
            )
        ):
            return False
        self._touch_activity()
        self._mark_ready("device started")
        return True

    def rejected(self, message: str, reason: str = "device rejected start") -> bool:
        """Project a genuine device rejection into the active operation."""
        lifecycle = self._lifecycle
        if (
            lifecycle is None
            or not lifecycle.is_preparing
            or not (
                lifecycle.start_in_flight
                or lifecycle.start_disposition is not None
            )
        ):
            return False
        self._fail(message, reason)
        return True

    def progress(self, percent: float) -> bool:
        """Project printer-side URL preparation progress."""
        lifecycle = self._lifecycle
        if lifecycle is None or not lifecycle.is_preparing:
            return False
        self._device_progress(percent)
        if lifecycle.start_disposition is StartDisposition.AWAIT_DEVICE_TRANSFER:
            if self._progress_state.percent >= 100:
                if lifecycle.awaiting_since is None:
                    lifecycle.awaiting_since = lifecycle.activity_at
            else:
                lifecycle.awaiting_since = None
        return True

    def _normalize_request(self, data: FileDemandData) -> FileDemandData:
        if not data.file_name:
            raise FileOperationError("No file name provided", "missing file_name")
        normalized_name = slugify(data.file_name)
        if not normalized_name:
            raise FileOperationError("Invalid file name", "slugified filename is empty")
        return data.model_copy(update={"file_name": normalized_name})

    def _executing(self) -> _TransferLifecycle:
        lifecycle = _executing_lifecycle.get()
        if lifecycle is None or self._lifecycle is not lifecycle:
            raise asyncio.CancelledError
        return lifecycle

    def _begin_lifecycle(self, lifecycle: _TransferLifecycle) -> None:
        lifecycle.activity_at = time.monotonic()
        lifecycle.awaiting_since = None
        self._logger.debug("Begin print prepare (%s)", lifecycle.route.value)
        self._progress_state.message = None
        self._progress_state.state = FileProgressStateEnum.DOWNLOADING
        self._schedule_watchdog(lifecycle)

    def _end_lifecycle(self, lifecycle: _TransferLifecycle, reason: str) -> None:
        if not lifecycle.is_preparing:
            return
        self._logger.debug("End print prepare (%s)", reason)
        lifecycle.activity_at = None
        lifecycle.awaiting_since = None
        lifecycle.start_in_flight = False
        lifecycle.start_disposition = None
        watchdog, lifecycle.watchdog = lifecycle.watchdog, None
        if watchdog is not None:
            watchdog.cancel()

    def _mark_ready(self, reason: str) -> None:
        lifecycle = self._active_for_context()
        if lifecycle is None:
            return
        lifecycle.terminal = FileProgressStateEnum.READY
        self._progress_state.state = FileProgressStateEnum.READY
        self._progress_state.percent = 100
        self._end_lifecycle(lifecycle, reason)

    def _fail(self, message: str, reason: str) -> None:
        lifecycle = self._active_for_context()
        if lifecycle is None:
            return
        if lifecycle.start_in_flight or lifecycle.start_disposition is not None:
            self._sent_start_filename = None
        lifecycle.terminal = FileProgressStateEnum.ERROR
        self._progress_state.state = FileProgressStateEnum.ERROR
        self._progress_state.message = message
        self._logger.warning("Print prepare aborted: %s - %s", reason, message)
        self._end_lifecycle(lifecycle, reason)
        lifecycle.cancelled.set()
        task = lifecycle.task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _active_for_context(self) -> Optional[_TransferLifecycle]:
        executing = _executing_lifecycle.get()
        if executing is not None and self._lifecycle is not executing:
            return None
        lifecycle = self._lifecycle
        if lifecycle is None or not lifecycle.is_preparing:
            return None
        return lifecycle

    def _download_progress(self, percent: float) -> None:
        if self.cancelled:
            raise _TransferCancelled("download cancelled")
        self._touch_activity()
        self._progress_state.percent = min(max(percent, 0.0) / 2.0, 50.0)

    def _upload_progress(self, percent: float) -> None:
        if self.cancelled:
            raise _TransferCancelled("upload cancelled")
        self._touch_activity()
        self._progress_state.percent = min(max(percent, 0.0) / 2.0 + 50.0, 100.0)

    def _device_progress(self, percent: float) -> None:
        self._touch_activity()
        self._progress_state.percent = min(max(percent, 0.0), 100.0)

    def _touch_activity(self) -> None:
        lifecycle = self._lifecycle
        if lifecycle is not None and lifecycle.is_preparing:
            lifecycle.activity_at = time.monotonic()

    def _has_stalled(self) -> bool:
        lifecycle = self._lifecycle
        return bool(
            lifecycle is not None
            and lifecycle.activity_at is not None
            and time.monotonic() - lifecycle.activity_at
            > self.no_progress_timeout_seconds
        )

    def _schedule_watchdog(self, lifecycle: _TransferLifecycle) -> None:
        if lifecycle.watchdog is None or lifecycle.watchdog.done():
            lifecycle.watchdog = asyncio.get_running_loop().create_task(
                self._watchdog_loop(lifecycle)
            )

    async def _watchdog_loop(self, lifecycle: _TransferLifecycle) -> None:
        try:
            while self._lifecycle is lifecycle and lifecycle.is_preparing:
                await asyncio.sleep(self.watchdog_interval_seconds)
                if self._lifecycle is not lifecycle:
                    return
                if (
                    lifecycle.awaiting_since is not None
                    and time.monotonic() - lifecycle.awaiting_since > self.grace_seconds
                ):
                    try:
                        await self.driver.cancel_start()
                    except Exception:
                        self._logger.warning(
                            "Failed to cancel timed-out printer start", exc_info=True
                        )
                    self._fail(
                        f"Print did not start within {int(self.grace_seconds)}s",
                        "grace expired",
                    )
                    return
                if self._has_stalled():
                    self._fail(
                        "No file transfer progress for "
                        f"{int(self.no_progress_timeout_seconds)}s",
                        "transfer stalled",
                    )
                    return
        except asyncio.CancelledError:
            pass

    def _report_start_error(self, message: str) -> None:
        self._progress_state.state = FileProgressStateEnum.ERROR
        self._progress_state.message = message

    async def _report_job_error(self, job_id: int, message: str) -> None:
        if self._report_job_error_callback is None:
            return
        await self._report_job_error_callback(job_id, message)


class _TransferCancelled(Exception):
    pass


__all__ = [
    "FileOperationError",
    "FileTransfer",
    "PreparationKind",
    "PrintFileDriver",
    "RetryableFileError",
    "StartDisposition",
    "UnsupportedFileOperation",
    "UploadedFile",
]
