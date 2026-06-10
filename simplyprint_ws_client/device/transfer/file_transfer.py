"""The file-transfer + prepare-to-print lifecycle, as one business object.

Every LAN-upload integration runs the same play when a print file arrives:

1. cancel any in-flight transfer, claim the job
   (``_set_active_job_for_prepare``),
2. ``begin_prepare`` -> download from the SP CDN (0-50%),
3. transform the file for this printer, upload it (50-100%),
4. either stash it (no auto-start) or send the start command and arm a grace
   timer (``mark_transfer_complete``),
5. watch the firmware: land ``READY`` once it actually starts, ``ERROR`` if it
   rejects the start or the grace window expires.

This used to be ~200 near-identical lines in each brand's ``files.py``. It now
lives here as :class:`FileTransfer`; a brand subclass supplies only what is
genuinely brand-specific via the hooks at the bottom of the class:

* ``_upload``           -- push the local file to the printer (FTP / HTTP / ...)
* ``_send_start``       -- the brand's "start this file" command(s)
* ``_firmware_outcome`` -- read a state push and decide started / failed / pending
* and a handful of optional hooks (file transform, existing-file lookup,
  device-error text, in-firmware download progress).

The firmware-ACK gate is the subtle part and is deliberately delegated whole to
``_firmware_outcome`` because it differs by model: some firmwares discriminate a
fresh attempt by a task id; others (whose print id is always empty) key off a
genuine state transition.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from simplyprint_ws_client import FileDemandData, FileProgressStateEnum
from simplyprint_ws_client.cloud.files.file_download import FileDownload
from simplyprint_ws_client.common.utils.slugify import slugify

from simplyprint_ws_client.device.transfer.checksum import file_md5

if TYPE_CHECKING:
    from simplyprint_ws_client import Client

#: Seconds to wait, after the file reached the printer and the start command was
#: sent, for the firmware to broadcast a started print state.
DEFAULT_GRACE_SECONDS = 60.0
PREEMPTION_GRACE_SECONDS = 0.05

#: Prepared (dest, demand, md5) ready to print once a start is requested.
PreparedPrint = Tuple[PurePosixPath, FileDemandData, str]


class _TransferCancelled(Exception):
    """Cooperative cancellation raised from synchronous progress callbacks."""


class FirmwareStartOutcome(Enum):
    """What a firmware state push tells us about a print we're awaiting."""

    PENDING = auto()  #: nothing conclusive yet
    STARTED = auto()  #: the print is actually running
    FAILED = auto()  #: the printer rejected / aborted the start


class FileTransfer(ABC):
    """Owns the file-transfer + prepare lifecycle for one printer.

    A brand instantiates this with its high-level client and wires its own
    print-update / connect / disconnect events to :meth:`on_print_changed` and
    :meth:`on_connection_changed`.
    """

    #: Default transfer-type label recorded while a prepare is in flight.
    transfer_type: str = "file"
    #: Fallback message when a start is rejected with no specific device error.
    reject_message: str = "Print start was rejected by the printer"
    #: Grace window for the firmware to confirm a started print.
    grace_seconds: float = DEFAULT_GRACE_SECONDS

    def __init__(self, client: "Client") -> None:
        self.client = client
        self.next_to_print: Optional[PreparedPrint] = None

        # Prepare lifecycle. None transfer_type => no prepare in flight.
        # awaiting_since flips from None to a monotonic timestamp at
        # mark_transfer_complete (arming the grace window).
        self._prepare_transfer_type: Optional[str] = None
        self._prepare_awaiting_since: Optional[float] = None

        # The transfer runs as an asyncio.Task on the client's own event loop; a
        # new dispatch preempts the one in flight (cooperative cancel + await).
        # The canceller stays a threading.Event -- it is only ever polled via
        # is_set() at the cooperative checkpoints, never awaited.
        self._download_task_lock = asyncio.Lock()
        self._download_lock = asyncio.Lock()
        self._download_task: Optional["asyncio.Task"] = None
        self._download_canceller = threading.Event()
        # Strong ref to the scheduled task until it registers itself as
        # _download_task (asyncio only weakly references tasks).
        self._download_dispatch: Optional[object] = None

    @property
    def is_preparing_to_print(self) -> bool:
        return self._prepare_transfer_type is not None

    @property
    def current_download_type(self) -> Optional[str]:
        return self._prepare_transfer_type

    @property
    def _is_awaiting_firmware_start(self) -> bool:
        return self._prepare_awaiting_since is not None

    def begin_prepare(self, transfer_type: Optional[str] = None) -> None:
        if self.is_preparing_to_print:
            return
        transfer_type = transfer_type or self.transfer_type
        self.client.logger.debug(f"Begin print prepare ({transfer_type})")
        self._prepare_transfer_type = transfer_type
        self._prepare_awaiting_since = None
        self._on_begin_prepare()
        self.client.printer.file_progress.message = None
        self.client.printer.file_progress.state = FileProgressStateEnum.DOWNLOADING

    def mark_transfer_complete(self) -> None:
        """File reached the printer and the start command was sent; arm the
        grace timer. State stays DOWNLOADING -- READY is only set once the
        firmware actually starts, so a concurrent ``fail_prepare`` is never
        clobbered."""
        if not self.is_preparing_to_print or self._is_awaiting_firmware_start:
            return
        self._prepare_awaiting_since = time.monotonic()
        self.client.printer.file_progress.percent = 100

    def end_prepare(self, reason: str = "done") -> None:
        if not self.is_preparing_to_print:
            return
        self.client.logger.debug(f"End print prepare ({reason})")
        self._prepare_transfer_type = None
        self._prepare_awaiting_since = None
        self._on_end_prepare()

    def fail_prepare(self, message: str, reason: str) -> None:
        self.client.printer.file_progress.state = FileProgressStateEnum.ERROR
        self.client.printer.file_progress.message = message
        self.client.logger.warning(f"Print prepare aborted: {reason} - {message}")
        self.end_prepare(reason)

    def _mark_ready(self, reason: str) -> None:
        self.client.printer.file_progress.percent = 100
        self.client.printer.file_progress.state = FileProgressStateEnum.READY
        self.end_prepare(reason)

    def ensure_file_and_start_task(self, data: FileDemandData) -> None:
        """Schedule the prepare-and-start on the client's event loop, returning
        immediately. A new call preempts any transfer still in flight (see
        :meth:`ensure_file_and_start`)."""
        coro = self.ensure_file_and_start(data)
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        client_loop = getattr(self.client, "event_loop", None)
        if running_loop is not None and (
            client_loop is None or running_loop is client_loop
        ):
            self._download_dispatch = running_loop.create_task(coro)
            return

        submit_to_loop = getattr(self.client, "submit_to_loop", None)
        if submit_to_loop is not None:
            self._download_dispatch = submit_to_loop(coro)
        elif client_loop is not None:
            self._download_dispatch = asyncio.run_coroutine_threadsafe(
                coro, client_loop
            )
        else:
            coro.close()
            raise RuntimeError("No running client event loop for file transfer")

    async def ensure_file_and_start(self, data: FileDemandData) -> None:
        """Ensure the SP file is on the printer and, if auto-start is set,
        start it. A new call cancels any transfer still in flight."""
        # Take over as the active download task, signalling any in-flight
        # transfer to cancel, then await it. Because the transfer runs as a task
        # on this loop, awaiting the previous one suspends only this coroutine
        # (the previous task cooperates via the canceller and bows out) -- the
        # loop keeps running. The set -> await -> clear ordering mirrors the old
        # set -> join -> clear: set under the lock, await the previous task,
        # clear after.
        async with self._download_task_lock:
            prev = self._download_task
            if prev is not None:
                self._download_canceller.set()
            self._download_task = asyncio.current_task()

        if prev is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(prev), timeout=PREEMPTION_GRACE_SECONDS
                )
            except asyncio.TimeoutError:
                prev.cancel()
                try:
                    await prev
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass

        self._download_canceller.clear()

        try:
            async with self._download_lock:
                self._set_active_job_for_prepare(data.job_id, data.action_token)

                try:
                    self.begin_prepare()
                    result = await self.ensure_file(data)

                    if not result:
                        self.end_prepare("ensure_file returned no result")
                        return

                    if self._download_canceller.is_set():
                        self.client.logger.info("Download was cancelled")
                        self.end_prepare("download cancelled")
                        return

                    dest, md5checksum = result

                    if not data.auto_start:
                        self.next_to_print = (dest, data, md5checksum)
                        self.client.printer.file_progress.percent = 100
                        self.client.printer.file_progress.state = (
                            FileProgressStateEnum.READY
                        )
                        self.end_prepare("file ready (no auto_start)")
                        return

                    # Hand off to firmware; on_print_changed ends the prepare once
                    # the firmware reports a started (READY) or failed (ERROR) state.
                    await self.start_print(dest, data, md5checksum)
                    self.mark_transfer_complete()

                except asyncio.CancelledError:
                    if self._download_canceller.is_set():
                        self.client.logger.info("Download was cancelled")
                        self.end_prepare("download cancelled")
                    else:
                        self.fail_prepare(
                            "File transfer was cancelled", "task cancelled"
                        )
                    raise
                except Exception as e:
                    self.client.logger.warning(
                        f'Ensure file for "{data.file_name}" failed', exc_info=e
                    )
                    self.fail_prepare(
                        f"Failed to download file: {e}",
                        f"exception: {e}",
                    )
        finally:
            async with self._download_task_lock:
                if self._download_task is asyncio.current_task():
                    self._download_task = None

    async def ensure_file(
        self, data: FileDemandData
    ) -> Optional[Tuple[PurePosixPath, str]]:
        if not data.file_name:
            self.client.logger.error("No file name provided")
            return None

        # Slugify the file name (this is the one we compare with).
        data.file_name = slugify(data.file_name)

        self._pre_ensure()

        existing = self._locate_existing(data)
        if existing is not None:
            return existing

        if self._download_canceller.is_set():
            self.client.logger.info("Download was cancelled")
            return None

        return await self.download_file_and_upload(data)

    async def download_file_and_upload(
        self, data: FileDemandData
    ) -> Optional[Tuple[PurePosixPath, str]]:
        # Download is the first half of the progress bar; upload the second.
        downloader = FileDownload(self.client)

        with tempfile.TemporaryDirectory() as local_folder:
            local_dest = Path(local_folder) / data.file_name
            local_dest = await downloader.download_as_file(
                data, local_dest, lambda x: x // 2
            )

            local_dest = self._prepare_local_file(data, local_dest)

            def on_progress(progress):
                self.client.printer.file_progress.percent = min(
                    (progress // 2) + 50, 100
                )
                if self._download_canceller.is_set():
                    raise _TransferCancelled("Upload cancelled")

            try:
                dest = (await self._upload(local_dest, on_progress)).relative_to("/")
                # Hash the final (post-transform) file, off-loop, so a large
                # gcode can't stall the firmware-ACK grace window.
                return dest, await file_md5(local_dest)
            except _TransferCancelled:
                self.client.logger.info("Upload was cancelled")
                return None
            except Exception as e:
                self.client.logger.warning(
                    f'Upload error for "{data.file_name}"', exc_info=e
                )
                self.fail_prepare(self._upload_error_message(e), f"upload error: {e}")
                return None

    async def start_print(
        self,
        path: Optional[PurePosixPath] = None,
        data: Optional[FileDemandData] = None,
        md5checksum: Optional[str] = None,
    ) -> None:
        """Start a print from a previously-prepared file. With no ``path`` the
        stashed ``next_to_print`` is used (the no-auto-start path)."""
        if path is None:
            if self.next_to_print is None:
                self.client.logger.error("No path or data provided to start_print")
                return
            path, data, md5checksum = self.next_to_print
            self.next_to_print = None

        if path is None or data is None:
            self.client.logger.error("No path or data provided to start_print")
            return

        await self._send_start(path, data, md5checksum)

    def on_print_changed(self, changes) -> None:
        """Drive the prepare terminal from a firmware state push.

        Per the backend invariant, file_progress lands in exactly one terminal
        per prepare: READY (the print actually started) or ERROR (rejected, or
        the grace timer expired). The started/failed discrimination is wholly
        delegated to the brand via :meth:`_firmware_outcome`.
        """
        if not self.is_preparing_to_print:
            return

        outcome = self._firmware_outcome(changes)
        if outcome is FirmwareStartOutcome.STARTED:
            self._mark_ready("firmware started")
            return
        if outcome is FirmwareStartOutcome.FAILED:
            self.fail_prepare(
                self._device_error_message() or self.reject_message,
                "firmware reported failure",
            )
            return

        # In-firmware download progress (printers that pull from the CDN
        # themselves); no-op otherwise.
        self._on_progress_tick(changes)

        if (
            self._is_awaiting_firmware_start
            and time.monotonic() - self._prepare_awaiting_since > self.grace_seconds
        ):
            self.fail_prepare(
                self._device_error_message()
                or f"Print did not start within {int(self.grace_seconds)}s",
                "grace expired",
            )

    def _on_client_connect_change(self, *_args, **_kwargs) -> None:
        # Mid-prepare (dis)connect: land file_progress in ERROR so the backend
        # invariant (exactly one terminal per job_id) holds even when the
        # prepare is interrupted by a connection blip.
        if self.is_preparing_to_print:
            self.fail_prepare(
                "Printer connection changed during file transfer",
                "client (dis)connected",
            )

    @abstractmethod
    async def _upload(
        self, local_path: Path, on_progress: Callable[[float], None]
    ) -> PurePosixPath:
        """Push ``local_path`` to the printer; return its remote path."""

    @abstractmethod
    async def _send_start(
        self, path: PurePosixPath, data: FileDemandData, md5checksum: Optional[str]
    ) -> None:
        """Issue the brand command(s) that start printing ``path``."""

    @abstractmethod
    def _firmware_outcome(self, changes) -> FirmwareStartOutcome:
        """Classify a firmware state push while a prepare is in flight."""

    def _prepare_local_file(self, data: FileDemandData, local_path: Path) -> Path:
        """Transform the downloaded file for this printer (e.g. wrap as 3mf)."""
        return local_path

    def _locate_existing(
        self, data: FileDemandData
    ) -> Optional[Tuple[PurePosixPath, str]]:
        """Return an already-present remote file to skip re-upload, else None."""
        return None

    def _pre_ensure(self) -> None:
        """Run before locating/downloading (e.g. reset the FTP connection)."""

    def _on_progress_tick(self, changes) -> None:
        """React to in-firmware transfer progress (brands that download CDN-side)."""

    def _device_error_message(self) -> Optional[str]:
        """A specific user-facing message when the firmware reports an error."""
        return None

    def _upload_error_message(self, error: Exception) -> str:
        return f"Last error: {error}"

    def _on_begin_prepare(self) -> None:
        """Brand bookkeeping at prepare start (e.g. capture baseline task_id)."""

    def _on_end_prepare(self) -> None:
        """Brand bookkeeping at prepare end."""

    def _set_active_job_for_prepare(
        self, job_id: Optional[int], action_token: Optional[str]
    ) -> None:
        """Mark ``job_id`` as the printer's active job at prepare start.

        Records which job is now "the active job" (so later pause/cancel/resume
        target the right one) and resets the bed-cleared flag.
        """
        self.client.printer.current_job_id = job_id
        self.client.printer.file_action_token = action_token
        self.client.printer.have_cleared_bed = False
