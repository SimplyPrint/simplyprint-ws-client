"""Reusable file-transfer building blocks (prepare lifecycle, job locks).

The headline :class:`FileTransfer` owns the whole prepare -> download ->
transform -> upload -> await-firmware lifecycle and exposes brand hooks; a brand
subclasses it. The leaf primitives an integration's file handler also composes
are :func:`set_active_job` (active-job bookkeeping) and :func:`start_in_thread`.
"""

from simplyprint_ws_client.contrib.transfer.concurrency import start_in_thread
from simplyprint_ws_client.contrib.transfer.job_lock import set_active_job
from simplyprint_ws_client.contrib.transfer.file_transfer import (
    FileTransfer,
    FirmwareStartOutcome,
    PreparedPrint,
)

__all__ = [
    "FileTransfer",
    "FirmwareStartOutcome",
    "PreparedPrint",
    "set_active_job",
    "start_in_thread",
]
