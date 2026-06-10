"""Reusable file-transfer building blocks (prepare lifecycle).

The headline :class:`FileTransfer` owns the whole prepare -> download ->
transform -> upload -> await-firmware lifecycle and exposes brand hooks; a brand
subclasses it. Active-job bookkeeping at prepare start is now owned by
:class:`FileTransfer` as ``_set_active_job_for_prepare``. Leaf primitives an
integration's file handler also composes live here too.
"""

from simplyprint_ws_client.integration.transfer.download import download_to_file
from simplyprint_ws_client.integration.transfer.file_transfer import (
    FileTransfer,
    FirmwareStartOutcome,
    PreparedPrint,
)

__all__ = [
    "FileTransfer",
    "FirmwareStartOutcome",
    "PreparedPrint",
    "download_to_file",
]
