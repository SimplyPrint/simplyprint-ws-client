"""Active-job bookkeeping shared by every integration.

When a file demand arrives, the client records which job is now "the active
job" on the printer (so later pause/cancel/resume target the right one) and
clears the bed-cleared flag. This was the same three lines in every brand's
file handler.
"""

from __future__ import annotations

from typing import Optional

from simplyprint_ws_client.core.state import PrinterState


def set_active_job(
    printer: PrinterState, job_id: Optional[int], action_token: Optional[str]
) -> None:
    """Mark ``job_id`` as the printer's active job and reset the bed-cleared flag."""
    printer.current_job_id = job_id
    printer.file_action_token = action_token
    printer.have_cleared_bed = False
