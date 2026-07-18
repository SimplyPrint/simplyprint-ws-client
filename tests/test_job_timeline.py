from unittest.mock import patch

import pytest

from simplyprint_ws_client import ClientContext, PrinterConfig, PrinterStatus
from simplyprint_ws_client.core.job import (
    JobOutcome,
    NativeJobObservation,
    NativeJobTerminal,
)
from simplyprint_ws_client.core.protocol.messages import JobInfoMsg, StateChangeMsg
from simplyprint_ws_client.integration.client import JobEdge, PrinterClient


class TimelinePrinter(PrinterClient[PrinterConfig]):
    def __init__(self) -> None:
        config = PrinterConfig.get_new()
        config.id = 1
        config.in_setup = False
        super().__init__(config, context=ClientContext())
        self.outcome = JobOutcome.FINISHED

    def on_job_start(self, _edge: JobEdge) -> None:
        self.printer.job_info.filename = "part.gcode"

    def on_job_finish(self, _edge: JobEdge) -> JobOutcome:
        return self.outcome


@pytest.fixture
def printer() -> TimelinePrinter:
    return TimelinePrinter()


def _commit_all(printer: TimelinePrinter) -> list:
    pending = printer.pending_messages()
    for item in pending:
        printer.commit_message(item)
    return [item.message for item in pending]


def test_start_and_terminal_survive_as_ordered_snapshots(printer: TimelinePrinter):
    printer.printer.status = PrinterStatus.OPERATIONAL
    _commit_all(printer)

    with patch("simplyprint_ws_client.core.job.time.monotonic", return_value=100.0):
        printer.apply_status(PrinterStatus.PRINTING)
    with patch("simplyprint_ws_client.core.job.time.monotonic", return_value=105.0):
        printer.apply_status(PrinterStatus.OPERATIONAL)

    with patch("simplyprint_ws_client.core.job.time.monotonic", return_value=110.0):
        pending = printer.pending_messages()

    assert [item.message.type for item in pending] == [
        "job_info",
        "job_info",
        "state_change",
    ]
    assert pending[0].message.data == {
        "filename": "part.gcode",
        "started": True,
        "delay": 10,
    }
    assert pending[1].message.data == {
        "filename": "part.gcode",
        "finished": True,
        "delay": 5,
    }
    assert pending[2].message.data == {"new": PrinterStatus.OPERATIONAL}

    printer.commit_message(pending[0])
    remaining = printer.pending_messages()
    assert isinstance(remaining[0].message, JobInfoMsg)
    assert remaining[0].message.data["finished"] is True


def test_terminal_send_owns_backend_job_id_release(printer: TimelinePrinter):
    printer.printer.status = PrinterStatus.OPERATIONAL
    _commit_all(printer)
    printer.printer.current_job_id = 42
    printer.apply_status(PrinterStatus.PRINTING)
    _commit_all(printer)

    printer.apply_status(PrinterStatus.OPERATIONAL)
    pending = printer.pending_messages()
    terminal = next(
        item
        for item in pending
        if isinstance(item.message, JobInfoMsg)
        and item.message.data.get("finished") is True
    )
    status = next(item for item in pending if isinstance(item.message, StateChangeMsg))

    printer.commit_message(status)
    assert printer.printer.current_job_id == 42

    printer.commit_message(terminal)
    assert printer.printer.current_job_id is None


def test_same_native_job_suppresses_transient_operational(printer: TimelinePrinter):
    printer.printer.status = PrinterStatus.OPERATIONAL
    _commit_all(printer)
    observation = NativeJobObservation(active_id="native-7")
    printer.apply_status(PrinterStatus.PRINTING, job_observation=observation)
    _commit_all(printer)

    applied = printer.apply_status(
        PrinterStatus.OPERATIONAL,
        job_observation=observation,
    )

    assert applied == PrinterStatus.PRINTING
    assert printer.printer.status == PrinterStatus.PRINTING
    assert not printer.job_timeline.has_pending


def test_native_terminal_precedes_operational_status(printer: TimelinePrinter):
    printer.printer.status = PrinterStatus.OPERATIONAL
    _commit_all(printer)
    printer.apply_status(
        PrinterStatus.PRINTING,
        job_observation=NativeJobObservation(active_id="native-8"),
    )
    _commit_all(printer)

    printer.apply_status(
        PrinterStatus.OPERATIONAL,
        job_observation=NativeJobObservation(
            active_id=None,
            terminal=NativeJobTerminal("native-8", JobOutcome.FAILED),
        ),
    )
    pending = printer.pending_messages()

    assert isinstance(pending[0].message, JobInfoMsg)
    assert pending[0].message.data["failed"] is True
    assert isinstance(pending[1].message, StateChangeMsg)
    assert pending[1].message.data == {"new": PrinterStatus.OPERATIONAL}


def test_authoritative_offline_ends_the_active_job_first(printer: TimelinePrinter):
    printer.printer.status = PrinterStatus.OPERATIONAL
    _commit_all(printer)
    printer.apply_status(PrinterStatus.PRINTING)
    _commit_all(printer)
    printer.outcome = JobOutcome.FAILED

    printer.apply_status(PrinterStatus.OFFLINE)
    pending = printer.pending_messages()

    assert isinstance(pending[0].message, JobInfoMsg)
    assert pending[0].message.data["failed"] is True
    assert isinstance(pending[1].message, StateChangeMsg)
    assert pending[1].message.data == {"new": PrinterStatus.OFFLINE}


def test_native_identity_change_finishes_before_starting_replacement(
    printer: TimelinePrinter,
):
    printer.printer.status = PrinterStatus.OPERATIONAL
    _commit_all(printer)
    printer.apply_status(
        PrinterStatus.PRINTING,
        job_observation=NativeJobObservation(active_id="native-old"),
    )
    _commit_all(printer)

    printer.outcome = JobOutcome.CANCELLED
    printer.apply_status(
        PrinterStatus.PRINTING,
        job_observation=NativeJobObservation(active_id="native-new"),
    )
    pending = printer.pending_messages()

    assert [item.message.data for item in pending] == [
        {"cancelled": True},
        {"filename": "part.gcode", "started": True},
    ]
