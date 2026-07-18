from simplyprint_ws_client import Client, PrinterStatus
from simplyprint_ws_client.core.protocol.messages import JobInfoMsg
from simplyprint_ws_client.core.state import Interval
from tests.test_intervals import TimeControlledIntervals


def test_projection_stays_pending_until_committed(client: Client):
    client.printer.job_info.progress = 12

    first = client.pending_messages()

    assert client.has_changes
    assert [item.message.data for item in first] == [{"progress": 12}]
    assert [item.message.data for item in client.pending_messages()] == [
        {"progress": 12}
    ]

    client.commit_message(first[0])

    assert not client.has_changes
    assert client.pending_messages() == []


def test_commit_preserves_a_newer_mutation(client: Client):
    client.printer.intervals.job = 0
    client.printer.job_info.progress = 12
    pending = client.pending_messages()[0]

    client.printer.job_info.progress = 34
    client.commit_message(pending)

    assert client.has_changes
    current = client.pending_messages()
    assert len(current) == 1
    assert isinstance(current[0].message, JobInfoMsg)
    assert current[0].message.data == {"progress": 34}


def test_rate_limited_progress_keeps_the_client_dirty(client: Client):
    intervals = client.printer.intervals = TimeControlledIntervals()
    intervals.set_time(30_000)
    client.printer.status = PrinterStatus.PRINTING
    status = client.pending_messages()
    for item in status:
        client.commit_message(item)

    client.printer.job_info.progress = 10
    first = client.pending_messages()
    assert len(first) == 1
    client.commit_message(first[0])
    assert not intervals.is_ready(Interval.JOB)

    client.printer.job_info.progress = 20

    assert client.pending_messages() == []
    assert client.has_changes

    intervals.step_time(intervals.job)
    pending = client.pending_messages()
    assert [item.message.data for item in pending] == [{"progress": 20}]
