from simplyprint_ws_client import (
    Client,
    PrinterStatus,
    FileProgressStateEnum,
    JobInfoMsg,
    FileProgressMsg,
    StateChangeMsg,
)


def test_job_id_consistency_flow(client: Client):
    """Test that current_job_id stays consistent throughout operational->downloading->printing flow."""

    # Start with operational state and no job_id
    client.printer.status = PrinterStatus.OPERATIONAL
    client.printer.current_job_id = None

    # Clear any initial messages
    client.consume()

    # Set a job_id and transition to downloading
    job_id = 12345
    client.printer.current_job_id = job_id
    client.printer.status = PrinterStatus.DOWNLOADING
    client.printer.file_progress.state = FileProgressStateEnum.DOWNLOADING
    client.printer.file_progress.percent = 25.0

    msgs, _ = client.consume()

    # Find the relevant messages
    status_msg = next((m for m in msgs if isinstance(m, StateChangeMsg)), None)
    file_progress_msg = next((m for m in msgs if isinstance(m, FileProgressMsg)), None)

    # Both messages should contain the same job_id
    assert status_msg is not None
    assert file_progress_msg is not None

    assert file_progress_msg.data["job_id"] == job_id
    assert client.printer.current_job_id == job_id

    # Transition to printing status, emit job_info
    client.printer.status = PrinterStatus.PRINTING
    client.printer.job_info.started = True
    client.printer.job_info.progress = 10.0

    msgs, _ = client.consume()

    # Find the relevant messages
    job_info_msg = next((m for m in msgs if isinstance(m, JobInfoMsg)), None)
    status_msg = next((m for m in msgs if isinstance(m, StateChangeMsg)), None)

    assert job_info_msg is not None
    assert status_msg is not None

    # Job info message should contain the same job_id
    assert job_info_msg.data["job_id"] == job_id
    assert client.printer.current_job_id == job_id

    # Continue printing with progress updates
    client.printer.job_info.progress = 50.0

    msgs, _ = client.consume()
    job_info_msg = next((m for m in msgs if isinstance(m, JobInfoMsg)), None)

    # Should still have the same job_id
    assert job_info_msg is not None
    assert job_info_msg.data["job_id"] == job_id
    assert client.printer.current_job_id == job_id

    # Finish the job
    client.printer.job_info.finished = True

    msgs, _ = client.consume()
    job_info_msg = next((m for m in msgs if isinstance(m, JobInfoMsg)), None)

    # Should still have job_id in the finished message
    assert job_info_msg is not None
    assert job_info_msg.data["job_id"] == job_id

    # After reset_changes is called (simulating message dispatch), job_id should be cleared
    job_info_msg.reset_changes(client.printer)
    assert client.printer.current_job_id is None

    # Going back to operational should also clear job_id if it wasn't already cleared
    client.printer.current_job_id = job_id  # Set it again
    client.printer.status = PrinterStatus.OPERATIONAL

    msgs, _ = client.consume()
    status_msg = next((m for m in msgs if isinstance(m, StateChangeMsg)), None)

    assert status_msg is not None

    # After reset_changes is called, job_id should be cleared
    status_msg.reset_changes(client.printer)
    assert client.printer.current_job_id is None
