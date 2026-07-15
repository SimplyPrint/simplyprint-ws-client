import weakref

import pytest

from simplyprint_ws_client import (
    Client,
    ClientContext,
    PrinterConfig,
    PrinterState,
)
from simplyprint_ws_client.core.protocol.messages import JobInfoMsg
from simplyprint_ws_client.core.state import JobInfoState


def job_state_consistent(state: JobInfoState):
    return sum((state.started, state.finished, state.cancelled, state.failed)) <= 1


def build_job_state_msg(state: PrinterState) -> JobInfoMsg:
    return JobInfoMsg(data=dict(JobInfoMsg.build(state)))


@pytest.fixture
def job_client() -> Client:
    """Client fixture specific to job info tests (without id/in_setup setup)."""
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    return client


@pytest.fixture
def ctx(job_client: Client):
    return weakref.ref(job_client)


def test_exclusivity(job_client: Client, ctx):
    state = JobInfoState()
    state.provide_context(ctx)

    assert state.model_changed_fields == set()
    assert job_state_consistent(state)

    state.started = True
    assert state.model_changed_fields & {"started"}
    assert job_state_consistent(state)

    state.model_reset_changed()
    state.started = True

    assert state.model_changed_fields & {"started"}
    assert job_state_consistent(state)

    state.failed = True

    assert state.failed
    assert job_state_consistent(state)

    state.model_reset_changed()
    assert state.model_changed_fields == set()


def test_generated_message(job_client: Client, ctx):
    printer = PrinterState(config=job_client.config)
    printer.provide_context(ctx)
    state = printer.job_info

    assert state.model_changed_fields == set()
    assert job_state_consistent(state)

    assert dict(JobInfoMsg.build(printer)) == {}

    state.started = True

    assert state.model_changed_fields & {"started"}
    assert job_state_consistent(state)

    msg = build_job_state_msg(printer)
    assert msg.model_dump(mode="json") == {
        "type": "job_info",
        "data": {"started": True},
    }
    msg.reset_changes(printer)
    assert state.model_changed_fields == set()

    state.started = False

    msg = build_job_state_msg(printer)
    assert msg.model_dump(mode="json") == {"type": "job_info"}
    msg.reset_changes(printer)
    assert state.model_changed_fields == set()

    state.cancelled = True
    state.progress = 10
    state.reprint = True

    msg = build_job_state_msg(printer)

    assert msg.model_dump(mode="json") == {
        "type": "job_info",
        "data": {"cancelled": True, "progress": 10, "reprint": 1},
    }

    msg.reset_changes(printer)

    assert state.model_changed_fields == set()
    assert job_state_consistent(state)

    state.reprint = True

    msg = build_job_state_msg(printer)
    assert msg.model_dump(mode="json") == {"type": "job_info", "data": {"reprint": 1}}

    msg.reset_changes(printer)
    assert state.model_changed_fields == set()
