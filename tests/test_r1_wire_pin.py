"""Pin: in the normal (no-failure) path, the scheduler writes exactly what
``Client.consume()`` builds -- every frame, byte for byte, in order, once.

R1 moves the marker reset from build time to after the socket write. That must
not change a single normal-path frame: the brands' golden corpora pin
``consume()``, and this pins that the real write path (scheduler -> client ->
manager view -> connection -> protocol -> socket) reproduces it exactly, in
SINGLE and in MULTI mode (where the hub adds ``for``).

The reference is a twin client fed the same changes and drained with
``consume()``; a few frames are also pinned literally, so a change to
``consume()`` itself cannot hide here.
"""

from typing import Callable, List

import pytest

from simplyprint_ws_client import (
    Client,
    FileProgressStateEnum,
    PrinterStatus,
)
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.messages import (
    MultiPrinterAddConnectionMsg,
)
from simplyprint_ws_client.core.state import PrinterState

from tests._wire_harness import Wire, new_client, with_for

Step = Callable[[Client], None]


def _operational(c: Client) -> None:
    p: PrinterState = c.printer
    p.status = PrinterStatus.OPERATIONAL
    p.bed.temperature.actual = 21.0
    p.tool0.temperature.actual = 22.0
    p.firmware.name = "Marlin"
    p.firmware.version = "2.1"
    p.webcam_info.connected = True


def _start(c: Client) -> None:
    p = c.printer
    p.current_job_id = 42
    p.status = PrinterStatus.PRINTING
    p.job_info.started = True
    p.job_info.filename = "cube.gcode"
    p.job_info.progress = 1.0
    p.tool0.temperature.target = 210.0


def _progress(c: Client) -> None:
    # The start edge bypassed the job interval, so this uses it (goes out).
    c.printer.job_info.progress = 50.0


def _progress_again(c: Client) -> None:
    # Inside the job interval: rate limited, so nothing goes out yet.
    c.printer.job_info.progress = 60.0


def _finish(c: Client) -> None:
    c.printer.job_info.finished = True
    c.printer.status = PrinterStatus.OPERATIONAL


def _download(c: Client) -> None:
    p = c.printer
    p.current_job_id = 43
    p.file_progress.state = FileProgressStateEnum.DOWNLOADING
    p.file_progress.percent = 10.0


def _ready(c: Client) -> None:
    c.printer.file_progress.state = FileProgressStateEnum.READY


def _materials(c: Client) -> None:
    p = c.printer
    p.material0.type = "PLA"
    p.material0.hex = "#FFFFFF"
    p.light("main").set(True)
    p.filament_sensor.state = None
    p.psu_info.on = True


STEPS: List[Step] = [
    _operational,
    _start,
    _progress,
    _progress_again,
    _finish,
    _download,
    _ready,
    _materials,
]

#: The frames of ``_finish``: the rate-limited progress rides along with the
#: edge, job info goes before the status that changed after it, and the job
#: id is still there for the end.
FINISH_FRAMES = [
    '{"type":"job_info","data":{"job_id":42,"progress":60,"finished":true}}',
    '{"type":"state_change","data":{"new":"operational"}}',
]


def _reference_frames(ref: Client, step: Step) -> List[str]:
    step(ref)
    return [m.model_dump_json() for m in ref.consume()]


async def _run(mode: ConnectionMode):
    wire = Wire(mode)
    client = new_client()
    ref = new_client()

    await wire.open(client)
    # What the client's own `connected` handler did, for the twin.
    ref.printer.mark_common_fields_as_changed()

    written_before = len(wire.frames(client))
    expected: List[str] = _reference_frames(ref, lambda _c: None)
    await wire.schedule(client)
    per_step = [(None, wire.frames(client)[written_before:], expected)]

    for step in STEPS:
        before = len(wire.frames(client))
        step(client)
        await wire.schedule(client)
        per_step.append(
            (step.__name__, wire.frames(client)[before:], _reference_frames(ref, step))
        )

    return wire, client, per_step


@pytest.mark.asyncio
async def test_single_mode_writes_exactly_what_consume_builds():
    _wire, _client, per_step = await _run(ConnectionMode.SINGLE)

    for name, written, expected in per_step:
        assert written == expected, name

    finish = dict((name, written) for name, written, _ in per_step)["_finish"]
    assert finish == FINISH_FRAMES
    # The normal path is not empty: the pin compares real traffic.
    assert sum(len(written) for _, written, _ in per_step) >= 10


@pytest.mark.asyncio
async def test_multi_mode_writes_exactly_what_consume_builds_plus_for():
    wire, client, per_step = await _run(ConnectionMode.MULTI)
    uid = client.unique_id

    add = MultiPrinterAddConnectionMsg(client.config, True)
    add.for_client = uid
    # The hub wrote nothing before the add, and the add exactly once.
    assert wire.frames(client)[0] == add.model_dump_json()
    assert sum(f.startswith('{"type":"add_connection"') for f in wire.frames(client)) == 1

    for name, written, expected in per_step:
        assert written == [with_for(frame, uid) for frame in expected], name

    finish = dict((name, written) for name, written, _ in per_step)["_finish"]
    assert finish == [with_for(frame, uid) for frame in FINISH_FRAMES]
