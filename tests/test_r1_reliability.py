"""R1's client-side reliability inside frozen v0 (docs/r1-agreement.md).

Driven through the real write path (scheduler -> client -> view -> connection
-> protocol -> socket) over a hand-driven socket, so each case is
deterministic:

- a rate-limited burst sends its last value on the next allowed pass, with no
  new change needed to wake it (the ``last_msg_id`` bug);
- markers are reset only after the frame is written: a failed write or a dead
  socket loses nothing;
- an edge that provably never left survives the reconnect and goes first,
  before the full-state resend; one that may have left is not sent again;
- a reconnect resends every value-type message the server was ever sent.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import List, Optional

import pytest
import yarl

from simplyprint_ws_client import Client, PrinterStatus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.messages import JobInfoMsg
from simplyprint_ws_client.core.protocol.models import WriteOutcome
from simplyprint_ws_client.core.state.models import FilamentSensorEnum, Intervals
from simplyprint_ws_client.wire.errors import NotConnected
from simplyprint_ws_client.wire.transport import TransientError
from simplyprint_ws_client.wire.websockets import Websockets

from tests._wire_harness import Wire, new_client, with_for

MODES = [ConnectionMode.SINGLE, ConnectionMode.MULTI]


@pytest.fixture
def clock(monkeypatch):
    """The send intervals' clock, in ms, advanced by hand."""
    now = {"ms": 1_000_000}
    monkeypatch.setattr(Intervals, "now", staticmethod(lambda: now["ms"]))
    return now


async def _connected(mode: ConnectionMode = ConnectionMode.SINGLE):
    wire = Wire(mode)
    client = new_client()
    await wire.open(client)
    await wire.schedule(client)  # the connect's own frames
    return wire, client


def _payload(wire: Wire, client: Client, since: int) -> List[str]:
    """Frames written since ``since``, minus hub bookkeeping (adds)."""
    return [
        f
        for f in wire.frames(client)[since:]
        if not f.startswith('{"type":"add_connection"')
    ]


def _expect(mode: ConnectionMode, client: Client, frames: List[str]) -> List[str]:
    if mode is ConnectionMode.MULTI:
        return [with_for(f, client.unique_id) for f in frames]
    return frames


# -- 1. rate limiting: suppressed changes stay pending -------------------------


@pytest.mark.asyncio
async def test_a_rate_limited_burst_sends_its_last_value_without_a_new_change(clock):
    wire, client = await _connected()
    temperature = client.printer.tool0.temperature

    temperature.actual = 20.0
    await wire.schedule(client)
    assert wire.frames(client)[-1] == '{"type":"temps","data":{"tool0":[20]}}'

    before = len(wire.frames(client))
    for value in (21.0, 22.0, 23.0):
        temperature.actual = value
        await wire.schedule(client)
    assert wire.frames(client)[before:] == []  # inside the temps interval
    # Nothing new to announce it -- before R1 this is where 23 went stale.
    assert not client.has_changes
    assert client.has_deferred

    clock["ms"] += client.printer.intervals.temps
    await wire.schedule(client)  # the scheduler's next pass, no new change

    assert wire.frames(client)[before:] == ['{"type":"temps","data":{"tool0":[23]}}']
    assert not client.has_deferred


# -- 2. markers reset only after a successful write ----------------------------


@pytest.mark.asyncio
async def test_a_failed_write_keeps_the_change_for_the_next_send():
    wire, client = await _connected()
    socket = wire.socket(client)
    before = len(socket.sent)

    socket.fail_writes = TransientError("the socket died under the write")
    client.printer.status = PrinterStatus.PRINTING
    await wire.schedule(client)

    assert len(socket.sent) == before
    assert client.printer.model_has_changes("status")  # still pending
    assert client.has_deferred

    socket.fail_writes = None
    await wire.schedule(client)  # the retry pass, no new change

    assert socket.sent[before:] == ['{"type":"state_change","data":{"new":"printing"}}']
    assert not client.printer.model_has_changes("status")


@pytest.mark.asyncio
async def test_the_protocol_records_what_became_of_each_frame():
    wire, client = await _connected()
    socket = wire.socket(client)
    sent = []

    original = client.send

    async def recording_send(msg, skip_dispatch=False):
        await original(msg, skip_dispatch=skip_dispatch)
        sent.append(msg.write_outcome)

    client.send = recording_send

    client.printer.status = PrinterStatus.PRINTING
    await wire.schedule(client)
    socket.fail_writes = TransientError("boom")
    client.printer.status = PrinterStatus.PAUSED
    await wire.schedule(client)
    socket.fail_writes = None
    socket.drop()
    client.printer.status = PrinterStatus.OPERATIONAL
    await wire.schedule(client)

    assert sent == [
        WriteOutcome.WRITTEN,
        WriteOutcome.MAYBE_SENT,
        WriteOutcome.NOT_SENT,
    ]


# -- 3. reconnect: kept edges first, then the full state -----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_an_edge_that_never_left_goes_first_after_the_reconnect(mode):
    wire, client = await _connected(mode)
    printer = client.printer
    printer.current_job_id = 42
    printer.status = PrinterStatus.PRINTING
    printer.job_info.started = True
    printer.job_info.progress = 10.0
    await wire.schedule(client)

    # The socket dies before the protocol hears of it; the job ends meanwhile.
    socket = wire.socket(client)
    socket.drop()
    printer.job_info.finished = True
    printer.status = PrinterStatus.OPERATIONAL
    before = len(socket.sent)
    await wire.schedule(client)

    assert len(socket.sent) == before  # nothing reached the socket
    assert printer.current_job_id == 42  # the kept edge keeps its job id

    await wire.drop(client)
    await wire.open(client)
    await wire.schedule(client)
    after = _payload(wire, client, before)

    assert after[:2] == _expect(
        mode,
        client,
        [
            '{"type":"job_info","data":{"job_id":42,"finished":true}}',
            '{"type":"state_change","data":{"new":"operational"}}',
        ],
    )
    assert sum('"finished":true' in f for f in after) == 1
    assert printer.current_job_id is None  # released once it went out


@pytest.mark.asyncio
async def test_an_edge_that_may_have_left_is_not_sent_again():
    wire, client = await _connected()
    printer = client.printer
    printer.current_job_id = 7
    printer.status = PrinterStatus.PRINTING
    printer.job_info.started = True
    await wire.schedule(client)

    socket = wire.socket(client)
    socket.fail_writes = TransientError("died mid-write")
    printer.job_info.finished = True
    await wire.schedule(client)
    socket.fail_writes = None
    before = len(socket.sent)

    await wire.drop(client)
    await wire.open(client)
    await wire.schedule(client)
    after = _payload(wire, client, before)

    # At-most-once: no acknowledgements, so a possibly written edge stays sent.
    assert not any('"finished":true' in f for f in after)
    # The state is still resent in full.
    assert '{"type":"state_change","data":{"new":"printing"}}' in after


@pytest.mark.asyncio
async def test_a_reconnect_resends_every_value_the_server_was_sent(clock):
    wire, client = await _connected()
    printer = client.printer
    printer.status = PrinterStatus.OPERATIONAL
    printer.bed.temperature.actual = 60.0
    printer.tool0.temperature.actual = 200.0
    printer.webcam_info.connected = True
    printer.cpu_info.usage = 12.0
    printer.psu_info.on = True
    printer.filament_sensor.state = FilamentSensorEnum.LOADED
    printer.light("main").set(True)
    printer.material0.type = "PLA"
    await wire.schedule(client)

    clock["ms"] += 60_000  # every send interval is open again
    before = len(wire.frames(client))
    await wire.drop(client)
    await wire.open(client)
    await wire.schedule(client)
    after = _payload(wire, client, before)

    assert {json.loads(f)["type"] for f in after} == {
        "state_change",
        "machine_data",
        "firmware",
        "temps",
        "webcam_status",
        "cpu_info",
        "psu",
        "filament_sensor",
        "peripheral",
        "material_data",
    }
    # Never sent before (ambient, job info, file progress, latency): not
    # invented from defaults now.
    assert '{"type":"temps","data":{"bed":[60],"tool0":[200]}}' in after
    assert '{"type":"state_change","data":{"new":"operational"}}' in after


@pytest.mark.asyncio
async def test_an_edge_taken_over_mid_write_is_the_takers():
    # An integration about to overwrite what an in-flight edge was built from
    # takes it over: reset at once, its job id stays released, and its
    # outcome is left to the taker (here: nobody resends it).
    wire, client = await _connected()
    printer = client.printer
    printer.current_job_id = 42
    printer.status = PrinterStatus.PRINTING
    printer.job_info.started = True
    await wire.schedule(client)

    socket = wire.socket(client)
    original = socket.write
    taken: List[object] = []

    async def write(data: object) -> None:
        if '"finished":true' in str(data):
            taken.append(client.take_held_edge(JobInfoMsg))
            assert client.take_held_edge(JobInfoMsg) is None  # taken once
            raise NotConnected("the link died before the write")
        await original(data)

    socket.write = write
    printer.job_info.finished = True
    await wire.schedule(client)

    (msg,) = taken
    assert isinstance(msg, JobInfoMsg) and msg.data["finished"] is True
    assert msg.write_outcome is WriteOutcome.NOT_SENT
    assert not printer.job_info.model_has_changes("finished")
    assert printer.current_job_id is None  # not restored over the taker's

    before = len(socket.sent)
    await wire.schedule(client)
    assert not any('"finished":true' in f for f in socket.sent[before:])


@pytest.mark.asyncio
async def test_a_send_stops_at_the_first_frame_that_did_not_go_out():
    # The link dies under the kept edge's write and a reconnect completes
    # before the next write: had the send gone on, the status would reach
    # the new socket ahead of the finished, starting the server's job-failure
    # clock. The rest of the send waits for the next one, in its order.
    wire, client = await _connected()
    printer = client.printer
    printer.current_job_id = 9
    printer.status = PrinterStatus.PRINTING
    printer.job_info.started = True
    await wire.schedule(client)

    socket = wire.socket(client)
    original = socket.write
    failed: List[str] = []

    async def write(data: object) -> None:
        if '"finished":true' in str(data) and not failed:
            failed.append(str(data))
            raise NotConnected("the old link died under this write")
        await original(data)  # the new link takes the rest

    socket.write = write
    printer.job_info.finished = True
    printer.status = PrinterStatus.OPERATIONAL
    before = len(socket.sent)
    await wire.schedule(client)

    assert failed and socket.sent[before:] == []
    assert client.has_deferred

    await wire.schedule(client)  # the retry pass
    assert socket.sent[before:] == [
        '{"type":"job_info","data":{"job_id":9,"finished":true}}',
        '{"type":"state_change","data":{"new":"operational"}}',
    ]


# -- 4. a write on a socket the library already knows is closed ----------------


class _LibrarySocket:
    """A ``websockets`` connection's shape: the library's protocol state, and a
    ``send`` that records what it was given."""

    def __init__(self, state, error: Optional[Exception] = None) -> None:
        self.protocol = SimpleNamespace(state=state)
        self.sent: List[object] = []
        self._error = error

    async def send(self, frame: object) -> None:
        if self._error is not None:
            raise self._error
        self.sent.append(frame)


def _websockets(socket) -> Websockets:
    transport = Websockets(
        yarl.URL("ws://x"), provider=EventLoopProvider(loop=asyncio.get_running_loop())
    )
    transport.socket = socket
    return transport


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["CLOSING", "CLOSED"])
async def test_a_write_on_a_socket_the_library_knows_is_closed_never_left(state):
    # websockets raises ConnectionClosed before writing anything once its
    # state is not OPEN: the frame provably never left, so the protocol must
    # record NOT_SENT (NotConnected), not MAYBE_SENT. Otherwise a `finished`
    # written after the server closed, before supervise noticed, is forgotten.
    from websockets.protocol import State

    socket = _LibrarySocket(State[state])
    with pytest.raises(NotConnected):
        await _websockets(socket).write("frame")
    assert socket.sent == []


@pytest.mark.asyncio
async def test_a_write_with_no_socket_never_left():
    with pytest.raises(NotConnected):
        await _websockets(None).write("frame")


@pytest.mark.asyncio
async def test_a_write_that_fails_on_an_open_socket_may_have_left():
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close
    from websockets.protocol import State

    error = ConnectionClosedError(None, Close(1011, "keepalive ping timeout"))
    with pytest.raises(TransientError) as raised:
        await _websockets(_LibrarySocket(State.OPEN, error)).write("frame")
    assert not isinstance(raised.value, NotConnected)
