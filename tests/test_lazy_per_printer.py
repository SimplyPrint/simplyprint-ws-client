"""A ``lazy_per_printer`` client behaves as one that builds everything with
each printer: its class routes dispatch as registered routes would, its state
models lock as eagerly locked ones, and its camera controller, built on first
use, serves the first demand."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest
from yarl import URL

from simplyprint_ws_client import ClientContext, PrinterClient, PrinterConfig
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.utils.synchronized import locked_on_first_use
from simplyprint_ws_client.core.client import class_routes
from simplyprint_ws_client.core.protocol.messages import WebcamSnapshotDemandData
from simplyprint_ws_client.core.protocol.models import DemandMsgType, ServerMsgType
from simplyprint_ws_client.core.state import PrinterState
from simplyprint_ws_client.events.event import Event
from simplyprint_ws_client.events.event_bus import EventBus, Routes
from simplyprint_ws_client.integration.camera.controller import CameraController


class _Eager(PrinterClient[PrinterConfig]):
    def __init__(self, camera_pool=None) -> None:
        config = PrinterConfig.get_new()
        config.id = 1
        try:  # the camera schedules onto the client's loop
            provider = EventLoopProvider(loop=asyncio.get_running_loop())
        except RuntimeError:
            provider = None
        context = ClientContext(camera_pool=camera_pool, event_loop_provider=provider)
        super().__init__(config, context=context)

    def resolve_camera_uri(self):
        return URL("http://camera.local/stream")


class _Lazy(_Eager):
    lazy_per_printer = True


def _table(bus: EventBus) -> list:
    """Every event's listeners as dispatched: (event, [(function, priority)])."""
    return sorted(
        (
            (repr(event), [(l.handler.__func__, l.priority) for l in listeners.ordered()])
            for event, listeners in bus.listeners.items()
        ),
        key=lambda row: row[0],
    )


# --- the routes -----------------------------------------------------------------


def test_class_routes_are_the_registered_routes():
    eager, lazy = _Eager(), _Lazy()
    assert lazy._routed and not eager._routed
    assert lazy.event_bus._listeners == {}  # nothing registered per printer
    assert _table(lazy.event_bus) == _table(eager.event_bus)
    assert class_routes(_Lazy) is class_routes(_Lazy)  # once per class


class _Recorder:
    def __init__(self, name):
        self.name = name

    def first(self, *args):
        CALLS.append((self.name, "first", args))
        return ("changed",)

    async def second(self, *args):
        CALLS.append((self.name, "second", args))

    def third(self, *args):
        CALLS.append((self.name, "third", args))


class _Ping(Event): ...


CALLS: list = []
_ROUTES = (
    ("key", _Recorder.second, {}),
    ("key", _Recorder.first, {"priority": 1}),
    (_Ping, _Recorder.third, {}),
)


def _buses():
    """A bus with the routes registered, and one given them as class routes."""
    registered_owner, routed_owner = _Recorder("registered"), _Recorder("routed")
    registered = EventBus()
    for event, function, options in _ROUTES:
        registered.on(event, function.__get__(registered_owner), **options)
    routed = EventBus(routes=Routes(_ROUTES), owner=routed_owner)
    return registered, routed


def _arg(value):
    return type(value) if isinstance(value, Event) else value


async def _emitted(bus, *events) -> list:
    CALLS.clear()
    for event in events:
        await bus.emit(event, 1)
    bus.emit_sync("key", 2)
    return [(kind, tuple(map(_arg, args))) for _, kind, args in CALLS]


@pytest.mark.asyncio
async def test_routed_dispatch_is_registered_dispatch():
    registered, routed = _buses()
    events = ("key", _Ping(), "none")
    assert await _emitted(routed, *events) == await _emitted(registered, *events)
    assert CALLS and CALLS[0][0] == "registered"


@pytest.mark.asyncio
async def test_a_registration_after_the_routes_comes_after_them():
    registered, routed = _buses()
    late = []
    for bus in (registered, routed):
        bus.on("key", lambda *args: late.append(args))
        bus.on(_Ping, lambda *args: late.append(args), priority=5)
    assert await _emitted(routed, "key", _Ping()) == await _emitted(
        registered, "key", _Ping()
    )
    assert len(late) == 6


@pytest.mark.asyncio
async def test_off_and_clear_reach_the_routes():
    registered, routed = _buses()
    for bus in (registered, routed):
        owner = next(iter(bus.listeners["key"])).handler.__self__
        bus.off("key", owner.first)
        bus.clear(_Ping)
    assert await _emitted(routed, "key", _Ping()) == await _emitted(
        registered, "key", _Ping()
    ) == [("second", (1,))]
    routed.clear_all()
    assert routed.listeners == {} and await _emitted(routed, "key") == []


def test_a_replaced_route_method_is_seen_by_the_next_printer(monkeypatch):
    before = class_routes(_Lazy)
    replaced = AsyncMock()
    monkeypatch.setattr(_Lazy, "on_pause", replaced)
    assert class_routes(_Lazy) is None  # not a plain method: registered per printer
    client = _Lazy()
    assert not client._routed
    listeners = client.event_bus.listeners[DemandMsgType.PAUSE].ordered()
    assert [l.handler for l in listeners] == [client.on_pause]
    monkeypatch.undo()
    assert class_routes(_Lazy) is not before  # built again for the methods now
    assert _Lazy()._routed


@pytest.mark.asyncio
async def test_a_routed_demand_reaches_the_printer():
    client = _Lazy()
    client.on_pause = AsyncMock()  # an instance's own attribute is not a route
    seen = []

    async def on_pause(self, data):
        seen.append((self, data))

    _Lazy.on_pause, original = on_pause, _Lazy.on_pause
    try:
        routed = _Lazy()
        await routed.event_bus.emit(DemandMsgType.PAUSE, "data")
        await routed.event_bus.emit(ServerMsgType.PONG, None)  # a core route
    finally:
        _Lazy.on_pause = original
    assert seen == [(routed, "data")]


# --- the locks ------------------------------------------------------------------


def _locks(model, out=None) -> list:
    out = [] if out is None else out
    out.append("_Synchronized__lock" in model.__dict__)
    for value in model.__dict__.values():
        children = value if isinstance(value, list) else [value]
        for child in children:
            if hasattr(child, "model_self_changed_fields"):
                _locks(child, out)
    return out


def test_state_built_lazily_locks_on_first_use():
    state = locked_on_first_use(lambda: PrinterState(config=PrinterConfig.get_new()))
    assert not any(_locks(state))
    assert all(_locks(PrinterState(config=PrinterConfig.get_new())))
    with state.bed:
        lock = state.bed.__dict__["_Synchronized__lock"]
        assert lock.locked()
        with pytest.raises(RuntimeError, match="Deadlock"):
            state.bed.__enter__()
    assert not lock.locked()
    assert all(_locks(PrinterState(config=PrinterConfig.get_new())))  # flag reset


def test_two_threads_entering_first_share_one_lock():
    state = locked_on_first_use(lambda: PrinterState(config=PrinterConfig.get_new()))
    barrier, locks = threading.Barrier(8), []

    def enter():
        barrier.wait()
        with state.bed:
            locks.append(state.bed.__dict__["_Synchronized__lock"])

    threads = [threading.Thread(target=enter) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(map(id, locks))) == 1


def test_a_lazy_printer_and_its_reset_state_lock_on_first_use():
    client = _Lazy()
    assert not any(_locks(client.printer))
    assert not any(_locks(client.reset_printer_state()))


# --- the camera -----------------------------------------------------------------


def test_no_controller_until_the_camera_is_used():
    client = _Lazy()
    client.update_camera_uri()  # refused without a pool: as a new one would
    client.clear_camera_uri()
    assert client._camera_controller is None
    assert client.printer.webcam_info.connected is False


class _Handle:
    def __init__(self):
        self.id, self.fps, self.stopped = 1, None, False

    async def receive_frame(self, *args, **kwargs):
        return b"FRAME"

    def stop(self):
        self.stopped = True


class _Pool:
    def __init__(self):
        self.handles = []

    def create(self, uri, **kwargs):
        self.handles.append(_Handle())
        return self.handles[-1]


@pytest.mark.asyncio
async def test_the_camera_works_on_first_use():
    pool = _Pool()
    client = _Lazy(camera_pool=pool)
    sent = []
    client.send = AsyncMock(side_effect=lambda msg, *a, **k: sent.append(msg))
    client.update_camera_uri()  # its first use: the URI
    camera = client._camera_controller
    assert isinstance(camera, CameraController) and client.camera is camera
    assert client.printer.webcam_info.connected is True
    assert len(pool.handles) == 1

    # A stream frame (no id): sent to SimplyPrint as the stream.
    await client.event_bus.emit(DemandMsgType.WEBCAM_SNAPSHOT, WebcamSnapshotDemandData())
    for _ in range(200):
        if sent:
            break
        await asyncio.sleep(0.01)
    assert sent, "the first stream request was not served"
    await client.teardown()
    assert pool.handles[0].stopped


@pytest.mark.asyncio
async def test_a_demand_builds_the_controller():
    client = _Lazy()
    # Without a camera the demand waits (bounded) for one: built meanwhile.
    demand = asyncio.ensure_future(client.event_bus.emit(DemandMsgType.STREAM_OFF, None))
    await asyncio.sleep(0)
    assert isinstance(client._camera_controller, CameraController)
    demand.cancel()
    await asyncio.gather(demand, return_exceptions=True)
    await client.teardown()


@pytest.mark.asyncio
async def test_the_camera_reports_into_the_reset_state():
    for cls in (_Eager, _Lazy):
        client = cls(camera_pool=_Pool())
        client.update_camera_uri()
        state = client.reset_printer_state()
        client.clear_camera_uri()
        client.update_camera_uri()
        assert state.webcam_info.connected is True, cls
