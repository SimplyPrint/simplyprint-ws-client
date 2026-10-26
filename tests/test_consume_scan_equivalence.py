"""``Client.consume`` pinned against the full-tree scan it was built on.

``_reference_message_changes`` and ``_ScanClient.consume`` below are the scan
and the consume of the framework before change tracking went per kind, copied
verbatim: every consume walks the whole printer state for every message kind.
A seeded random walk of state edits, sends, settles, aborted sends, rate
limits, reconnect resends and pending toggles drives a real :class:`Client`
and a ``_ScanClient`` side by side; after every step both must agree exactly:

- the messages built (kinds, order, data, key order);
- every change marker left in the tree, the message ids, ``has_changes``,
  ``has_deferred`` and the current job id.

So the shortcut ``consume`` takes (reading only the kinds a change can have
reached) is held to what the plain scan does; the kinds it read as changed are
checked against the scan's directly too, versions and order included.
"""

import datetime
import random
import uuid
import weakref
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.common.model.reactive import ReactiveModel
from simplyprint_ws_client.core import client as client_module
from simplyprint_ws_client.core.protocol.messages import (
    AmbientTemperatureMsg,
    ClientMsg,
    CpuInfoMsg,
    FilamentSensorMsg,
    FileProgressMsg,
    FirmwareMsg,
    FirmwareWarningMsg,
    JobInfoMsg,
    LatencyMsg,
    MachineDataMsg,
    MaterialDataMsg,
    NotificationMsg,
    PeripheralMsg,
    PowerControllerMsg,
    StateChangeMsg,
    TemperatureMsg,
    ToolMsg,
    WebcamMsg,
    WebcamStatusMsg,
)
from simplyprint_ws_client.core.protocol.models import DispatchMode, WriteOutcome
from simplyprint_ws_client.core.state.models import (
    BedType,
    FilamentSensorEnum,
    FileProgressStateEnum,
    Intervals,
    MultiMaterialSolution,
    NozzleType,
    PrinterSettings,
    PrinterStatus,
    VolumeType,
)
from simplyprint_ws_client.core.state.notifications import NotificationEventPayload
from simplyprint_ws_client.core.state.printer import (
    MaterialEntry,
    MaterialLayoutEntry,
    PrinterState,
)

# -- the reference: the full scan, verbatim -----------------------------------


def _reference_field_version(
    model: ReactiveModel,
    field: str,
    nested: Iterable[ReactiveModel] = (),
) -> Optional[int]:
    versions = []
    if field in model.model_self_changed_fields:
        versions.append(model.model_self_changed_fields[field])
    for child in nested:
        if changes := child.model_recursive_changeset:
            versions.append(max(changes.values()))
    return max(versions) if versions else None


def _reference_message_changes(
    state: PrinterState,
) -> Iterable[Tuple[Type[ClientMsg], Iterable[Optional[int]]]]:
    fv = _reference_field_version
    yield MachineDataMsg, (fv(state, "info", (state.info,)),)
    yield WebcamStatusMsg, (fv(state.webcam_info, "connected"),)
    yield WebcamMsg, (fv(state, "webcam_settings", (state.webcam_settings,)),)
    yield FirmwareMsg, (fv(state, "firmware", (state.firmware,)),)
    yield (
        FirmwareWarningMsg,
        (fv(state, "firmware_warning", (state.firmware_warning,)),),
    )
    yield ToolMsg, (fv(tool, "active_material") for tool in state.tools)
    yield (
        TemperatureMsg,
        (
            fv(state.bed, "temperature", (state.bed.temperature,)),
            fv(state.chamber, "temperature", (state.chamber.temperature,)),
            *(fv(tool, "temperature", (tool.temperature,)) for tool in state.tools),
        ),
    )
    yield AmbientTemperatureMsg, (fv(state.ambient_temperature, "ambient"),)
    yield StateChangeMsg, (fv(state, "status"),)
    yield JobInfoMsg, (fv(state, "job_info", (state.job_info,)),)
    yield LatencyMsg, (fv(state.latency, "pong"),)
    yield FileProgressMsg, (fv(state, "file_progress", (state.file_progress,)),)
    yield (
        FilamentSensorMsg,
        (fv(state, "filament_sensor", (state.filament_sensor,)),),
    )
    yield PowerControllerMsg, (fv(state, "psu_info", (state.psu_info,)),)
    yield CpuInfoMsg, (fv(state, "cpu_info", (state.cpu_info,)),)
    yield (
        MaterialDataMsg,
        (
            *(fv(tool, "materials", tool.materials) for tool in state.tools),
            *(fv(tool, "size") for tool in state.tools),
            *(fv(tool, "type") for tool in state.tools),
            *(fv(tool, "volume_type") for tool in state.tools),
            fv(state.bed, "type"),
            fv(state, "mms_layout", state.mms_layout),
        ),
    )
    yield (
        NotificationMsg,
        (
            fv(
                state.notifications,
                "notifications",
                state.notifications.notifications.values(),
            ),
        ),
    )
    yield (
        PeripheralMsg,
        (fv(state.peripherals, "entries", state.peripherals.entries.values()),),
    )


class _ScanClient(Client):
    """The consume every kind's versions came from a full scan, verbatim."""

    def consume(self) -> list:
        self.last_msg_id = self.msg_id
        self._deferred = False
        self._release_held()

        msg_kinds = {}

        for msg_kind, candidates in _reference_message_changes(self.printer):
            if versions := tuple(v for v in candidates if v is not None):
                msg_kinds[msg_kind] = (min(versions), max(versions))

        is_pending = self.printer.config.is_pending()

        msgs = []

        for msg_kind, (lowest, highest) in sorted(
            msg_kinds.items(), key=lambda item: item[1][0]
        ):
            if is_pending and not msg_kind.msg_type().when_pending():
                continue

            data = dict(msg_kind.build(self.printer))

            if not data:
                continue

            msg = msg_kind(data)

            dispatch_mode = msg.dispatch_mode(self.printer)
            if dispatch_mode != DispatchMode.DISPATCH:
                if dispatch_mode == DispatchMode.RATELIMIT:
                    self._deferred = True
                continue

            msgs.append(msg)
            if self._settle_on_write:
                self._hold(msg, highest)
            else:
                msg.reset_changes(self.printer, v=highest)
                self._sent_kinds.add(msg_kind)

        if self._edges_first:
            self._edges_first = False
            msgs.sort(key=lambda m: not m.carries_edge())  # stable

        return msgs


def _reference_kinds(state: PrinterState) -> Dict[Type[ClientMsg], Tuple[int, int]]:
    """The kinds the scan finds changed, with (lowest, highest), in its order."""
    kinds = {}
    for kind, candidates in _reference_message_changes(state):
        if versions := tuple(v for v in candidates if v is not None):
            kinds[kind] = (min(versions), max(versions))
    return kinds


class _FastClient(Client):
    """The client under test, keeping what its consume read as changed."""

    read_kinds: Optional[Dict[Type[ClientMsg], Tuple[int, int]]] = None

    def _changed_kinds(self) -> Dict[Type[ClientMsg], Tuple[int, int]]:
        self.read_kinds = super()._changed_kinds()
        return self.read_kinds


# -- observation ---------------------------------------------------------------


def _markers(model: ReactiveModel, path: str = "printer") -> Dict[str, Dict]:
    """Every change marker in the tree, by position."""
    out = {path: dict(model.model_self_changed_fields)}
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, ReactiveModel):
            out.update(_markers(value, f"{path}.{name}"))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, ReactiveModel):
                    out.update(_markers(item, f"{path}.{name}[{i}]"))
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, ReactiveModel):
                    out.update(_markers(item, f"{path}.{name}[{key}]"))
    return out


def _models(model: ReactiveModel) -> Iterable[ReactiveModel]:
    yield model
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, ReactiveModel):
            yield from _models(value)
        elif isinstance(value, (list, tuple, dict)):
            items = value.values() if isinstance(value, dict) else value
            for item in items:
                if isinstance(item, ReactiveModel):
                    yield from _models(item)


def _observe(client: Client) -> tuple:
    return (
        client.msg_id,
        client.last_msg_id,
        client.has_changes,
        client.has_deferred,
        client.printer.current_job_id,
        _markers(client.printer),
    )


def _wire(msgs: List[ClientMsg]) -> list:
    return [(type(m), list(m.data.items())) for m in msgs]


def _spec(state: PrinterState) -> list:
    changes = client_module._message_changes(state)
    return [(kind, tuple(versions)) for kind, versions in changes]


def _reference_spec(state: PrinterState) -> list:
    changes = _reference_message_changes(state)
    return [(kind, tuple(versions)) for kind, versions in changes]


# -- the random walk -----------------------------------------------------------

_Apply = Callable[[Client], None]
_ISSUED = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _temperature(rng: random.Random) -> _Apply:
    where = rng.choice(["bed", "chamber", 0, 1, 2])
    field = rng.choice(["actual", "target"])
    value = rng.choice([None, 20.0, 20.4, 60.0, 200.0, 215.6])

    def apply(c: Client) -> None:
        if isinstance(where, int):
            tool = c.printer.tool(where)
            if tool is None:
                return
            model = tool.temperature
        else:
            model = getattr(c.printer, where).temperature
        setattr(model, field, value)

    return apply


def _tool_field(rng: random.Random) -> _Apply:
    nozzle = rng.randrange(3)
    field, value = rng.choice(
        [
            ("active_material", rng.choice([None, 0, 1, 2])),
            ("size", rng.choice([None, 0.4, 0.6])),
            ("type", rng.choice([None, *NozzleType])),
            ("volume_type", rng.choice([None, *VolumeType])),
        ]
    )

    def apply(c: Client) -> None:
        tool = c.printer.tool(nozzle)
        if tool is not None:
            setattr(tool, field, value)

    return apply


def _material(rng: random.Random) -> _Apply:
    nozzle, ext = rng.randrange(3), rng.randrange(6)
    field, value = rng.choice(
        [
            ("type", rng.choice([None, "PLA", "PETG", 3])),
            ("color", rng.choice([None, "Red", "Blue"])),
            ("hex", rng.choice([None, "#FF0000", "#0000FF"])),
            ("raw", rng.choice([None, {"k": 1}])),
        ]
    )

    def apply(c: Client) -> None:
        entry = c.printer.material(nozzle, ext)
        if entry is not None:
            setattr(entry, field, value)

    return apply


def _virtual_spool(rng: random.Random) -> _Apply:
    nozzle = rng.randrange(2)

    def apply(c: Client) -> None:
        # As a driver adds the external spool: bound, then appended.
        tool = c.printer.tool(nozzle)
        if tool is None or any(m.ext == 256 for m in tool.materials):
            return
        entry = MaterialEntry(nozzle=nozzle, ext=256)
        entry.provide_context(tool)
        tool.materials.append(entry)
        entry.type = "PLA"

    return apply


def _plain_field(rng: random.Random) -> _Apply:
    path, field, value = rng.choice(
        [
            ("", "status", rng.choice([None, *PrinterStatus])),
            ("bed", "type", rng.choice([None, *BedType])),
            ("job_info", "progress", rng.choice([None, 1.0, 50.4, 99.0])),
            ("job_info", "layer", rng.choice([None, 1, 2])),
            ("job_info", "time", rng.choice([None, 10.0])),
            ("job_info", "filename", rng.choice([None, "a.gcode", "b.gcode"])),
            ("job_info", "started", rng.choice([True, False])),
            ("job_info", "finished", rng.choice([True, False])),
            ("job_info", "cancelled", True),
            ("job_info", "failed", True),
            ("job_info", "reprint", rng.choice([None, 1])),
            ("job_info", "skipped_objects", rng.choice([None, [1, "b"]])),
            ("file_progress", "state", rng.choice([None, *FileProgressStateEnum])),
            ("file_progress", "percent", rng.choice([0.0, 10.0, 55.5])),
            ("file_progress", "message", rng.choice([None, "boom"])),
            ("info", "ui", rng.choice([None, "ui", "ui2"])),
            ("info", "hostname", rng.choice([None, "host"])),
            ("firmware", "name", rng.choice([None, "fw", "fw2"])),
            ("firmware", "version", rng.choice([None, "1.0"])),
            ("firmware_warning", "severity", rng.choice([None, "high"])),
            ("cpu_info", "usage", rng.choice([None, 1.0, 2.0])),
            ("cpu_info", "temp", rng.choice([None, 40.0])),
            ("psu_info", "on", rng.choice([True, False])),
            ("filament_sensor", "state", rng.choice([None, *FilamentSensorEnum])),
            ("webcam_info", "connected", rng.choice([True, False])),
            ("webcam_settings", "flipH", rng.choice([True, False])),
            ("webcam_settings", "rotate90", rng.choice([True, False])),
            ("latency", "ping", rng.choice([None, 1.0, 2.0])),
            ("latency", "pong", rng.choice([None, 1.5, 2.5])),
            ("ambient_temperature", "ambient", rng.choice([0, 21, 22])),
        ]
    )

    def apply(c: Client) -> None:
        model = getattr(c.printer, path) if path else c.printer
        setattr(model, field, value)

    return apply


def _set_changed(rng: random.Random) -> _Apply:
    # Includes fields no message kind reads ("nozzle", "intervals", "ping").
    path, field = rng.choice(
        [
            ("", "status"),
            ("", "mms_layout"),
            ("", "intervals"),
            ("", "bed"),
            ("bed", "type"),
            ("bed", "temperature"),
            ("tool", "nozzle"),
            ("tool", "materials"),
            ("tool", "active_material"),
            ("info", "sp_version"),
            ("firmware", "name"),
            ("latency", "ping"),
            ("webcam_info", "connected"),
            ("peripherals", "entries"),
            ("notifications", "notifications"),
        ]
    )

    def apply(c: Client) -> None:
        if path == "tool":
            model = c.printer.tools[-1]
        else:
            model = getattr(c.printer, path) if path else c.printer
        model.model_set_changed(field)

    return apply


def _peripheral(rng: random.Random) -> _Apply:
    pid = rng.choice(["fan:0", "light:0", "door:0"])
    value = rng.choice([None, 0, 50, True])
    available = rng.choice([None, True, False])

    def apply(c: Client) -> None:
        c.printer.peripherals.update(pid, value=value, available=available)

    return apply


def _notification(rng: random.Random) -> _Apply:
    action = rng.choice(["new", "new", "resolve", "remove", "payload"])
    event_id = uuid.UUID(int=rng.getrandbits(128))
    pick = rng.randrange(1 << 16)
    title = rng.choice(["a", "b", None])

    def apply(c: Client) -> None:
        center = c.printer.notifications
        if action == "new":
            center.new(
                event_id=event_id,
                issued_at=_ISSUED,
                payload=NotificationEventPayload(title=title),
            )
            return
        ids = list(center.notifications)
        if not ids:
            return
        target = ids[pick % len(ids)]
        if action == "resolve":
            center.notifications[target].resolve(_ISSUED)
        elif action == "remove":
            center.remove(target)
        else:
            center.notifications[target].payload.title = title

    return apply


def _structure(rng: random.Random) -> _Apply:
    action = rng.choice(
        ["tools", "materials", "layout", "settings", "detach", "reattach", "replace"]
    )
    count = rng.randint(1, 3)
    nozzle = rng.randrange(3)
    layout = [
        (rng.randrange(2), rng.choice(list(MultiMaterialSolution)), rng.randint(1, 4))
        for _ in range(rng.randint(0, 3))
    ]

    def apply(c: Client) -> None:
        p = c.printer
        if action == "tools":
            p.tool_count = count
        elif action == "materials":
            tool = p.tool(nozzle)
            if tool is not None:
                tool.material_count = count + 1
        elif action == "layout":
            p.update_mms_layout(
                [
                    MaterialLayoutEntry(nozzle=n, mms=mms, chains=chains)
                    for n, mms, chains in layout
                ]
            )
        elif action == "settings":
            p.settings = PrinterSettings(has_psu=bool(count % 2))
        elif action == "detach":
            # A tool leaves the tree with whatever changes it carries...
            if len(p.tools) > 1:
                c.__dict__.setdefault("_detached", []).append(p.tools.pop())
        elif action == "reattach":
            # ...and comes back, bound again: those changes count again.
            detached = c.__dict__.get("_detached")
            if detached and len(p.tools) < 3:
                tool = detached.pop()
                p.tools.append(tool)
                tool.provide_context(p)
        else:
            # As PrinterClient.reset_printer_state: a new tree, bound.
            c.printer = PrinterState(config=c.config)
            c.printer.provide_context(weakref.ref(c))
            c.printer.tool_count = count

    return apply


def _lifecycle(rng: random.Random) -> _Apply:
    action = rng.choice(["common", "full", "job_id", "pending"])
    job_id = rng.choice([None, 7, 8])
    pending = rng.random() < 0.3

    def apply(c: Client) -> None:
        if action == "common":
            c.printer.mark_common_fields_as_changed()
        elif action == "full":
            c._mark_full_state()
        elif action == "job_id":
            c.printer.current_job_id = job_id
        else:
            c.config.in_setup = pending

    return apply


_EDITS = (
    (_temperature, 6),
    (_tool_field, 2),
    (_material, 2),
    (_virtual_spool, 1),
    (_plain_field, 6),
    (_set_changed, 2),
    (_peripheral, 2),
    (_notification, 2),
    (_structure, 1),
    (_lifecycle, 1),
)


def _send(rng: random.Random) -> Tuple[str, List[Optional[WriteOutcome]], int]:
    mode = rng.choice(["plain", "settle", "settle", "abort"])
    outcomes = [rng.choice([None, *WriteOutcome]) for _ in range(32)]
    return mode, outcomes, rng.randrange(4)


def _consume(c: Client, mode: str, outcomes, cut: int) -> List[ClientMsg]:
    if mode == "plain":
        return c.consume()
    with c.settling_on_write():
        msgs = c.consume()
    for i, msg in enumerate(msgs):
        if mode == "abort" and i >= cut:
            break  # the rest of the send never settles
        if outcomes[i] is not None:
            msg.record_write(outcomes[i])
        c.settle(msg)
    return msgs


def _pair() -> Tuple[_FastClient, Client]:
    clients = []
    for cls in (_FastClient, _ScanClient):
        c = cls(PrinterConfig.get_new(), context=ClientContext())
        c.config.id = 1
        c.config.in_setup = False
        c.printer.tool_count = 2
        clients.append(c)
    return clients[0], clients[1]


@pytest.mark.parametrize("seed", range(40))
def test_consume_matches_the_full_scan(seed, monkeypatch):
    # Rate limits read the clock: both clients read this one.
    clock = [10_000_000]
    monkeypatch.setattr(Intervals, "now", staticmethod(lambda: clock[0]))

    rng = random.Random(seed)
    fast, scan = _pair()
    edits = [edit for edit, _ in _EDITS]
    weights = [weight for _, weight in _EDITS]

    for step in range(250):
        roll = rng.random()
        if roll < 0.2:
            mode, outcomes, cut = _send(rng)
            assert _spec(fast.printer) == _reference_spec(fast.printer), step
            for model in _models(fast.printer):
                changes = model.model_recursive_changeset.values()
                assert model.model_latest_change() == max(changes, default=None)
            want_kinds = _reference_kinds(fast.printer)
            got = _consume(fast, mode, outcomes, cut)
            want = _consume(scan, mode, outcomes, cut)
            # What consume read as changed: the scan's kinds, versions, order.
            assert list(fast.read_kinds.items()) == list(want_kinds.items())
            assert _wire(got) == _wire(want), (seed, step, mode)
        elif roll < 0.27:
            clock[0] += rng.choice([0, 100, 1_000, 3_000, 6_000, 40_000])
        else:
            apply = rng.choices(edits, weights)[0](rng)
            apply(fast)
            apply(scan)

        assert _observe(fast) == _observe(scan), (seed, step)
