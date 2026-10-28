"""model_write leaves a model exactly as the same assignment does."""

import random

from simplyprint_ws_client import Client, ClientContext, PrinterConfig


def _pair():
    a = Client(PrinterConfig.get_new(), context=ClientContext())
    b = Client(PrinterConfig.get_new(), context=ClientContext())
    return a, b


def _state(client):
    p = client.printer
    models = [p.bed.temperature, p.chamber.temperature, p.tool0.temperature, p.job_info, p.firmware]
    return [
        (m.model_dump(), sorted(m.model_self_changed_fields), sorted(m.__pydantic_fields_set__))
        for m in models
    ], client.msg_id


def test_random_writes_match_assignments():
    rng = random.Random(7)
    a, b = _pair()
    targets = [
        (lambda p: p.bed.temperature, "actual", [None, 60.0, 61.5, 60]),
        (lambda p: p.bed.temperature, "target", [None, 0.0, 65.0]),
        (lambda p: p.tool0.temperature, "actual", [200.0, 201.0, None]),
        (lambda p: p.job_info, "progress", [0.0, 10.0, 10.0, 55.5]),
        (lambda p: p.job_info, "filename", ["a.gcode", "b.gcode", None]),
        (lambda p: p.job_info, "started", [True, False]),
        (lambda p: p.job_info, "finished", [True, False]),
        (lambda p: p.firmware, "name", ["fw", "fw2", None]),
    ]
    for _ in range(2000):
        get, name, values = rng.choice(targets)
        value = rng.choice(values)
        setattr(get(a.printer), name, value)
        changed_before = b.msg_id
        changed = get(b.printer).model_write(name, value)
        assert changed == (b.msg_id != changed_before)
        assert _state(a) == _state(b)


def test_model_write_many_matches_assignments_in_order():
    rng = random.Random(11)
    a, b = _pair()
    fields = {
        "ui": ["Device", "Other", None],
        "api": ["Client", None],
        "local_ip": ["192.0.2.1", "192.0.2.2", None],
        "is_ethernet": [True, False, None],
        "core_count": [4, 8, None],
        "sp_version": ["2.0.0", "3.0.0"],
    }
    for _ in range(500):
        items = [(name, rng.choice(values)) for name, values in rng.sample(list(fields.items()), 4)]
        for name, value in items:
            setattr(a.printer.info, name, value)
        before = b.msg_id
        changed = b.printer.info.model_write_many(items)
        assert changed == b.msg_id - before
        assert (
            a.printer.info.model_dump(),
            a.printer.info.model_self_changed_fields,
            sorted(a.printer.info.__pydantic_fields_set__),
            a.msg_id,
        ) == (
            b.printer.info.model_dump(),
            b.printer.info.model_self_changed_fields,
            sorted(b.printer.info.__pydantic_fields_set__),
            b.msg_id,
        )


def test_model_write_many_signals_once(monkeypatch):
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    signals = []
    monkeypatch.setattr(client, "signal", lambda: signals.append(1))
    changed = client.printer.info.model_write_many(
        [("ui", "Device"), ("api", "Client"), ("ui", "Device"), ("os", "Linux")]
    )
    assert changed == 3
    assert signals == [1]
    assert client.printer.info.model_write_many([("ui", "Device")]) == 0
    assert signals == [1]


def _fields(m):
    state = {k: v for k, v in m.__dict__.items() if not k.startswith("_Synchronized")}
    return (
        list(state.items()),
        sorted(m.__pydantic_fields_set__),
        m.__pydantic_extra__,
        m.__pydantic_private__,
        m.model_self_changed_fields,
    )


def test_model_fresh_is_the_validated_model():
    from simplyprint_ws_client.core.state.printer import PeripheralState

    ctx = lambda: None  # noqa: E731
    for values in ({"id": "fan:a"}, {"id": "fan:b", "value": 50, "available": True}):
        fresh = PeripheralState.model_fresh(ctx, values)
        built = PeripheralState(ctx, id=values["id"])
        for name, value in values.items():
            built.model_write(name, value)  # unbound: nothing marked
        assert _fields(fresh) == _fields(built)
        assert fresh.ctx is ctx
        with fresh:  # its own lock
            pass


def test_model_fresh_builds_other_models_the_validated_way():
    from simplyprint_ws_client.core.state.printer import JobInfoState, PeripheralState

    # Validators or a post-init of its own: not from the template.
    assert JobInfoState.model_fresh(lambda: None, {}).model_dump() == JobInfoState().model_dump()
    # A required field missing: validation says so, as the constructor does.
    try:
        PeripheralState.model_fresh(lambda: None, {})
    except ValueError:
        pass
    else:
        raise AssertionError("a missing id must fail")


def _old_add(peripherals, peripheral_id, **fields):
    """The entry as it was built before the template: constructed, bound,
    attached, marked, then written field by field."""
    from simplyprint_ws_client.core.state.printer import PeripheralState

    entry = PeripheralState(peripherals.ctx, id=peripheral_id)
    if (live := peripherals.ctx()) is not None:
        live.state_attached(entry)
    peripherals.entries[peripheral_id] = entry
    peripherals.model_set_changed("entries")
    entry.model_set_changed("id")
    for name, value in fields.items():
        entry.model_write(name, value)
    return entry


def test_new_peripheral_entry_matches_construct_then_write():
    from simplyprint_ws_client.core.protocol.messages import PeripheralMsg

    rng = random.Random(5)
    values = {"value": [None, 0, 50, True, False], "available": [None, True, False], "updated": [None, 0, 1700000000]}
    for i in range(300):
        a, b = _pair()
        fields = {name: rng.choice(options) for name, options in values.items() if rng.random() < 0.7}
        pid = f"fan:{i}"
        _old_add(a.printer.peripherals, pid, **fields)
        b.printer.peripherals.update(pid, **fields)
        old, new = a.printer.peripherals.entries[pid], b.printer.peripherals.entries[pid]
        assert _fields(old) == _fields(new)
        assert a.printer.peripherals.model_self_changed_fields == b.printer.peripherals.model_self_changed_fields
        assert (a.msg_id, a._touched_kinds) == (b.msg_id, b._touched_kinds)
        assert dict(PeripheralMsg.build(a.printer)) == dict(PeripheralMsg.build(b.printer))


def _typed(m):
    """Fields with their types (an int and a float compare equal), marks, fields set."""
    return (
        [(k, type(v), v) for k, v in m.__dict__.items() if not k.startswith("_")],
        sorted(m.model_self_changed_fields.items()),
        sorted(m.__pydantic_fields_set__),
    )


# Values a projection or anyone else may write, incl. ones validation turns
# into something else (an int into a float), ones it refuses and the ends of
# a float's exact int range.
_ANY = [None, 0, 1, -3, 2**53, 2**53 + 1, 10**30, 10**400, 0.0, 1.5, 1.0, float("nan"),
        float("inf"), True, False, "7", "x", [1], b"1"]


def _assign(model, name, value):
    try:
        setattr(model, name, value)
    except Exception as error:  # pydantic's ValidationError
        return type(error)


def _write(model, name, value):
    try:
        model.model_write(name, value)
    except Exception as error:
        return type(error)


def test_validated_fields_written_as_assigned():
    """A validating model (job info): model_write stores what validation would,
    refuses what it refuses, and marks the same."""
    rng = random.Random(23)
    a, b = _pair()
    names = list(type(a.printer.job_info).model_fields)
    for _ in range(3000):
        name, value = rng.choice(names), rng.choice(_ANY)
        assert _assign(a.printer.job_info, name, value) == _write(b.printer.job_info, name, value)
        assert (_typed(a.printer.job_info), a.msg_id) == (_typed(b.printer.job_info), b.msg_id), (name, value)


def test_own_setattr_runs_for_the_fields_it_acts_on():
    """FileProgressState resets the percent when the state moves: a write of
    the state through model_write still does."""
    from simplyprint_ws_client.core.state.models import FileProgressStateEnum

    a, b = _pair()
    for model in (a.printer.file_progress, b.printer.file_progress):
        model.percent = 40.0
    a.printer.file_progress.state = FileProgressStateEnum.READY
    b.printer.file_progress.model_write("state", FileProgressStateEnum.READY)
    assert b.printer.file_progress.percent == 0.0
    assert (_typed(a.printer.file_progress), a.msg_id) == (_typed(b.printer.file_progress), b.msg_id)


def test_model_writer_is_model_write_many():
    """The generated writer leaves each model as model_write_many with the
    same names does: plain, validated and intercepted fields, any values."""
    rng = random.Random(29)
    a, b = _pair()
    targets = [
        (lambda p: p.peripherals, None),  # replaced below by an entry
        (lambda p: p.job_info, ["progress", "time", "layer", "filename", "started", "skipped_objects"]),
        (lambda p: p.bed.temperature, ["actual", "target"]),
        (lambda p: p.file_progress, ["state", "percent", "message"]),
        (lambda p: p.info, ["ui", "core_count", "is_ethernet"]),
    ]
    for client in (a, b):
        client.printer.peripherals.update("fan:x", value=1, available=True, updated=1)
    targets[0] = (lambda p: p.peripherals.entries["fan:x"], ["value", "available", "updated", "id"])
    from simplyprint_ws_client.core.state.models import FileProgressStateEnum

    values = _ANY + list(FileProgressStateEnum)
    for _ in range(2000):
        get, names = rng.choice(targets)
        chosen = tuple(rng.sample(names, rng.randint(1, len(names))))
        items = [(name, rng.choice(values)) for name in chosen]
        try:
            expected = get(a.printer).model_write_many(items)
        except Exception as error:
            expected = type(error)
        writer = type(get(b.printer)).model_writer(*chosen)
        try:
            got = writer(get(b.printer), *(v for _, v in items))
        except Exception as error:
            got = type(error)
        assert expected == got, items
        assert (_typed(get(a.printer)), a.msg_id) == (_typed(get(b.printer)), b.msg_id), items


def test_model_writer_is_built_once_and_checks_names():
    from simplyprint_ws_client.core.state.printer import TemperatureState

    assert TemperatureState.model_writer("actual", "target") is TemperatureState.model_writer("actual", "target")
    import pytest

    with pytest.raises(AttributeError):
        TemperatureState.model_writer("nope")
