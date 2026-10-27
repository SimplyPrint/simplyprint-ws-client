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
