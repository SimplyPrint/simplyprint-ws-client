"""``Client.shared_peripherals``: a sent unavailable peripheral entry is one
object across the printers that have it (same id, same second), bound to
none and read-only; a write takes the printer's own copy first, so what each
printer builds and sends is what it would without sharing (the consume walk
in ``test_consume_scan_equivalence`` holds that over random edits)."""

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.common.model.reactive import SHARED_MARKS
from simplyprint_ws_client.core.protocol.messages import PeripheralMsg


def _client(shared: bool = True) -> Client:
    c = Client(PrinterConfig.get_new(), context=ClientContext())
    c.config.id = 1
    c.config.in_setup = False
    c.shared_peripherals = shared
    return c


def _sent(c: Client) -> list:
    return [(type(m), m.data) for m in c.consume()]


def _first_report(c: Client) -> None:
    c.printer.peripherals.add_all(
        [("fan:model", 40, True), ("fan:aux", None, False), ("light:0", None, False)],
        1_700_000_000,
    )


def test_two_printers_hold_one_entry_once_it_is_sent():
    a, b, plain = _client(), _client(), _client(shared=False)
    for c in (a, b, plain):
        _first_report(c)
    assert _sent(a) == _sent(b) == _sent(plain)
    for pid in ("fan:aux", "light:0"):
        entry = a.printer.peripherals.entries[pid]
        assert entry is b.printer.peripherals.entries[pid]
        assert entry.model_self_changed_fields is SHARED_MARKS
        assert entry == plain.printer.peripherals.entries[pid]
    # One with a value is each printer's own.
    assert a.printer.peripherals.entries["fan:model"] is not (
        b.printer.peripherals.entries["fan:model"]
    )
    assert not plain.printer.peripherals.entries["fan:aux"].model_self_changed_fields


def test_a_write_takes_the_printers_own_copy():
    a, b, plain = _client(), _client(), _client(shared=False)
    for c in (a, b, plain):
        _first_report(c)
        _sent(c)
    for c in (a, plain):
        c.printer.peripherals.update("fan:aux", value=30, available=True, updated=1_700_000_005)
    assert _sent(a) == _sent(plain) == [(PeripheralMsg, {"fan:aux": {"v": 30, "a": True, "u": 1_700_000_005}})]
    assert a.printer.peripherals.entries["fan:aux"].value == 30
    assert b.printer.peripherals.entries["fan:aux"].value is None  # b's untouched
    assert b.printer.peripherals.entries["fan:aux"].model_self_changed_fields is SHARED_MARKS


def test_writing_the_same_values_keeps_it_shared():
    a = _client()
    _first_report(a)
    _sent(a)
    shared = a.printer.peripherals.entries["light:0"]
    assert a.printer.peripherals.update(
        "light:0", value=None, available=False, updated=1_700_000_000
    ) is shared
    assert a.printer.peripherals.entries["light:0"] is shared
    assert _sent(a) == []


def test_a_shared_entry_cannot_be_written_or_bound():
    a, b = _client(), _client()
    for c in (a, b):
        _first_report(c)
        _sent(c)
    shared = a.printer.peripherals.entries["fan:aux"]
    with pytest.raises(TypeError):
        shared.value = 10
    with pytest.raises(TypeError):
        shared.model_write("value", 10)
    with pytest.raises(TypeError):
        shared.model_set_changed("value")
    assert shared.value is None and not shared.model_self_changed_fields
    b.printer.provide_context(a.printer)  # a rebind leaves it bound to none
    assert b.printer.peripherals.entries["fan:aux"] is shared
    with pytest.raises(TypeError):
        shared.ctx()


def test_a_reconnect_resends_it_from_the_printers_own_copy():
    a, b, plain = _client(), _client(), _client(shared=False)
    for c in (a, b, plain):
        _first_report(c)
        _sent(c)
        c._mark_full_state()
    assert a.printer.peripherals.entries["fan:aux"].model_self_changed_fields  # own, marked
    assert _sent(a) == _sent(b) == _sent(plain)
    # Sent again: shared again.
    assert a.printer.peripherals.entries["fan:aux"] is b.printer.peripherals.entries["fan:aux"]
