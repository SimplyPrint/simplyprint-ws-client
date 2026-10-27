"""A batch of state changes wakes the scheduler once."""

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.client import ClientStateChangeEvent


def _client():
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    signals = []
    client.event_bus.on(ClientStateChangeEvent, lambda *a: signals.append(1))
    return client, signals


def test_changes_inside_a_batch_signal_once_at_its_end():
    client, signals = _client()
    with client.batched_signals():
        client.printer.bed.temperature.actual = 60.0
        client.printer.bed.temperature.target = 65.0
        with client.batched_signals():  # nested: still one, at the outermost end
            client.printer.tool0.temperature.actual = 200.0
        assert signals == []
    assert signals == [1]


def test_a_batch_without_changes_signals_nothing():
    client, signals = _client()
    with client.batched_signals():
        pass
    assert signals == []


def test_outside_a_batch_each_change_signals():
    client, signals = _client()
    client.printer.bed.temperature.actual = 60.0
    client.printer.bed.temperature.target = 65.0
    assert len(signals) == 2
