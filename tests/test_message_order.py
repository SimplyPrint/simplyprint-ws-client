from simplyprint_ws_client import (
    Client,
    PeripheralMsg,
    PrinterStatus,
    JobInfoMsg,
    StateChangeMsg,
)
from tests.test_intervals import TimeControlledIntervals


def test_message_order_simple(client: Client):
    msgs, _ = client.consume()
    assert len(msgs) == 0

    client.printer.job_info.started = True
    client.printer.status = PrinterStatus.PRINTING

    # Assert the order of messages
    msgs, _ = client.consume()
    assert len(msgs) == 2
    assert msgs[0].__class__ == JobInfoMsg
    assert msgs[1].__class__ == StateChangeMsg


def test_peripheral_messages_emit_individually_in_order(client: Client):
    client.printer.fan("part_cooling").set(50, available=True, updated=1712345678)
    client.printer.light("chamber_light").set(False, available=True, updated=1712345679)

    msgs, _ = client.consume()

    assert len(msgs) == 1
    assert msgs[0].__class__ == PeripheralMsg
    assert msgs[0].data == {
        "fan:part_cooling": {"v": 50, "a": True, "u": 1712345678},
        "light:chamber_light": {"v": False, "a": True, "u": 1712345679},
    }

    msgs, _ = client.consume()
    assert msgs == []


def test_peripheral_messages_are_ratelimited(client: Client):
    intervals = client.printer.intervals = TimeControlledIntervals()
    intervals.set_time(0)

    client.printer.fan("part_cooling").set(50, available=True, updated=1712345678)

    msgs, _ = client.consume()
    assert len(msgs) == 1
    assert msgs[0].__class__ == PeripheralMsg

    client.printer.light("chamber_light").set(False, available=True, updated=1712345679)

    msgs, _ = client.consume()
    assert msgs == []

    intervals.step_time(intervals.peripheral)

    msgs, _ = client.consume()
    assert len(msgs) == 1
    assert msgs[0].__class__ == PeripheralMsg
    assert msgs[0].data == {
        "light:chamber_light": {"v": False, "a": True, "u": 1712345679},
    }


def test_peripheral_helpers_accept_numeric_names(client: Client):
    client.printer.fan(0).set(50, available=True, updated=1712345680)

    msgs, _ = client.consume()

    assert len(msgs) == 1
    assert msgs[0].__class__ == PeripheralMsg
    assert msgs[0].data == {
        "fan:0": {"v": 50, "a": True, "u": 1712345680},
    }
