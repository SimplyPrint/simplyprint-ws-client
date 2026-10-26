"""What ``Client.consume`` costs after one temperature change.

The common case on a busy farm: a device report moves one tool temperature and
the scheduler consumes. The printer here is a full one: 2 tools, 4 AMS units
(16 slots, each with a material), the mms layout, 5 peripherals, 2 open
notifications, firmware and host info. ``consume`` reads only the kinds a
change can have reached; ``_ScanClient`` (the equivalence test's verbatim
copy of the consume before that) scans every kind's branches of the whole
tree. Both run the same edits, interleaved, timed in thread CPU.

Printed for a human to compare; the assertion is only that the change-driven
consume is not slower than the full scan.
"""

import datetime
import time
import uuid
from typing import List, Type

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.protocol.messages import TemperatureMsg
from simplyprint_ws_client.core.state.models import (
    MultiMaterialSolution,
    PrinterStatus,
)
from simplyprint_ws_client.core.state.notifications import NotificationEventPayload
from simplyprint_ws_client.core.state.printer import MaterialLayoutEntry
from tests.test_consume_scan_equivalence import _ScanClient

pytestmark = pytest.mark.performance

_ISSUED = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _full_printer(cls: Type[Client]) -> Client:
    client = cls(PrinterConfig.get_new(), context=ClientContext())
    client.config.id = 1
    client.config.in_setup = False
    p = client.printer

    # Every temperature change dispatches: the measured path builds, sends
    # and resets (no rate limit deferral).
    p.intervals.temps = 0
    p.intervals.temps_target = 0

    p.status = PrinterStatus.OPERATIONAL
    p.set_info("bench", "1.0")
    p.info.hostname = "farm-01"
    p.firmware.name = "fw"
    p.firmware.version = "1.2.3"
    p.tool_count = 2
    p.update_mms_layout(
        [
            MaterialLayoutEntry(nozzle=nozzle, mms=MultiMaterialSolution.BAMBU_AMS)
            for nozzle in (0, 0, 1, 1)
        ]
    )
    for tool in p.tools:
        tool.temperature.actual = 200.0
        tool.temperature.target = 210.0
        for i, material in enumerate(tool.materials):
            material.type = "PLA"
            material.color = "Red"
            material.hex = f"#FF00{i:02X}"
    p.bed.temperature.actual = 60.0
    p.chamber.temperature.actual = 30.0
    for pid in ("fan:0", "fan:1", "light:0", "door:0", "power:psu"):
        p.peripherals.update(pid, value=1, available=True)
    for n in range(2):
        p.notifications.new(
            event_id=uuid.UUID(int=n + 1),
            issued_at=_ISSUED,
            payload=NotificationEventPayload(title=f"open {n}"),
        )

    client.consume()  # drain the setup
    assert sum(len(t.materials) for t in p.tools) == 16
    return client


def test_consume_after_one_temperature_change():
    fast, scan = _full_printer(Client), _full_printer(_ScanClient)
    for client in (fast, scan):
        assert not client.consume()  # nothing left pending

    iterations = 3_000
    spent: List[int] = [0, 0]
    for i in range(iterations):
        actual = 201.0 - (i % 2)  # differs from the last value every time
        for side, client in enumerate((fast, scan)):
            client.printer.tools[0].temperature.actual = actual
            start = time.thread_time_ns()
            msgs = client.consume()
            spent[side] += time.thread_time_ns() - start
            assert [type(m) for m in msgs] == [TemperatureMsg]

    fast_us, scan_us = (ns / iterations / 1_000 for ns in spent)
    print(
        f"\nconsume after one temperature change: {fast_us:.1f} us"
        f" (full scan: {scan_us:.1f} us, {scan_us / fast_us:.1f}x)"
    )
    assert fast_us < scan_us, (fast_us, scan_us)
