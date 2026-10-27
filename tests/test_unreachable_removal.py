"""A printer whose device link drops, on a multi-printer hub: when it leaves
SimplyPrint (``remove_connection``) and how it comes back (``add_connection``).

Driven through the scheduler's own pass (``_schedule_client``) with a fake
hub manager, so every add and remove the printer would send is recorded.
"""

from datetime import datetime
from typing import List

import pytest

from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.core.client import ClientState
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.manager import ClientList
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionOutgoingEvent,
)
from simplyprint_ws_client.core.protocol.models import ClientMsgType
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.integration.client import PrinterClient
from simplyprint_ws_client.integration.drivers import (
    DeviceDriver,
    DeviceReachability,
)


class _Link(DeviceDriver):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Printer(PrinterClient[PrinterConfig]):
    def __init__(self) -> None:
        config = PrinterConfig.get_new()
        config.id = 1
        super().__init__(config, context=ClientContext())
        self.link = self.attach_driver(_Link(self, name="link"))
        self.sent: List[object] = []
        self.halts = 0
        self.event_bus.on(SimplyPrintConnectionOutgoingEvent, self._record)

    def _record(self, msg, *_args) -> None:
        self.sent.append(msg)

    def clear_camera_uri(self) -> None:
        pass

    async def halt(self) -> None:
        self.halts += 1

    def sent_types(self) -> List[ClientMsgType]:
        return [msg.type for msg in self.sent]


class _FakeManager:
    is_replaying = False

    def __init__(self) -> None:
        self.allocated = set()
        self.allocate_calls = 0
        self.deallocate_calls = 0

    def is_allocated(self, client) -> bool:
        return client.unique_id in self.allocated

    def is_misplaced(self, client) -> bool:
        return False

    async def allocate(self, client) -> None:
        self.allocate_calls += 1
        self.allocated.add(client.unique_id)

    async def deallocate(self, client) -> None:
        self.deallocate_calls += 1
        self.allocated.discard(client.unique_id)


@pytest.fixture
def scheduler() -> Scheduler:
    settings = ClientSettings(mode=ConnectionMode.MULTI, camera_workers=None)
    scheduler = Scheduler(client_list=ClientList(), settings=settings)
    scheduler.manager = _FakeManager()
    return scheduler


async def _added_printer(scheduler: Scheduler) -> _Printer:
    """A printer whose device answers, allocated and added on its hub."""
    printer = _Printer()
    printer.initialized = True
    await printer.link.set_reachability(DeviceReachability.UP)
    scheduler.manager.allocated.add(printer.unique_id)
    printer.state = ClientState.CONNECTED
    printer.last_msg_id = printer.msg_id  # nothing pending
    printer.sent.clear()
    return printer


async def _pass(scheduler: Scheduler, printer: _Printer) -> None:
    """One scheduler pass for the printer, its tick left out."""
    scheduler._last_ticked[printer.unique_id] = datetime.now()
    await scheduler._schedule_client(printer)


@pytest.mark.asyncio
async def test_a_printer_whose_link_drops_is_removed(scheduler):
    printer = await _added_printer(scheduler)

    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    await _pass(scheduler, printer)

    assert printer.active is False
    assert printer.sent_types() == [ClientMsgType.REMOVE_CONNECTION]
    assert printer.state == ClientState.PENDING_REMOVED


@pytest.mark.asyncio
async def test_a_removed_printer_is_deallocated_then_added_again_once_back(
    scheduler,
):
    printer = await _added_printer(scheduler)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    await _pass(scheduler, printer)
    printer.state = ClientState.NOT_CONNECTED  # the server's answer
    await _pass(scheduler, printer)
    assert scheduler.manager.deallocate_calls == 1
    assert printer.halts == 1

    await printer.link.set_reachability(DeviceReachability.UP)
    printer._pending_action_ts = datetime.min  # no add hold
    printer.state = ClientState.NOT_CONNECTED  # the hub's established event
    await _pass(scheduler, printer)

    assert scheduler.manager.allocate_calls == 1
    assert printer.sent_types() == [
        ClientMsgType.REMOVE_CONNECTION,
        ClientMsgType.ADD_CONNECTION,
    ]


@pytest.mark.asyncio
async def test_a_printer_made_inactive_directly_is_removed_at_once(scheduler):
    # What Scheduler.remove does for a printer being deleted.
    printer = await _added_printer(scheduler)

    printer.active = False
    await _pass(scheduler, printer)

    assert printer.sent_types() == [ClientMsgType.REMOVE_CONNECTION]
