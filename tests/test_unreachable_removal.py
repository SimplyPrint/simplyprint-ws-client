"""A printer whose device link drops, on a multi-printer hub: when it leaves
SimplyPrint (``remove_connection``) and how it comes back (``add_connection``).

Driven through the scheduler's own pass (``_schedule_client``) with a fake
hub manager, so every add and remove the printer would send is recorded, on
a fake clock for the removal hold (``PrinterClient.unreachable_grace``).
"""

import asyncio
import time
from datetime import datetime
from typing import List

import pytest

from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.core import client as client_module
from simplyprint_ws_client.core.client import ClientState, ClientStateChangeEvent
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.manager import ClientList
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionOutgoingEvent,
)
from simplyprint_ws_client.core.protocol.models import ClientMsgType
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.core.state import PrinterStatus
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


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(client_module, "_clock", fake)
    return fake


GRACE = PrinterClient.unreachable_grace


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
async def test_a_printer_whose_link_drops_is_removed_once_the_grace_ends(
    scheduler, clock
):
    printer = await _added_printer(scheduler)

    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    await _pass(scheduler, printer)
    # Within the grace: still added, nothing sent, still on its hub.
    assert printer.active is False
    assert printer.sent_types() == []
    assert printer.state == ClientState.CONNECTED
    assert scheduler.manager.is_allocated(printer)

    clock.now += GRACE
    await _pass(scheduler, printer)

    assert printer.sent_types() == [ClientMsgType.REMOVE_CONNECTION]
    assert printer.state == ClientState.PENDING_REMOVED


@pytest.mark.asyncio
async def test_a_removed_printer_is_deallocated_then_added_again_once_back(
    scheduler, clock
):
    printer = await _added_printer(scheduler)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    clock.now += GRACE
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


@pytest.mark.asyncio
async def test_a_link_back_within_the_grace_skips_the_remove_and_the_re_add(
    scheduler, clock
):
    printer = await _added_printer(scheduler)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    await _pass(scheduler, printer)
    clock.now += GRACE - 1

    await printer.link.set_reachability(DeviceReachability.UP)
    await _pass(scheduler, printer)
    # Long after the grace would have ended: the hold ended with the outage.
    clock.now += 10 * GRACE
    await _pass(scheduler, printer)

    assert printer.active is True
    assert printer.removal_held() is False
    assert printer.state == ClientState.CONNECTED
    sent = printer.sent_types()
    assert ClientMsgType.REMOVE_CONNECTION not in sent
    assert ClientMsgType.ADD_CONNECTION not in sent
    assert (scheduler.manager.allocate_calls, scheduler.manager.deallocate_calls) == (
        0,
        0,
    )
    assert printer.halts == 0


@pytest.mark.asyncio
async def test_changes_during_the_grace_go_out_once_the_link_is_back(
    scheduler, clock
):
    printer = await _added_printer(scheduler)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    printer.printer.status = PrinterStatus.PAUSED
    await _pass(scheduler, printer)
    assert printer.sent_types() == []  # inactive: nothing is sent meanwhile

    await printer.link.set_reachability(DeviceReachability.UP)
    await _pass(scheduler, printer)

    sent = printer.sent_types()
    assert ClientMsgType.STATUS in sent
    assert ClientMsgType.ADD_CONNECTION not in sent


@pytest.mark.asyncio
async def test_a_second_link_going_down_does_not_extend_the_grace(scheduler, clock):
    printer = await _added_printer(scheduler)
    other = printer.attach_driver(_Link(printer, name="other"))
    await other.set_reachability(DeviceReachability.UP)

    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    clock.now += GRACE - 1
    await other.set_reachability(DeviceReachability.DOWN, reason="gone too")
    clock.now += 1
    await _pass(scheduler, printer)

    assert printer.sent_types() == [ClientMsgType.REMOVE_CONNECTION]


@pytest.mark.asyncio
async def test_a_printer_being_deleted_is_not_held(scheduler, clock):
    printer = await _added_printer(scheduler)
    scheduler.client_list.add(printer)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    assert printer.removal_held()

    scheduler.remove(printer)
    await _pass(scheduler, printer)

    assert printer.sent_types() == [ClientMsgType.REMOVE_CONNECTION]


@pytest.mark.asyncio
async def test_a_held_printer_is_not_scheduled_on_every_wake(scheduler, clock):
    printer = await _added_printer(scheduler)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")
    now = datetime.now()
    scheduler._last_ticked[printer.unique_id] = now
    printer.last_msg_id = printer.msg_id

    assert not scheduler._should_schedule_client(printer, now)
    clock.now += GRACE
    assert scheduler._should_schedule_client(printer, now)


@pytest.mark.asyncio
async def test_losing_the_hub_during_the_grace_resends_the_full_state(
    scheduler, clock
):
    from simplyprint_ws_client.core.protocol.events import (
        SimplyPrintConnectionLostEvent,
    )

    printer = await _added_printer(scheduler)
    await printer.link.set_reachability(DeviceReachability.DOWN, reason="gone")

    await printer.event_bus.emit(SimplyPrintConnectionLostEvent(printer.v))

    # Still added when the hub went: the server may have missed what it was
    # sent, so the next `connected` sends everything again.
    assert printer._resend_state is True


@pytest.mark.asyncio
async def test_the_end_of_the_grace_wakes_the_scheduler(monkeypatch):
    monkeypatch.setattr(client_module, "_clock", time.monotonic)
    printer = _Printer()
    woken = asyncio.Event()
    printer.event_bus.on(ClientStateChangeEvent, lambda *_: woken.set())

    printer.hold_removal(0.05)
    woken.clear()
    await asyncio.wait_for(woken.wait(), timeout=2.0)

    assert printer.removal_held() is False


# -- a printer waiting on its add or remove is not run on every wake ------------


def _idle(scheduler: Scheduler, printer: _Printer) -> datetime:
    """Ticked just now and nothing changed: only the connection state can
    get the printer scheduled."""
    now = datetime.now()
    scheduler._last_ticked[printer.unique_id] = now
    printer.last_msg_id = printer.msg_id
    return now


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [ClientState.CONNECTING, ClientState.PENDING_ADDED, ClientState.PENDING_REMOVED],
)
async def test_a_printer_waiting_on_its_hub_or_the_server_is_not_scheduled(
    scheduler, state
):
    printer = await _added_printer(scheduler)
    printer.state = state

    assert not scheduler._should_schedule_client(printer, _idle(scheduler, printer))


@pytest.mark.asyncio
async def test_a_printer_whose_add_is_held_is_scheduled_once_the_hold_ends(scheduler):
    from datetime import timedelta

    printer = await _added_printer(scheduler)
    printer.state = ClientState.NOT_CONNECTED
    printer._pending_action_ts = datetime.now()
    printer._pending_action_delay = timedelta(seconds=30)
    assert not scheduler._should_schedule_client(printer, _idle(scheduler, printer))

    printer._pending_action_ts -= timedelta(seconds=31)
    assert scheduler._should_schedule_client(printer, _idle(scheduler, printer))


@pytest.mark.asyncio
async def test_an_unplaced_active_printer_is_scheduled(scheduler):
    printer = await _added_printer(scheduler)
    scheduler.manager.allocated.clear()
    printer.state = ClientState.CONNECTING

    assert scheduler._should_schedule_client(printer, _idle(scheduler, printer))


@pytest.mark.asyncio
async def test_an_add_or_remove_hold_wakes_the_scheduler_when_it_ends():
    printer = _Printer()
    woken = asyncio.Event()
    printer.event_bus.on(ClientStateChangeEvent, lambda *_: woken.set())
    printer._pending_action_backoff.delay = lambda: 0.05

    printer._do_pending()
    woken.clear()
    await asyncio.wait_for(woken.wait(), timeout=2.0)

    assert printer.pending_action_ready()
