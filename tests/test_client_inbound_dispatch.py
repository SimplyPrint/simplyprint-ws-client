"""Cloud messages retain FIFO routing but execute independently per message."""

import asyncio

import pytest

from simplyprint_ws_client import Client, PrinterConfig
from simplyprint_ws_client.core.protocol.events import SimplyPrintConnectionLostEvent
from simplyprint_ws_client.core.protocol.messages import (
    DemandMsg,
    WebcamSnapshotDemandData,
)
from simplyprint_ws_client.core.protocol.models import DemandMsgType, ServerMsgType
from simplyprint_ws_client.wire.events import Disconnected


def _client() -> Client:
    client = Client(PrinterConfig.get_new())
    client.use_running_loop()
    client.v = 0
    return client


def _snapshot(request_id: str) -> DemandMsg:
    return DemandMsg(
        type=ServerMsgType.DEMAND,
        data=WebcamSnapshotDemandData(id=request_id),
    )


@pytest.mark.asyncio
async def test_slow_message_does_not_block_later_message():
    client = _client()
    slow_entered = asyncio.Event()
    release_slow = asyncio.Event()
    fast_finished = asyncio.Event()

    async def handler(data: WebcamSnapshotDemandData) -> None:
        if data.id == "slow":
            slow_entered.set()
            await release_slow.wait()
        else:
            fast_finished.set()

    client.event_bus.on(DemandMsgType.WEBCAM_SNAPSHOT, handler, priority=10)

    client._on_connection_incoming(_snapshot("slow"), 0)
    await asyncio.wait_for(slow_entered.wait(), 1.0)
    client._on_connection_incoming(_snapshot("fast"), 0)

    await asyncio.wait_for(fast_finished.wait(), 0.2)
    assert any(not task.done() for task in client._inbound_tasks)

    release_slow.set()
    await client.shutdown_inbound_dispatch()
    assert not client._inbound_tasks


@pytest.mark.asyncio
async def test_handler_failure_is_observed_and_later_messages_continue(caplog):
    client = _client()
    handled = asyncio.Event()

    async def handler(data: WebcamSnapshotDemandData) -> None:
        if data.id == "bad":
            raise RuntimeError("broken handler")
        handled.set()

    client.event_bus.on(DemandMsgType.WEBCAM_SNAPSHOT, handler, priority=10)

    with caplog.at_level("ERROR"):
        client._on_connection_incoming(_snapshot("bad"), 0)
        client._on_connection_incoming(_snapshot("good"), 0)
        await asyncio.wait_for(handled.wait(), 1.0)
        await asyncio.sleep(0)

    assert any("inbound handler failed" in record.message for record in caplog.records)
    await client.shutdown_inbound_dispatch()


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_owned_message_tasks():
    client = _client()
    entered = asyncio.Event()

    async def blocked(_data: WebcamSnapshotDemandData) -> None:
        entered.set()
        await asyncio.Event().wait()

    client.event_bus.on(DemandMsgType.WEBCAM_SNAPSHOT, blocked, priority=10)
    client._on_connection_incoming(_snapshot("blocked"), 0)
    await asyncio.wait_for(entered.wait(), 1.0)

    task = next(iter(client._inbound_tasks))
    await client.shutdown_inbound_dispatch()

    assert task.cancelled()
    assert not client._inbound_tasks


@pytest.mark.asyncio
async def test_disconnect_advances_epoch_even_when_lost_listener_fails():
    # Use a real connection protocol so the assertion covers its lifecycle
    # EventBus and not a copied implementation.
    from simplyprint_ws_client.core.protocol.connection import SimplyPrintConnection

    connection = SimplyPrintConnection()

    async def fail(_event: SimplyPrintConnectionLostEvent) -> None:
        raise RuntimeError("lost listener failed")

    connection.event_bus.on(SimplyPrintConnectionLostEvent, fail)
    with pytest.raises(RuntimeError, match="lost listener failed"):
        await connection.protocol._on_disconnected(Disconnected(1))

    assert connection.protocol.v == 1
