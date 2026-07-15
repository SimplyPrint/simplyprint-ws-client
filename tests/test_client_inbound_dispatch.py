"""Cloud message handling stays on the protocol's single FIFO consumer."""

import asyncio

import pytest
from yarl import URL

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.protocol.events import SimplyPrintConnectionLostEvent
from simplyprint_ws_client.core.protocol.messages import (
    DemandMsg,
    WebcamSnapshotDemandData,
)
from simplyprint_ws_client.core.protocol.models import DemandMsgType, ServerMsgType
from simplyprint_ws_client.wire.events import Disconnected

_WS = URL("wss://ws.example")


def _client() -> Client:
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    client.use_running_loop()
    client.v = 0
    return client


def _snapshot(request_id: str) -> DemandMsg:
    return DemandMsg(
        type=ServerMsgType.DEMAND,
        data=WebcamSnapshotDemandData(id=request_id),
    )


@pytest.mark.asyncio
async def test_incoming_message_awaits_handler_completion():
    client = _client()
    slow_entered = asyncio.Event()
    release_slow = asyncio.Event()

    async def handler(data: WebcamSnapshotDemandData) -> None:
        slow_entered.set()
        await release_slow.wait()

    client.event_bus.on(DemandMsgType.WEBCAM_SNAPSHOT, handler, priority=10)

    slow = asyncio.create_task(client._on_connection_incoming(_snapshot("slow"), 0))
    await asyncio.wait_for(slow_entered.wait(), 1.0)
    assert not slow.done()

    release_slow.set()
    await slow


@pytest.mark.asyncio
async def test_handler_failure_propagates_to_protocol_consumer():
    client = _client()

    async def handler(data: WebcamSnapshotDemandData) -> None:
        raise RuntimeError(f"broken handler: {data.id}")

    client.event_bus.on(DemandMsgType.WEBCAM_SNAPSHOT, handler, priority=10)

    with pytest.raises(RuntimeError, match="broken handler: bad"):
        await client._on_connection_incoming(_snapshot("bad"), 0)


@pytest.mark.asyncio
async def test_disconnect_advances_epoch_even_when_lost_listener_fails():
    # Use a real connection protocol so the assertion covers its lifecycle
    # EventBus and not a copied implementation.
    from simplyprint_ws_client.core.protocol.connection import SimplyPrintConnection

    connection = SimplyPrintConnection(_WS)

    async def fail(_event: SimplyPrintConnectionLostEvent) -> None:
        raise RuntimeError("lost listener failed")

    connection.event_bus.on(SimplyPrintConnectionLostEvent, fail)
    with pytest.raises(RuntimeError, match="lost listener failed"):
        await connection.protocol._on_disconnected(Disconnected(1))

    assert connection.protocol.v == 1
