from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from simplyprint_ws_client import ClientContext, PrinterConfig
from simplyprint_ws_client.core.client import Client, ClientState
from simplyprint_ws_client.core.protocol.messages import (
    ConnectedMsg,
    PauseDemandData,
    RefreshMaterialDataDemandData,
    StreamOffDemandData,
    StreamOnDemandData,
    WebcamSnapshotDemandData,
    WebcamTestDemandData,
)
from simplyprint_ws_client.core.protocol.models import DemandMsgType, ServerMsgType
from simplyprint_ws_client.integration.client import PrinterClient


class RecordingPrinter(PrinterClient[PrinterConfig]):
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        super().__init__(PrinterConfig.get_new(), context=ClientContext())

    async def on_connected(self, msg: ConnectedMsg) -> None:
        self.calls.append(("connected", (msg, self.state)))

    async def on_pause(self, data: PauseDemandData) -> None:
        self.calls.append(("pause", data))

    async def on_refresh_material_data(
        self, data: RefreshMaterialDataDemandData | None = None
    ) -> None:
        self.calls.append(("materials", data))

    async def on_webcam_snapshot(
        self, data: WebcamSnapshotDemandData | None = None
    ) -> None:
        self.calls.append(("snapshot", data))

    async def on_stream_on(self, data: StreamOnDemandData | None = None) -> None:
        self.calls.append(("stream_on", data))

    async def on_stream_off(self, data: StreamOffDemandData | None = None) -> None:
        self.calls.append(("stream_off", data))

    async def on_test_webcam(self, data: WebcamTestDemandData | None = None) -> None:
        self.calls.append(("test_webcam", data))


@pytest.mark.asyncio
async def test_printer_routes_deliver_typed_payloads_once() -> None:
    client = RecordingPrinter()
    pause = PauseDemandData()
    materials = RefreshMaterialDataDemandData()
    stream_on = StreamOnDemandData()
    stream_off = StreamOffDemandData()
    webcam_test = WebcamTestDemandData()
    snapshot = WebcamSnapshotDemandData(id="snapshot-1", timer=17)

    await client.event_bus.emit(DemandMsgType.PAUSE, pause)
    await client.event_bus.emit(DemandMsgType.REFRESH_MATERIAL_DATA, materials)
    await client.event_bus.emit(DemandMsgType.STREAM_ON, stream_on)
    await client.event_bus.emit(DemandMsgType.STREAM_OFF, stream_off)
    await client.event_bus.emit(DemandMsgType.TEST_WEBCAM, webcam_test)
    await client.event_bus.emit(DemandMsgType.WEBCAM_SNAPSHOT, snapshot)

    assert client.calls == [
        ("pause", pause),
        ("materials", materials),
        ("stream_on", stream_on),
        ("stream_off", stream_off),
        ("test_webcam", webcam_test),
        ("snapshot", snapshot),
    ]
    assert client.printer.intervals.webcam == 17


@pytest.mark.asyncio
async def test_camera_routes_delegate_to_composed_controller() -> None:
    client = PrinterClient(PrinterConfig.get_new(), context=ClientContext())
    stream_on = StreamOnDemandData()
    stream_off = StreamOffDemandData()
    webcam_test = WebcamTestDemandData()
    snapshot = WebcamSnapshotDemandData(id="snapshot-2")
    client.camera.stream_on = AsyncMock()
    client.camera.stream_off = AsyncMock()
    client.camera.test_webcam = AsyncMock()
    client.camera.snapshot = AsyncMock()

    await client.event_bus.emit(DemandMsgType.STREAM_ON, stream_on)
    await client.event_bus.emit(DemandMsgType.STREAM_OFF, stream_off)
    await client.event_bus.emit(DemandMsgType.TEST_WEBCAM, webcam_test)
    await client.event_bus.emit(DemandMsgType.WEBCAM_SNAPSHOT, snapshot)

    client.camera.stream_on.assert_awaited_once_with(stream_on)
    client.camera.stream_off.assert_awaited_once_with(stream_off)
    client.camera.test_webcam.assert_awaited_once_with(webcam_test)
    client.camera.snapshot.assert_awaited_once_with(snapshot)


@pytest.mark.asyncio
async def test_core_connected_state_precedes_public_handler() -> None:
    client = RecordingPrinter()
    msg = ConnectedMsg(type=ServerMsgType.CONNECTED)

    await client.event_bus.emit(ServerMsgType.CONNECTED, msg)

    assert client.calls == [("connected", (msg, ClientState.CONNECTED))]


class PlainClient(Client[PrinterConfig]):
    def __init__(self) -> None:
        self.pause_calls = 0
        super().__init__(PrinterConfig.get_new(), context=ClientContext())

    async def on_pause(self, _data: PauseDemandData) -> None:
        self.pause_calls += 1


@pytest.mark.asyncio
async def test_plain_client_method_names_do_not_implicitly_create_routes() -> None:
    client = PlainClient()

    await client.event_bus.emit(DemandMsgType.PAUSE, PauseDemandData())

    assert client.pause_calls == 0
