"""Regression coverage for protocol demand handlers and client settings."""

import uuid

import pytest

from simplyprint_ws_client.core.protocol.messages import (
    ConnectedMsg,
    ResolveNotificationDemandData,
    SetMaterialDataDemandData,
)
from simplyprint_ws_client.core.protocol.models import DemandMsgType, ServerMsgType
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.core.state import MaterialEntry


@pytest.mark.asyncio
async def test_set_material_data_applies_materials(client):
    data = SetMaterialDataDemandData(
        materials=[
            MaterialEntry(nozzle=0, ext=0, type="PLA", color="Red", hex="#FF0000"),
            # Out-of-range entries must be skipped, not crash.
            MaterialEntry(nozzle=0, ext=99, type="PETG"),
            MaterialEntry(nozzle=42, ext=0, type="ABS"),
        ]
    )

    await client.event_bus.emit(DemandMsgType.SET_MATERIAL_DATA, data)

    applied = client.printer.material(0, 0)
    assert applied is not None
    assert applied.type == "PLA"
    assert applied.color == "Red"
    assert applied.hex == "#FF0000"


@pytest.mark.asyncio
async def test_resolve_notification_for_unknown_event_is_ignored(client):
    data = ResolveNotificationDemandData(event_id=uuid.uuid4(), action=None)

    # Must not raise even though the event id is unknown client-side.
    await client.event_bus.emit(DemandMsgType.RESOLVE_NOTIFICATION, data)


@pytest.mark.asyncio
async def test_connected_msg_without_data_is_tolerated(client):
    msg = ConnectedMsg(type=ServerMsgType.CONNECTED)
    assert msg.data is None

    previous_name = client.config.name

    await client.event_bus.emit(ServerMsgType.CONNECTED, msg)

    assert client.config.name == previous_name


def test_client_settings_tick_rate_is_a_field():
    settings = ClientSettings(tick_rate=2.5)
    assert settings.tick_rate == 2.5
