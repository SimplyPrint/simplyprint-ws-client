import pytest
from pydantic import ValidationError

from simplyprint_ws_client import (
    Client,
    ClientMsgType,
    DefaultClient,
    DemandMsgType,
    PeripheralAction,
    PeripheralActionDemandData,
    PeripheralMsg,
    PrinterConfig,
    ServerMsg,
)
from simplyprint_ws_client.core.ws_protocol.events import (
    ConnectionIncomingEvent,
    ConnectionOutgoingEvent,
)


def _peripheral_msgs(client: Client):
    msgs, _ = client.consume()
    return [msg for msg in msgs if isinstance(msg, PeripheralMsg)]


def test_peripheral_state_is_sent_once_per_change(client: Client):
    door = client.printer.door("p2")
    assert door.state is None

    door.set(True, available=True, updated=100)

    (msg,) = _peripheral_msgs(client)
    assert msg.type == ClientMsgType.PERIPHERAL
    assert msg.data == {"door:p2": {"v": True, "a": True, "u": 100}}

    # Reporting the same state again is not a change.
    door.set(True, available=True, updated=100)
    assert _peripheral_msgs(client) == []

    door.update(value=False)
    (msg,) = _peripheral_msgs(client)
    assert msg.data == {"door:p2": {"v": False, "a": True, "u": 100}}


def test_only_changed_peripherals_are_sent(client: Client):
    client.printer.fan("part").set(40, available=True)
    client.printer.light("chamber").set(False, available=True)
    _peripheral_msgs(client)

    client.printer.fan("part").set(80)

    (msg,) = _peripheral_msgs(client)
    assert msg.data == {"fan:part": {"v": 80, "a": True}}


def test_peripheral_action_demand():
    msg = ServerMsg.model_validate(
        {
            "type": "demand",
            "data": {
                "demand": "peripheral_action",
                "id": "door:p2",
                "a": "set",
                "v": 1,
            },
        }
    ).root

    assert msg.data.demand == DemandMsgType.PERIPHERAL_ACTION
    assert msg.data.id == "door:p2"
    assert msg.data.action is PeripheralAction.SET
    assert msg.data.value == 1


def test_peripheral_action_rejects_unknown_operations():
    with pytest.raises(ValidationError):
        PeripheralActionDemandData(id="fan:part", a="toggle", v=True)


class _DoorClient(DefaultClient[PrinterConfig]):
    definitions = {"door:p2": {"f": "rw", "n": "Door Lock", "vt": "bool"}}

    def get_current_peripheral_definitions(self):
        return self.definitions


async def _refresh_peripherals(client: Client):
    sent = []
    client.event_bus.on(ConnectionOutgoingEvent, lambda msg, v: sent.append(msg))
    demand = ServerMsg.model_validate(
        {"type": "demand", "data": {"demand": "refresh_peripherals"}}
    ).root
    await client.event_bus.emit(ConnectionIncomingEvent, demand, client.v)
    return sent


@pytest.mark.asyncio
async def test_refresh_peripherals_sends_definitions():
    client = _DoorClient(PrinterConfig.get_new())

    (msg,) = await _refresh_peripherals(client)

    assert msg.type == ClientMsgType.PERIPHERAL_DEFINITIONS
    assert msg.data == _DoorClient.definitions


@pytest.mark.asyncio
async def test_refresh_peripherals_without_definitions_sends_nothing():
    client = DefaultClient(PrinterConfig.get_new())

    assert await _refresh_peripherals(client) == []
