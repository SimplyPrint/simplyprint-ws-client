import pytest
from pydantic import ValidationError

from simplyprint_ws_client.core.protocol.messages import PeripheralActionDemandData
from simplyprint_ws_client.core.protocol.models import PeripheralAction
from simplyprint_ws_client.core.state import PeripheralsState
from simplyprint_ws_client.integration import peripherals
from simplyprint_ws_client.integration.peripherals import (
    coerce_percentage_action,
    coerce_toggle_action,
    update_peripheral_state,
)


@pytest.mark.parametrize(
    ("action", "value", "expected"),
    [
        (PeripheralAction.ON, None, True),
        (PeripheralAction.OFF, None, False),
        (PeripheralAction.SET, True, True),
        (PeripheralAction.SET, False, False),
        (PeripheralAction.SET, 1, True),
        (PeripheralAction.SET, 50, None),
        (PeripheralAction.SET, "1", None),
    ],
)
def test_coerce_toggle_action(action, value, expected):
    assert coerce_toggle_action(action, value) is expected


@pytest.mark.parametrize(
    ("action", "value", "expected"),
    [
        (PeripheralAction.ON, None, 100),
        (PeripheralAction.OFF, None, 0),
        (PeripheralAction.SET, True, 100),
        (PeripheralAction.SET, False, 0),
        (PeripheralAction.SET, 34.6, 35),
        (PeripheralAction.SET, "45.5", 46),
        (PeripheralAction.SET, -5, 0),
        (PeripheralAction.SET, 105, 100),
        (PeripheralAction.SET, "invalid", None),
    ],
)
def test_coerce_percentage_action(action, value, expected):
    assert coerce_percentage_action(action, value) == expected


def test_peripheral_action_rejects_unknown_operations():
    with pytest.raises(ValidationError):
        PeripheralActionDemandData(id="fan:part_cooling", a="toggle", v=True)


@pytest.mark.parametrize(
    ("next_value", "next_available", "expected_updated"),
    [
        (40, True, 100),
        (50, True, 200),
        (40, False, 200),
    ],
)
def test_update_peripheral_state_only_stamps_changes(
    monkeypatch, next_value, next_available, expected_updated
):
    now = iter((100.9, 200.9))
    monkeypatch.setattr(peripherals.time, "time", lambda: next(now))
    peripheral = PeripheralsState().fan("part_cooling")

    update_peripheral_state(peripheral, 40, available=True)
    assert (peripheral.value, peripheral.available, peripheral.updated) == (
        40,
        True,
        100,
    )

    update_peripheral_state(peripheral, next_value, available=next_available)
    assert (peripheral.value, peripheral.available, peripheral.updated) == (
        next_value,
        next_available,
        expected_updated,
    )
