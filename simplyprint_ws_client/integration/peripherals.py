"""Shared peripheral action and state semantics for printer integrations."""

from __future__ import annotations

import time
from typing import Any, Optional

from simplyprint_ws_client.core.protocol.models import PeripheralAction
from simplyprint_ws_client.core.state import PeripheralHandle

__all__ = [
    "coerce_percentage_action",
    "coerce_toggle_action",
    "update_peripheral_state",
]


def coerce_toggle_action(action: PeripheralAction, value: Any) -> Optional[bool]:
    """Resolve an on/off/set action to a boolean value."""
    if action is PeripheralAction.ON:
        return True
    if action is PeripheralAction.OFF:
        return False
    if action is not PeripheralAction.SET:
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def coerce_percentage_action(action: PeripheralAction, value: Any) -> Optional[int]:
    """Resolve an on/off/set action to a percentage from zero to one hundred."""
    if action is PeripheralAction.ON:
        return 100
    if action is PeripheralAction.OFF:
        return 0
    if action is not PeripheralAction.SET:
        return None
    if isinstance(value, bool):
        return 100 if value else 0
    if isinstance(value, (int, float)):
        return max(0, min(100, round(value)))
    if isinstance(value, str):
        try:
            return max(0, min(100, round(float(value))))
        except ValueError:
            return None
    return None


def update_peripheral_state(
    peripheral: PeripheralHandle, value: Any, *, available: bool
) -> None:
    """Update a peripheral only when its reported state changed."""
    state = peripheral.state
    if state is not None and state.value == value and state.available == available:
        return
    peripheral.update(value=value, available=available, updated=int(time.time()))
