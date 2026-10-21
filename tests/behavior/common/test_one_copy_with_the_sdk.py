"""The framework's merge-and-diff and G-code parser *are* the SDK's.

Brands build their device models on ``simplyprint_sdk.pydantic`` and parse
G-code with ``simplyprint_sdk.gcode``; framework code and older integrations
import the same names from ``simplyprint_ws_client``. When these were copies,
an ``isinstance(diff, UpdatedField)`` or ``is_atomic`` from one copy silently
never matched the other's. The framework now re-exports the SDK's objects, so
class identity, ``isinstance`` and the markers hold across both import paths.
"""

from typing import List, Optional

import pytest
from pydantic import BaseModel

from simplyprint_sdk import gcode as sdk_gcode
from simplyprint_sdk import pydantic as sdk

from simplyprint_ws_client.common import model as framework
from simplyprint_ws_client.common.model import annotations, diff
from simplyprint_ws_client.common.utils import gcode_parser

SHARED_MODEL_NAMES = [
    "Atomic",
    "ExtraInfo",
    "is_atomic",
    "is_extra",
    "SimpleUpdateModel",
    "UpdateModel",
    "UpdatedField",
    "UpdatedFieldsType",
    "updated_fields_pretty_print",
    "updated_fields_substitute_new",
]


@pytest.mark.parametrize("name", SHARED_MODEL_NAMES)
def test_the_framework_model_names_are_the_sdk_objects(name):
    assert getattr(framework, name) is getattr(sdk, name)


def test_the_old_module_paths_hold_the_sdk_objects_too():
    for name in diff.__all__:
        assert getattr(diff, name) is getattr(sdk, name), name
    for name in ("Atomic", "ExtraInfo", "is_atomic", "is_extra"):
        assert getattr(annotations, name) is getattr(sdk, name), name
    for name in gcode_parser.__all__:
        assert getattr(gcode_parser, name) is getattr(sdk_gcode, name), name


def test_the_framework_only_markers_stay_the_frameworks():
    # Exclusive / Untracked drive the framework's ReactiveModel changeset; the
    # SDK has no such engine, so it must not carry them.
    assert not hasattr(sdk, "Exclusive") and not hasattr(sdk, "Untracked")
    assert framework.Exclusive is annotations.Exclusive


class _SdkTray(BaseModel, sdk.SimpleUpdateModel):
    color: Optional[str] = None


class _FrameworkDevice(BaseModel, framework.SimpleUpdateModel):
    """A framework-path model holding an SDK-path one, with the SDK's marker."""

    trays: Optional[List[_SdkTray]] = None
    hms: sdk.Atomic[Optional[List[int]]] = None
    t_utc: framework.ExtraInfo[Optional[int]] = None


def test_models_and_markers_mix_across_the_two_import_paths():
    fields = _FrameworkDevice.model_fields
    # A marker from either path is seen by the predicate from either path.
    assert framework.is_atomic(fields["hms"]) and sdk.is_atomic(fields["hms"])
    assert framework.is_extra(fields["t_utc"]) and sdk.is_extra(fields["t_utc"])

    device = _FrameworkDevice(trays=[_SdkTray(color="red")], hms=[1])
    assert isinstance(device, sdk.SimpleUpdateModel)
    assert isinstance(device.trays[0], framework.UpdateModel)

    changes = device.update_model(
        _FrameworkDevice(trays=[_SdkTray(color="blue")], hms=[2], t_utc=5)
    )
    # The nested SDK model was merged as a model, not replaced; hms whole.
    assert changes == {
        "trays": [{"color": sdk.UpdatedField("red", "blue")}],
        "hms": framework.UpdatedField([1], [2]),
    }
    assert type(changes["hms"]) is sdk.UpdatedField
    assert isinstance(changes["trays"][0]["color"], framework.UpdatedField)
    assert device.t_utc is None


def test_gcode_commands_are_one_class():
    (command,) = gcode_parser.GcodeParser().parse_gcode(["M104 S200"])
    assert type(command) is sdk_gcode.GcodeCommand
    assert command == sdk_gcode.GcodeCommand("M104", [("S", 200)])
