"""Test MaterialDataMsg for producer triggers and the chain/list conversion bug."""

from itertools import chain

import pytest

from simplyprint_ws_client import Client, MaterialDataMsg
from simplyprint_ws_client.core.state import MaterialLayoutEntry
from simplyprint_ws_client.core.state.models import (
    BedType,
    MultiMaterialSolution,
    NozzleType,
    VIRTUAL_SPOOL_POSITION,
    VolumeType,
)


@pytest.mark.parametrize(
    "field,value,expected_data_key",
    [
        ("bed.type", BedType.BAMBU_TEXTURED_PEI_PLATE, "bed"),
        ("tools.*.size", 0.4, "nozzles"),
        ("tools.*.type", NozzleType.HARDENED_STEEL, "nozzles"),
    ],
)
def test_producer_triggers(client: Client, field, value, expected_data_key):
    """Test that producer-configured fields trigger MaterialDataMsg."""
    changeset = client.printer.model_recursive_changeset
    assert changeset == {}

    # Apply the change
    if field == "bed.type":
        client.printer.bed.type = value
    elif field == "tools.*.size":
        client.printer.tool0.size = value
    elif field == "tools.*.type":
        client.printer.tool0.type = value

    messages = client.consume()
    assert len(messages) == 1
    assert messages[0].__class__ == MaterialDataMsg
    assert expected_data_key in messages[0].data

    # Test reset_changes clears the changeset
    messages[0].reset_changes(client.printer)
    changeset = client.printer.model_recursive_changeset
    assert changeset == {}


def test_producer_vs_fields_inconsistency():
    """Test the inconsistency between producer config and _TOOL_FIELDS."""
    # _TOOL_FIELDS includes fields not watched by producer
    assert MaterialDataMsg._TOOL_FIELDS == {"nozzle", "type", "volume_type", "size"}
    assert MaterialDataMsg._BED_FIELDS == {"type"}


def test_materials_chain_iterator_bug(client: Client):
    """Test the bug fix: materials iterator consumption in chain.from_iterable."""
    # Set up multiple tools
    client.printer.tool_count = 2
    tool0 = client.printer.tools[0]
    tool1 = client.printer.tools[1]

    # Modify materials in both tools
    tool0.materials[0].type = "PLA"
    tool1.materials[0].type = "ABS"

    messages = client.consume()
    assert len(messages) == 1
    assert messages[0].__class__ == MaterialDataMsg
    assert "materials" in messages[0].data

    # This line tests the bug fix - materials must be converted to list
    # before being used in the generator, otherwise iterator gets consumed
    materials = list(
        chain.from_iterable(t.materials.values() for t in client.printer.tools)
    )
    assert len(materials) >= 2

    # Test reset_changes clears material changes
    messages[0].reset_changes(client.printer)
    changeset = client.printer.model_recursive_changeset
    assert changeset == {}


def test_mms_layout_changes(client: Client):
    """Test mms_layout changes trigger MaterialDataMsg."""
    client.printer.update_mms_layout([MaterialLayoutEntry(nozzle=0, size=4)])

    messages = client.consume()
    assert len(messages) == 1
    assert messages[0].__class__ == MaterialDataMsg
    assert "layout" in messages[0].data

    # Test reset_changes clears mms_layout changes
    messages[0].reset_changes(client.printer)
    changeset = client.printer.model_recursive_changeset
    assert changeset == {}


def test_flashforge_ifs_layout_serializes_as_dedicated_four_slot_mms(client: Client):
    client.printer.update_mms_layout(
        [
            MaterialLayoutEntry(
                nozzle=0,
                mms=MultiMaterialSolution.FLASHFORGE_IFS,
            )
        ]
    )

    data = dict(MaterialDataMsg.build_refresh(client.printer))

    assert data["layout"] == [{"nozzle": 0, "mms": "flashforge_ifs"}]
    assert client.printer.tool0.material_count == 4


def test_refresh_mode_includes_all_sections(client: Client):
    """Test build_refresh() includes all message sections."""
    # Deliberate API change: the refresh snapshot is its own classmethod now
    # (build() previously took an is_refresh flag violating its base signature).
    msg = MaterialDataMsg(data=dict(MaterialDataMsg.build_refresh(client.printer)))

    assert "refresh" in msg.data
    assert msg.data["refresh"] is True
    assert "bed" in msg.data
    assert "nozzles" in msg.data
    assert "layout" in msg.data
    assert "materials" in msg.data


def test_material_resize_is_visible_to_delta_build(client: Client):
    client.printer.tool0.material_count = 4
    msgs = client.consume()
    assert len(msgs) == 1 and msgs[0].__class__ == MaterialDataMsg
    sent = {(m["nozzle"], m["ext"]) for m in msgs[0].data["materials"]}
    assert sent == {(0, e) for e in range(4)}
    msgs[0].reset_changes(client.printer)
    assert client.printer.model_recursive_changeset == {}

    client.printer.tool0.material_count = 2
    msgs = client.consume()
    assert len(msgs) == 1
    sent = {(m["nozzle"], m["ext"]) for m in msgs[0].data["materials"]}
    assert sent == {(0, 0), (0, 1)}


def test_virtual_layout_entry_is_addressable(client: Client):
    client.printer.update_mms_layout(
        [
            MaterialLayoutEntry(nozzle=0, mms=MultiMaterialSolution.QIDI_BOX),
            MaterialLayoutEntry(nozzle=0, mms=MultiMaterialSolution.VIRTUAL),
        ]
    )
    tool = client.printer.tool0
    assert tool.material_count == 4
    assert set(tool.materials) == set(range(4)) | {VIRTUAL_SPOOL_POSITION}
    assert client.printer.material(0, VIRTUAL_SPOOL_POSITION) is not None

    # Removing VIRTUAL from the layout keeps the key but clears the entry:
    # deleting it would be invisible to SimplyPrint (only cleared entries are
    # sent), and leaving it populated would resend stale data on refresh.
    client.printer.material(0, VIRTUAL_SPOOL_POSITION).type = "PLA"
    client.printer.update_mms_layout(
        [MaterialLayoutEntry(nozzle=0, mms=MultiMaterialSolution.QIDI_BOX)]
    )
    assert set(tool.materials) == set(range(4)) | {VIRTUAL_SPOOL_POSITION}
    assert tool.materials[VIRTUAL_SPOOL_POSITION].type is None


def test_boxturtle_get_chains_no_max():
    entry = MaterialLayoutEntry(nozzle=0, mms=MultiMaterialSolution.BOXTURTLE, chains=2)
    assert entry.get_chains() == 2
    assert entry.get_computed_size() == 8


def test_reset_changes_mirrors_producer_fields(client: Client):
    """Test that reset_changes clears exactly the fields that producer watches."""
    # Make changes to all producer-watched fields
    client.printer.bed.type = BedType.BAMBU_TEXTURED_PEI_PLATE
    assert client.printer.bed.model_has_changes("type")

    client.printer.tool_count = 2

    for i in range(0, client.printer.tool_count):
        tool = client.printer.tool(i)
        tool.size = 0.4
        tool.type = NozzleType.HARDENED_STEEL
        tool.volume_type = VolumeType.HIGH_FLOW
        client.printer.update_mms_layout([MaterialLayoutEntry(nozzle=i, size=4)])
        for material in tool.materials.values():
            material.type = "PLA"
            material.color = "Blue"
            material.hex = "#FF0000"
        # Verify producer-watched fields have changes
        assert {"size", "type", "volume_type"}.issubset(tool.model_changed_fields)

    # Build and reset message
    msgs = client.consume()
    assert len(msgs) == 1
    assert msgs[0].__class__ == MaterialDataMsg
    msgs = client.consume()
    assert len(msgs) == 0
