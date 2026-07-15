"""Test active_material per tool -> ToolMsg flow."""

from simplyprint_ws_client import Client
from simplyprint_ws_client.core.protocol.messages import ToolMsg


def test_active_material_change(client: Client):
    changeset = client.printer.model_recursive_changeset
    assert changeset == {}

    # Change active_material on tool0
    client.printer.tool0.active_material = 1

    messages = client.consume()

    assert len(messages) == 1

    message = messages[0]

    assert message.__class__ == ToolMsg
    # active_tools is derived from each tool's active_material
    assert message.data == {"active_tools": {0: 1}}

    message.reset_changes(client.printer)

    changeset = client.printer.model_recursive_changeset
    assert changeset == {}
