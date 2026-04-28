"""Test Temperature state + index based state producer rules with * as index."""

from simplyprint_ws_client import Client, TemperatureMsg


def test_tool0_temperature(client: Client):
    tool0 = client.printer.tool0

    changeset = client.printer.model_recursive_changeset

    assert changeset == {}

    tool0.temperature.actual = 200
    tool0.temperature.target = 250

    messages, _ = client.consume()

    assert len(messages) == 1

    message = messages[0]

    assert message.__class__ == TemperatureMsg
    assert message.data == {"tool0": [200, 250]}

    message.reset_changes(client.printer)

    changeset = client.printer.model_recursive_changeset

    assert changeset == {}

    tool0.temperature.actual = 250
    client.printer.bed.temperature.target = 60

    messages, _ = client.consume()

    assert len(messages) == 1

    message = messages[0]

    assert message.__class__ == TemperatureMsg
    assert message.data == {"tool0": [250, 250], "bed": [None, 60]}

    message.reset_changes(client.printer)

    changeset = client.printer.model_recursive_changeset

    assert changeset == {}


def test_multiple_tools_temperatures(client: Client):
    client.printer.tool_count = 2
    tool0 = client.printer.tools[0]
    tool1 = client.printer.tools[1]

    changeset = client.printer.model_recursive_changeset
    assert changeset == {}

    tool0.temperature.actual = 200
    tool0.temperature.target = 250
    tool1.temperature.actual = 180
    tool1.temperature.target = 220

    messages, _ = client.consume()

    assert len(messages) == 1
    message = messages[0]

    assert message.__class__ == TemperatureMsg

    assert message.data == {"tool0": [200, 250], "tool1": [180, 220]}


def test_chamber_temperature(client: Client):
    client.printer.chamber.temperature.actual = 45

    messages, _ = client.consume()

    assert len(messages) == 1

    message = messages[0]

    assert message.__class__ == TemperatureMsg
    assert message.data == {"chamber": [45]}

    message.reset_changes(client.printer)

    client.printer.chamber.temperature.target = 60

    messages, _ = client.consume()

    assert len(messages) == 1

    message = messages[0]

    assert message.__class__ == TemperatureMsg
    assert message.data == {"chamber": [45, 60]}
