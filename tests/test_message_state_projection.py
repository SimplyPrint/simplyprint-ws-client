from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.protocol.messages import (
    CpuInfoMsg,
    FirmwareMsg,
    FirmwareWarningMsg,
    PeripheralMsg,
    WebcamMsg,
)


def test_state_messages_project_typed_values_to_wire_payloads():
    state = Client(PrinterConfig.get_new(), context=ClientContext()).printer

    state.webcam_settings.flipH = True
    state.firmware.name = "Klipper"
    state.firmware.machine_name = "CoreXY"
    state.firmware_warning.severity = "warning"
    state.cpu_info.usage = 42
    state.peripherals.update(
        "fan:part_cooling",
        value=65,
        available=True,
        updated=123,
    )

    assert dict(WebcamMsg.build(state)) == {"flipH": True}
    assert dict(FirmwareMsg.build(state)) == {
        "fw": {
            "firmware": "Klipper",
            "firmware_machine_name": "CoreXY",
        }
    }
    assert dict(FirmwareWarningMsg.build(state)) == {"severity": "warning"}
    assert dict(CpuInfoMsg.build(state)) == {"usage": 42.0}
    assert dict(PeripheralMsg.build(state)) == {
        "fan:part_cooling": {"v": 65, "a": True, "u": 123}
    }
