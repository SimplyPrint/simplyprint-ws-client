from simplyprint_ws_client.common.utils.gcode_parser import GcodeParser, GcodeCommand


def test_simple():
    gcode = [
        "G1    X10 Y20 Z30 ; Move to position",
        "G1 X40    Y50 Z60 ; Another move",
        "G1 X70    Y80 Z90 ; Final move",
        "M104 S1000",
        "M140 I13 S20",
        "M104 T1 S100 F0 B100",
        "M140 F1 S100.51 F0 B100",
    ]

    parser = GcodeParser()
    parsed = list(parser.parse_gcode(gcode))

    assert len(parsed) == 7
    assert parsed[0] == GcodeCommand("G1", [("X", 10), ("Y", 20), ("Z", 30)])
    assert parsed[1] == GcodeCommand("G1", [("X", 40), ("Y", 50), ("Z", 60)])
    assert parsed[2] == GcodeCommand("G1", [("X", 70), ("Y", 80), ("Z", 90)])
    assert parsed[3] == GcodeCommand("M104", [("S", 1000)])
    assert parsed[4] == GcodeCommand("M140", [("I", 13), ("S", 20)])
    assert parsed[5] == GcodeCommand(
        "M104", [("T", 1), ("S", 100), ("F", 0), ("B", 100)]
    )
    assert parsed[6] == GcodeCommand(
        "M140", [("F", 1), ("S", 100.51), ("F", 0), ("B", 100)]
    )
