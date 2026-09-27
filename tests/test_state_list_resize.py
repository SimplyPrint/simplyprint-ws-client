import functools
import random
import pytest
from concurrent.futures.thread import ThreadPoolExecutor

from simplyprint_ws_client import PrinterState, PrinterConfig
from simplyprint_ws_client.core.state import VIRTUAL_SPOOL_POSITION


def _assert_printer_state_consistent(printer: PrinterState):
    # Ensure that the internal state of the printer is consistent
    assert printer.tool_count >= 1
    assert printer.tool_count <= 255

    for tool in printer.tools:
        assert tool.material_count >= 1
        assert tool.material_count <= 255

    for i, tool in enumerate(printer.tools):
        assert tool.nozzle == i, (
            f"Tool index mismatch: expected {i}, got {tool.nozzle} {printer.tools=}"
        )
        # Snapshot under the tool lock: iterating during a concurrent resize
        # raises RuntimeError.
        with tool:
            items = list(tool.materials.items())
        for ext, material in items:
            assert material.ext == ext
            assert material.nozzle == i
        seq = sorted(e for e, _ in items if e < VIRTUAL_SPOOL_POSITION)
        assert seq == list(range(len(seq)))


@pytest.fixture
def printer():
    return PrinterState(config=PrinterConfig.get_new())


def _test_state_list_resize_by_property(
    printer, obj: object, property_name: str, n=1024, m=16
):
    # Fuzz material_count and nozzle_count properties
    # with random numbers between 1 and 255 and make sure
    # no assertions fail. Size up and down.

    def safe_rand(min_value, max_value):
        return (
            random.randint(min_value, max_value) if min_value < max_value else min_value
        )

    for i in range(n):
        s = getattr(obj, property_name)

        if i % 2 == 0:
            # Size up
            s = safe_rand(s or 1, m)
        else:
            # Size down
            s = safe_rand(1, s)

        setattr(obj, property_name, s)

        _assert_printer_state_consistent(printer)


def _test_state_list_resize_by_property_multithreaded(
    printer, obj: object, property_name: str, n=1024, tc=10
):
    with ThreadPoolExecutor(max_workers=tc) as executor:
        futures = []

        for i in range(tc):
            future = executor.submit(
                functools.partial(
                    _test_state_list_resize_by_property, printer, obj, property_name, n
                )
            )

            futures.append(future)

        for future in futures:
            future.result()


def test_state_list_resize(printer):
    _test_state_list_resize_by_property(printer, printer, "tool_count")
    _test_state_list_resize_by_property(printer, printer.tool(), "material_count")


def test_material_resize_preserves_virtual_entries(printer):
    tool = printer.tool0
    tool.material_count = 4
    virtual = tool.material(VIRTUAL_SPOOL_POSITION)
    virtual.type = "PLA"

    tool.material_count = 8
    assert set(tool.materials) == set(range(8)) | {VIRTUAL_SPOOL_POSITION}
    assert tool.materials[VIRTUAL_SPOOL_POSITION] is virtual
    assert tool.material_count == 8
    assert printer.material(0, 4) is not None
    assert printer.material(0, VIRTUAL_SPOOL_POSITION) is virtual
    assert printer.material(0, 99) is None

    tool.material_count = 4
    assert set(tool.materials) == set(range(4)) | {VIRTUAL_SPOOL_POSITION}
    assert tool.material_count == 4

    with pytest.raises(ValueError):
        tool.material(-1)


def test_state_list_resize_multithreaded(printer):
    with ThreadPoolExecutor(2) as executor:
        f1 = executor.submit(
            _test_state_list_resize_by_property_multithreaded,
            printer,
            printer,
            "tool_count",
        )
        f2 = executor.submit(
            _test_state_list_resize_by_property_multithreaded,
            printer,
            printer.tool(),
            "material_count",
        )
        f1.result()
        f2.result()
