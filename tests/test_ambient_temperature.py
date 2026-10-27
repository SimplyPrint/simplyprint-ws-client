import time

import pytest
from pydantic import BaseModel

from simplyprint_ws_client import PrinterStatus, PrinterState


class CustomTime(BaseModel):
    _time: float = 0
    _fail: bool = False

    def fail(self):
        self._fail = True

    def set(self, t: float):
        self._time = t

    def time(self) -> float:
        if self._fail:
            return -1

        return self._time


class CustomPrinterStateWithTime(PrinterState):
    config: None = None
    time: CustomTime

    # Progress ambient state n-steps
    def do(self, n=2):
        for _ in range(n):
            self.time.set(self.time.time() + self.ambient_temperature._update_interval)
            self.ambient_temperature.tick(self)


@pytest.fixture
def state(monkeypatch):
    custom_time = CustomTime()
    # Through monkeypatch, so the real clock is back after the test: a bare
    # assignment left every later test on a clock near 1970 (ZIP timestamps
    # before 1980 then broke the bundle tests, in whichever order ran them).
    monkeypatch.setattr(time, "time", lambda: custom_time.time())
    state = CustomPrinterStateWithTime(time=custom_time)
    state.tool_count = 1
    return state


def test_simple(state):
    tool0 = state.tools[0].temperature

    tool0.actual = 27.21875
    tool0.target = 0.0

    state.do()

    assert state.ambient_temperature.ambient == 27


def test_with_printing_state(state):
    tool0 = state.tools[0].temperature

    tool0.actual = 27.21875
    tool0.target = 0.0

    state.do()

    assert state.ambient_temperature.ambient == 27

    tool0.target = 200.0
    tool0.actual = 150.0

    state.do()

    assert state.ambient_temperature.ambient == 27

    state.status = PrinterStatus.PRINTING
    tool0.actual = 200.0

    state.do()

    assert state.ambient_temperature.ambient == 27

    state.status = PrinterStatus.OPERATIONAL
    tool0.target = 0

    # Cool down of 5 degrees per check - no change
    for _ in range(35):
        state.do(1)
        tool0.actual -= 5
        assert state.ambient_temperature.ambient == 27

    # Now temperature is 25 degrees and we do 2 checks
    state.do()
    # New ambient temperature is 25
    assert state.ambient_temperature.ambient == 25


def test_fail_case(state):
    tool0 = state.tools[0].temperature

    tool0.actual = 27.21875
    tool0.target = 0.0

    state.do()

    assert state.ambient_temperature.ambient == 27

    tool0.actual = 30

    state.do()

    assert state.ambient_temperature.ambient == 30

    tool0.actual = 25

    state.time.fail()

    state.do()

    assert state.ambient_temperature.ambient == 30
