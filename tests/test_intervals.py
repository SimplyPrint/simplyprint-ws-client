from typing import ClassVar

from simplyprint_ws_client.core.state import Interval, Intervals


class TimeControlledIntervals(Intervals):
    ms_time: ClassVar[float] = 30000.0

    @classmethod
    def set_time(cls, ms: float) -> None:
        cls.ms_time = ms

    @classmethod
    def step_time(cls, ms: float) -> None:
        cls.ms_time += ms

    @classmethod
    def now(cls) -> float:
        return cls.ms_time


def test_intervals():
    intervals = TimeControlledIntervals()
    intervals.set_time(30000.0)

    assert intervals.now() == 30000.0

    intervals.ping = 1000

    assert intervals.ping == 1000.0
    assert intervals.is_ready(Interval.PING)

    intervals.use(Interval.PING)

    assert not intervals.is_ready(Interval.PING)
    assert intervals.time_until_ready(Interval.PING) == 1000

    intervals.step_time(1000.0)

    assert intervals.is_ready(Interval.PING)
    assert intervals.time_until_ready(Interval.PING) == 0

    intervals.use(Interval.PING)

    assert not intervals.is_ready(Interval.PING)
    assert intervals.time_until_ready(Interval.PING) == 1000

    intervals.step_time(500.0)

    assert not intervals.is_ready(Interval.PING)
    assert intervals.time_until_ready(Interval.PING) == 500
    assert not intervals.use(Interval.PING)

    intervals.step_time(500.0)

    assert intervals.is_ready(Interval.PING)

    intervals.use(Interval.PING)

    assert not intervals.is_ready(Interval.PING)
    assert intervals.time_until_ready(Interval.PING) == 1000
