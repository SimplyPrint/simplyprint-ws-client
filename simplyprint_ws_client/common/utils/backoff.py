import time
from random import Random
from typing import Protocol


class Backoff(Protocol):
    def delay(self) -> float: ...

    def reset(self) -> None: ...


class ConstantBackoff(Backoff):
    d: int  # Delay

    def __init__(self, delay: int = 5):
        self.d = delay

    def delay(self) -> float:
        return self.d

    def reset(self) -> None:
        pass


class LinearBackoff(Backoff):
    d: int  # Delay
    min: int  # Minimum
    max: int  # Maximum
    inc: int  # Increment

    def __init__(self, mi: int = 0, ma: int = 0, fa: int = 1):
        self.d = mi
        self.min = mi
        self.max = ma
        self.inc = fa

    def delay(self) -> float:
        self.d = min(self.d + self.inc, self.max)
        return self.d

    def reset(self) -> None:
        self.d = self.min


class ExponentialBackoff(Backoff):
    """Derived from https://github.com/Rapptz/discord.py/blob/master/discord/backoff.py"""

    base: float
    exp: int
    reset_time: float
    ts: float  # Previous invocation
    rand: Random

    def __init__(
        self, base: float = 2.5, maximum: float = 30.0, reset_time: float = 60.0
    ):
        """
        Less-steep exponential backoff that starts at ~1s, hits ~5s quickly, and never exceeds ~30s.

        Args:
            base: Controls how steeply the backoff grows
            maximum: The max (cap) time in seconds for any delay
            reset_time: If no calls happen for 'reset_time' seconds, exponent resets
        """
        self.base = base
        self.exp = -1  # start at -1 so first call is around 1 second
        self.max_delay = maximum
        self.reset_time = reset_time

        self.ts = time.monotonic()  # timestamp of last .delay() call
        self.rand = Random()
        self.rand.seed()

    def delay(self) -> float:
        ts = time.monotonic()
        interval = ts - self.ts
        self.ts = ts

        # If too much time has passed since the last call, reset exponent
        if interval > self.reset_time:
            self.reset()

        self.exp += 1

        # The 'ideal' exponential factor
        ideal = self.base * (2**self.exp)

        # Clamp it to the maximum
        delay = min(ideal, self.max_delay)

        # You can also introduce a slight randomization around [delay*0.8, delay*1.2], etc.
        # For simplicity, let's just do a uniform between [delay/2, delay].
        return self.rand.uniform(delay / 2, delay)

    def reset(self) -> None:
        self.exp = -1
