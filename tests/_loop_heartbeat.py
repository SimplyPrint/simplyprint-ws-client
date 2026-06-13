"""Test-only helper: sample app-loop wake-gaps to prove the loop stays
responsive under a scenario.

This is a *test fixture*, not a runtime component -- there is no loop monitor in
the product. A background coroutine wakes every ``interval`` and records the gap
between successive wakeups; when something blocks the loop thread, exactly one
gap balloons to the block duration. A test asserts ``max_gap_ms`` stays near the
interval to prove the offending work no longer runs on the loop.

    async with LoopHeartbeat() as hb:
        await do_something_that_must_not_block_the_loop()
    assert hb.max_gap_ms < 200
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional


class LoopHeartbeat:
    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.gaps: List[float] = []
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "LoopHeartbeat":
        self._task = asyncio.get_running_loop().create_task(self._run())
        await asyncio.sleep(self.interval * 2)  # a couple of clean beats first
        return self

    async def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            await asyncio.sleep(self.interval)
            now = time.perf_counter()
            self.gaps.append(now - last)
            last = now

    async def __aexit__(self, *_exc) -> None:
        # Record one more beat so a stall right before exit is captured.
        await asyncio.sleep(self.interval * 2)
        self._stop.set()
        if self._task is not None:
            await self._task

    @property
    def max_gap_ms(self) -> float:
        return max(self.gaps) * 1000.0 if self.gaps else 0.0

    @property
    def interval_ms(self) -> float:
        return self.interval * 1000.0
