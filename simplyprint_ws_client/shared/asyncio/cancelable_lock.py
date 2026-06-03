import asyncio
import collections
from typing import Optional


class CancelableLock(asyncio.Lock):
    _waiters: Optional[collections.deque]

    def __len__(self) -> int:
        """Return the number of waiters in the queue."""
        if self._waiters is None:
            return 0

        return len(self._waiters)

    def cancel(self):
        if not self._waiters:
            return

        for waiter in self._waiters:
            if not asyncio.isfuture(waiter) or waiter.done():
                continue

            waiter.cancel("Lock was canceled")
