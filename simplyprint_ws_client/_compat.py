"""Explicit imports for standard-library features absent on older Python."""

import sys

__all__ = ["AsyncEvent", "AsyncLock", "AsyncQueue", "StrEnum"]

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # the `strenum` backport is a dependency on Python < 3.11
    from strenum import StrEnum  # noqa: F401

if sys.version_info >= (3, 10):
    from asyncio import Event as AsyncEvent
    from asyncio import Lock as AsyncLock
    from asyncio import Queue as AsyncQueue
else:
    # On Python 3.9 asyncio primitives bind to ``get_event_loop()`` when they
    # are constructed. Objects built outside their loop (on the main thread, or
    # before the loop thread runs) then fail with "no current event loop" or
    # "attached to a different loop". Python 3.10+ binds on first use from the
    # running loop instead; these subclasses backport exactly that.
    import asyncio
    import collections

    class _LoopBoundMixin:
        _bound_loop = None

        @property
        def _loop(self):
            loop = asyncio.get_running_loop()
            if self._bound_loop is None:
                self._bound_loop = loop
            elif loop is not self._bound_loop:
                raise RuntimeError(f"{self!r} is bound to a different event loop")
            return loop

    class AsyncEvent(_LoopBoundMixin, asyncio.Event):
        def __init__(self) -> None:
            self._waiters = collections.deque()
            self._value = False

    class AsyncLock(_LoopBoundMixin, asyncio.Lock):
        def __init__(self) -> None:
            self._waiters = None
            self._locked = False

    class AsyncQueue(_LoopBoundMixin, asyncio.Queue):
        def __init__(self, maxsize: int = 0) -> None:
            self._maxsize = maxsize
            self._getters = collections.deque()
            self._putters = collections.deque()
            self._unfinished_tasks = 0
            self._finished = AsyncEvent()
            self._finished.set()
            self._init(maxsize)
