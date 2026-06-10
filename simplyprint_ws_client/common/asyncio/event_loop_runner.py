"""
Running asyncio event loop across multiple versions.

Based on https://github.com/MagicStack/uvloop/blob/master/uvloop/__init__.py

Original License:

The MIT License
Copyright (C) 2016-present the uvloop authors and contributors.
"""

__all__ = ["Runner", "EventLoopBackend", "enable_asyncio_debug"]

import asyncio
import contextlib
import contextvars
import sys
from enum import Enum, auto
from typing import (
    Union,
    TypeVar,
    Coroutine,
    Any,
    List,
    Callable,
    Optional,
    TYPE_CHECKING,
)

try:
    import uvloop
except ImportError:
    uvloop = None

_T = TypeVar("_T")
Loop = Union[asyncio.AbstractEventLoop]


class EventLoopBackend(Enum):
    AUTO = auto()
    UVLOOP = auto()
    ASYNCIO = auto()

    def new_event_loop(self):
        uvloop_available = uvloop is not None

        if self is self.AUTO:
            return (
                uvloop.new_event_loop()
                if uvloop_available
                else asyncio.new_event_loop()
            )

        if self is self.UVLOOP:
            assert uvloop_available, "uvloop must be available to use"
            return uvloop.new_event_loop()

        return asyncio.new_event_loop()


_loop_debug_enabled = contextvars.ContextVar("_loop_debug_enabled", default=False)
_loop_backend = contextvars.ContextVar("_loop_backend", default=EventLoopBackend.AUTO)


@contextlib.contextmanager
def enable_asyncio_debug():
    token = _loop_debug_enabled.set(True)
    try:
        yield
    finally:
        _loop_debug_enabled.reset(token)


class Runner:
    """Wrapper around uvloop/asyncio implementations for running the main event loop."""

    debug = False
    context_stack: List[Callable[[], contextlib.AbstractContextManager]]
    backend: Optional[EventLoopBackend] = None

    def __init__(self, debug=False, context_stack=None, backend=None):
        self.debug = debug
        self.context_stack = [] if context_stack is None else context_stack
        self.backend = backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

    if TYPE_CHECKING:

        def run(
            self,
            main: Coroutine[Any, Any, _T],
            *,
            loop_factory: Optional[Callable[[], Loop]] = asyncio.new_event_loop,
            debug: Optional[bool] = None,
        ) -> _T:
            """The preferred way of running a coroutine with uvloop."""
    else:

        def run(self, *args, **kwargs) -> None:
            try:
                with contextlib.ExitStack() as stack:
                    for context_func in self.context_stack:
                        stack.enter_context(context_func())

                    kwargs["debug"] = kwargs.get("debug") or (
                        self.debug or _loop_debug_enabled.get()
                    )
                    kwargs["loop_factory"] = (
                        kwargs.get("loop_factory")
                        or (self.backend or _loop_backend.get()).new_event_loop
                    )

                    return run(*args, **kwargs)
            except asyncio.CancelledError:
                pass


def run(main, *, loop_factory=asyncio.new_event_loop, debug=None, **run_kwargs):
    vi = sys.version_info[:2]

    if vi <= (3, 10):
        # Copied from python/cpython

        if asyncio._get_running_loop() is not None:
            raise RuntimeError(
                "asyncio.run() cannot be called from a running event loop"
            )

        if not asyncio.iscoroutine(main):
            raise ValueError("a coroutine was expected, got {!r}".format(main))

        loop = loop_factory()
        try:
            asyncio.set_event_loop(loop)
            if debug is not None:
                loop.set_debug(debug)
            return loop.run_until_complete(main)
        finally:
            try:
                _cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    elif vi == (3, 11):
        if asyncio._get_running_loop() is not None:
            raise RuntimeError(
                "asyncio.run() cannot be called from a running event loop"
            )

        with asyncio.Runner(
            loop_factory=loop_factory, debug=debug, **run_kwargs
        ) as runner:
            return runner.run(main)

    else:
        assert vi >= (3, 12)
        return asyncio.run(main, loop_factory=loop_factory, debug=debug, **run_kwargs)


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    # Copied from python/cpython

    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )
