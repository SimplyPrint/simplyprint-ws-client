"""Scopes: one owner for everything a mount registered, undone in reverse.

A :class:`Scope` records how to undo each thing registered under it. Closing it
runs those undos last-in first-out, so whatever was registered last (the
printers of a domain) goes first and whatever it depended on (the registry
entry) goes last. Every undo is isolated: one that raises is recorded in the
:class:`CloseReport` and the rest still run.

* ``on_close(undo, loop=...)`` hops the undo onto ``loop`` (the loop that owns
  the state it touches) and waits for it, including any awaitable it returns.
  With no loop the undo runs on the closing thread; an awaitable it returns is
  driven on a private loop in a helper thread.
* ``spawn(coro, name=...)`` starts a task on the scope's loop. Close cancels it
  and waits for it to finish.
* ``child(name)`` makes a sub-scope. Children close before their parent's own
  undos, newest child first; a child can also be closed on its own.
* ``slot(name)`` makes a sub-scope that closes at its position in the parent's
  undo order instead. Use it to reserve a place in the order for work that is
  registered later (a service built at start, tasks attached later).
* ``close(reason, timeout=...)`` is thread-safe and idempotent. ``timeout`` is
  the budget of the whole close; an undo still running at the deadline is left
  to finish in the background and named in ``CloseReport.pending``. A plain
  callable that runs inline cannot be interrupted, so keep those short.

Call ``close`` off the loop its undos hop to (an executor, another thread): an
undo that targets the closing thread's own loop can only be scheduled there,
and is reported as pending.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Coroutine, List, Optional, Tuple, Union

__all__ = ["CloseReport", "Scope", "ScopeClosed", "Undo"]

Undo = Callable[[], Optional[Awaitable[None]]]

logger = logging.getLogger("simplyprint.scope")

#: Budget for an undo that is run because it was registered on a closed scope.
_LATE_UNDO_TIMEOUT = 30.0
#: How long ``spawn`` waits for a running loop to create the task.
_SPAWN_HOP_TIMEOUT = 30.0

_OPEN = "open"
_CLOSING = "closing"
_CLOSED = "closed"


class ScopeClosed(RuntimeError):
    """A registration on a scope that is already closing or closed."""


@dataclass(frozen=True)
class CloseReport:
    """What one :meth:`Scope.close` left behind.

    ``errors`` holds every exception an undo raised, in the order they ran.
    ``pending`` names the undos still running when the close timed out, as
    ``"child/undo"`` paths relative to the closed scope.
    """

    errors: Tuple[BaseException, ...] = ()
    pending: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors and not self.pending


class _Callback:
    __slots__ = ("name", "fn", "loop")

    def __init__(
        self, name: str, fn: Undo, loop: Optional[asyncio.AbstractEventLoop]
    ) -> None:
        self.name = name
        self.fn = fn
        self.loop = loop


class _Spawned:
    __slots__ = ("name", "task")

    def __init__(self, name: str, task: "asyncio.Task[Any]") -> None:
        self.name = name
        self.task = task


class _Nested:
    __slots__ = ("name", "scope")

    def __init__(self, scope: "Scope") -> None:
        self.name = scope.name
        self.scope = scope


_Entry = Union[_Callback, _Spawned, _Nested]


def _label(undo: Callable[..., Any]) -> str:
    return getattr(undo, "__qualname__", None) or repr(undo)


def _running_on(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _live(loop: Optional[asyncio.AbstractEventLoop]) -> bool:
    return loop is not None and not loop.is_closed() and loop.is_running()


async def _invoke(fn: Undo) -> None:
    result = fn()
    if inspect.isawaitable(result):
        await result


async def _cancel_and_wait(task: "asyncio.Task[Any]") -> None:
    task.cancel()
    await asyncio.wait({task})


def _drive_detached(awaitable: Awaitable[Any], name: str) -> concurrent.futures.Future:
    """Drive ``awaitable`` to completion on a private loop in a helper thread."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    async def runner() -> Any:
        return await awaitable

    def main() -> None:
        try:
            future.set_result(asyncio.run(runner()))
        except BaseException as error:  # noqa: BLE001 -- handed to the waiter
            future.set_exception(error)

    threading.Thread(target=main, name="sp-scope-undo:%s" % name, daemon=True).start()
    return future


def _create_task(
    loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, Any], name: str
) -> "asyncio.Task[Any]":
    if _running_on(loop):
        return loop.create_task(coro, name=name)
    if loop.is_closed():
        coro.close()
        raise RuntimeError("cannot spawn %r: its loop is closed" % name)
    if not loop.is_running():
        # The loop has not started yet (startup): the task runs once it does.
        task = loop.create_task(coro, name=name)
        try:
            loop.call_soon_threadsafe(lambda: None)  # wake it if it just started
        except RuntimeError:
            pass
        return task

    created: concurrent.futures.Future = concurrent.futures.Future()

    def create() -> None:
        try:
            created.set_result(loop.create_task(coro, name=name))
        except BaseException as error:  # noqa: BLE001 -- handed to the caller
            created.set_exception(error)

    loop.call_soon_threadsafe(create)
    return created.result(timeout=_SPAWN_HOP_TIMEOUT)


class Scope:
    """Everything one mount registered, and how to undo it. See the module doc."""

    def __init__(
        self, name: str, *, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        self.name = name
        #: The loop :meth:`spawn` starts tasks on; children inherit it.
        self.loop = loop
        #: The ``reason`` given to :meth:`close`, readable by undos while they run.
        self.close_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._state = _OPEN
        self._closing_thread: Optional[int] = None
        self._done = threading.Event()
        self._entries: List[_Entry] = []
        self._children: List[Scope] = []
        self._parent: Optional[Scope] = None

    def __repr__(self) -> str:
        return "Scope(%r, %s)" % (self.name, self._state)

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has started."""
        return self._state != _OPEN

    # ------------------------------------------------------------------ register

    def on_close(
        self,
        undo: Undo,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        name: Optional[str] = None,
    ) -> None:
        """Run ``undo`` when the scope closes, on ``loop`` if one is given.

        On a scope that is already closing, the undo runs right away (so the
        thing it undoes cannot leak) and :class:`ScopeClosed` is raised.
        """
        entry = _Callback(name or _label(undo), undo, loop)
        with self._lock:
            if self._state == _OPEN:
                self._entries.append(entry)
                return
        errors: List[BaseException] = []
        pending: List[str] = []
        self._run(entry, time.monotonic() + _LATE_UNDO_TIMEOUT, errors, pending)
        for error in errors:
            logger.warning(
                "scope %s: late undo %s failed", self.name, entry.name, exc_info=error
            )
        raise ScopeClosed("scope %r is closed; %s was undone" % (self.name, entry.name))

    def spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> "asyncio.Task[Any]":
        """Start ``coro`` as a task on the scope's loop; close cancels it."""
        loop = self.loop
        if loop is None:
            coro.close()
            raise RuntimeError("scope %r has no loop to spawn %r on" % (self.name, name))
        if self.closed:
            coro.close()
            raise ScopeClosed("scope %r is closed; %r was not spawned" % (self.name, name))
        task = _create_task(loop, coro, name)
        entry = _Spawned(name, task)
        with self._lock:
            accepted = self._state == _OPEN
            if accepted:
                self._entries.append(entry)
        if not accepted:
            loop.call_soon_threadsafe(task.cancel)
            raise ScopeClosed("scope %r closed while spawning %r" % (self.name, name))
        task.add_done_callback(lambda _task: self._discard(entry))
        return task

    def child(self, name: str) -> "Scope":
        """A sub-scope closed before this scope's own undos (newest first)."""
        child = self._new_sub(name)
        with self._lock:
            if self._state != _OPEN:
                raise ScopeClosed("scope %r is closed" % self.name)
            self._children.append(child)
        return child

    def slot(self, name: str) -> "Scope":
        """A sub-scope closed at this position in the undo order.

        What is registered into it later still comes down where the slot was
        taken, not first: the order of a mount stays the order it was written.
        """
        sub = self._new_sub(name)
        with self._lock:
            if self._state != _OPEN:
                raise ScopeClosed("scope %r is closed" % self.name)
            self._entries.append(_Nested(sub))
        return sub

    def _new_sub(self, name: str) -> "Scope":
        sub = Scope(name, loop=self.loop)
        sub._parent = self
        return sub

    def _discard(self, entry: _Entry) -> None:
        with self._lock:
            try:
                self._entries.remove(entry)
            except ValueError:
                pass

    def _forget(self, sub: "Scope") -> None:
        with self._lock:
            if sub in self._children:
                self._children.remove(sub)
                return
            self._entries = [
                entry
                for entry in self._entries
                if not (isinstance(entry, _Nested) and entry.scope is sub)
            ]

    # --------------------------------------------------------------------- close

    def close(self, reason: str, *, timeout: float = 30.0) -> CloseReport:
        """Undo everything, children first, then own undos LIFO.

        Thread-safe. A second call is a no-op returning an empty report; one
        racing the first waits (up to ``timeout``) for it to finish.
        """
        me = threading.get_ident()
        with self._lock:
            state = self._state
            if state == _OPEN:
                self._state = _CLOSING
                self._closing_thread = me
                self.close_reason = reason
                children = list(reversed(self._children))
                entries = list(reversed(self._entries))
                self._children.clear()
                self._entries.clear()
            else:
                reentrant = self._closing_thread == me
        if state != _OPEN:
            if state == _CLOSING and not reentrant:
                self._done.wait(timeout)
            return CloseReport()

        deadline = time.monotonic() + max(timeout, 0.0)
        errors: List[BaseException] = []
        pending: List[str] = []
        try:
            for child in children:
                self._run(_Nested(child), deadline, errors, pending)
            for entry in entries:
                self._run(entry, deadline, errors, pending)
        finally:
            with self._lock:
                self._state = _CLOSED
            self._done.set()
            if self._parent is not None:
                self._parent._forget(self)

        report = CloseReport(tuple(errors), tuple(pending))
        if not report.ok:
            logger.warning(
                "scope %s closed (%s) with %d error(s); still running: %s",
                self.name,
                reason,
                len(report.errors),
                ", ".join(report.pending) or "nothing",
            )
        return report

    def _run(
        self,
        entry: _Entry,
        deadline: float,
        errors: List[BaseException],
        pending: List[str],
    ) -> None:
        if isinstance(entry, _Nested):
            remaining = max(0.0, deadline - time.monotonic())
            sub = entry.scope.close(self.close_reason or "parent closed", timeout=remaining)
            errors.extend(sub.errors)
            pending.extend("%s/%s" % (entry.name, label) for label in sub.pending)
            return
        if isinstance(entry, _Spawned):
            self._cancel(entry, deadline, errors, pending)
            return
        self._call(entry, deadline, errors, pending)

    def _cancel(
        self,
        entry: _Spawned,
        deadline: float,
        errors: List[BaseException],
        pending: List[str],
    ) -> None:
        task = entry.task
        if task.done():
            return
        loop = task.get_loop()
        if _running_on(loop):
            task.cancel()  # cannot wait for it on its own loop; it is unwinding
            return
        if not _live(loop):
            task.cancel()  # the loop is not running: nothing else touches it
            return
        future = asyncio.run_coroutine_threadsafe(_cancel_and_wait(task), loop)
        self._wait(future, entry.name, deadline, errors, pending)

    def _call(
        self,
        entry: _Callback,
        deadline: float,
        errors: List[BaseException],
        pending: List[str],
    ) -> None:
        loop = entry.loop
        if _live(loop) and not _running_on(loop):
            future = asyncio.run_coroutine_threadsafe(_invoke(entry.fn), loop)
            self._wait(future, entry.name, deadline, errors, pending)
            return
        try:
            result = entry.fn()
        except Exception as error:  # noqa: BLE001 -- isolated and reported
            self._record(entry.name, error, errors)
            return
        if not inspect.isawaitable(result):
            return
        if loop is not None and _running_on(loop):
            # Closing on the undo's own loop: it can only be scheduled there.
            task = asyncio.ensure_future(result)
            task.add_done_callback(lambda done: self._log_late(entry.name, done))
            pending.append(entry.name)
            return
        self._wait(
            _drive_detached(result, entry.name), entry.name, deadline, errors, pending
        )

    def _wait(
        self,
        future: concurrent.futures.Future,
        name: str,
        deadline: float,
        errors: List[BaseException],
        pending: List[str],
    ) -> None:
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError as error:
            # From 3.11 this is the builtin TimeoutError, which the undo itself
            # may have raised: only an unfinished future is a timeout.
            if future.done():
                self._record(name, error, errors)
            else:
                pending.append(name)
        except BaseException as error:  # noqa: BLE001 -- isolated and reported
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            self._record(name, error, errors)

    def _record(
        self, name: str, error: BaseException, errors: List[BaseException]
    ) -> None:
        logger.warning("scope %s: undo %s failed", self.name, name, exc_info=error)
        errors.append(error)

    def _log_late(self, name: str, task: "asyncio.Future[Any]") -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "scope %s: undo %s failed after close",
                self.name,
                name,
                exc_info=task.exception(),
            )
