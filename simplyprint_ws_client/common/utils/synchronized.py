import threading
from typing import Callable, TypeVar

from pydantic import PrivateAttr

T = TypeVar("T")


class _Lazy(threading.local):
    #: Within :func:`locked_on_first_use` on this thread.
    on = False


_LAZY = _Lazy()


def locked_on_first_use(build: Callable[[], T]) -> T:
    """``build()``, every reactive model it constructs (on this thread)
    making its lock on its first ``with`` instead of with the model, as
    :meth:`Synchronized.lazily` does. The same locking either way: the first
    enter creates the lock atomically. For state built per printer whose
    models are rarely or never locked."""
    if _LAZY.on:
        return build()
    _LAZY.on = True
    try:
        return build()
    finally:
        _LAZY.on = False


class Synchronized:
    """Simple synchronization mixin using threading.Lock, with 1-user deadlock detection."""

    __lock: threading.Lock = PrivateAttr(...)
    __lock_owner: int = PrivateAttr(...)

    def __init__(self, *args, **kwargs):
        # Stored directly: on a pydantic model an assignment would go through
        # its ``__setattr__``, which never memoizes these (undeclared private)
        # names, on every state model built. Where they end up is the same.
        object.__setattr__(self, "_Synchronized__lock", threading.Lock())
        object.__setattr__(self, "_Synchronized__lock_owner", -1)

    def __enter__(self):
        ident = threading.get_ident()
        lock = self.__dict__.get("_Synchronized__lock")
        if lock is None:
            lock = self._Synchronized__first_lock()
        if lock.locked() and self.__lock_owner == ident:
            raise RuntimeError("Deadlock detected: re-entrant locking is not allowed.")
        lock.acquire()
        self.__lock_owner = ident
        return self

    def __first_lock(self) -> threading.Lock:
        """The lock of an instance built without one (``lazily()``), made on
        its first use. ``setdefault`` is one atomic step: two threads entering
        at once get the same lock."""
        lock = self.__dict__.setdefault("_Synchronized__lock", threading.Lock())
        self.__dict__.setdefault("_Synchronized__lock_owner", -1)
        return lock

    @staticmethod
    def lazily(instance: object) -> None:
        """``__init__`` for an instance that is rarely, if ever, locked (a
        printer's many small entries): its lock is made on first use."""
        object.__setattr__(instance, "_Synchronized__lock_owner", -1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__lock_owner = -1
        self.__lock.release()
