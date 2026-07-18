import threading

from pydantic import PrivateAttr


class Synchronized:
    """Simple synchronization mixin using threading.Lock, with 1-user deadlock detection."""

    __lock: threading.Lock = PrivateAttr(...)
    __lock_owner: int = PrivateAttr(...)

    def __init__(self, *args, **kwargs):
        self.__lock = threading.Lock()
        self.__lock_owner = -1

    def __enter__(self):
        ident = threading.get_ident()
        if self.__lock.locked() and self.__lock_owner == ident:
            raise RuntimeError("Deadlock detected: re-entrant locking is not allowed.")
        self.__lock.acquire()
        self.__lock_owner = ident
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__lock_owner = -1
        self.__lock.release()
