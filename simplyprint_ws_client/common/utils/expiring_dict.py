from __future__ import annotations

import threading
from typing import Any, Iterator, List, Optional, Tuple
from time import time


class ExpiringDict:
    """A dictionary whose keys have a TTL.

    Composes over a private dict (instead of subclassing it) so no unproxied
    ``dict`` method can bypass the lock or leak the raw ``(value, expiry)``
    tuples -- the only available operations are the lock-guarded ones below.

    Thread-safe: the discovery backends write from their own loop thread while
    the web/onboarding and client threads read concurrently, so every public
    operation (and the lazy clean it triggers) is guarded by a single lock. The
    real hazard without it is ``_clean`` deleting entries while another thread
    iterates the same dict.
    """

    def __init__(self, ttl: int = 300, scan_interval: Optional[float | int] = None):
        self.ttl = ttl
        self.scan_interval = scan_interval or ttl / 2
        self.last_clean = time()
        self._data: dict = {}
        self._lock = threading.Lock()

    def _clean_locked(self) -> None:
        # Caller must hold ``self._lock``.
        now = time()

        if now - self.last_clean < self.scan_interval:
            return

        self.last_clean = now

        for key, value in list(self._data.items()):
            if value[1] < now:
                del self._data[key]

    def __setitem__(self, __key: Any, __value: Any) -> None:
        with self._lock:
            self._clean_locked()
            self._data[__key] = (__value, time() + self.ttl)

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            self._clean_locked()
            return self._data[key][0]

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            self._clean_locked()
            del self._data[key]

    def __contains__(self, __key: object) -> bool:
        with self._lock:
            self._clean_locked()
            return __key in self._data

    def __len__(self) -> int:
        with self._lock:
            self._clean_locked()
            return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        # Iterate a snapshot, so a concurrent writer can never invalidate it.
        with self._lock:
            self._clean_locked()
            return iter(list(self._data.keys()))

    def get(self, key: Any, default=None) -> Any:
        with self._lock:
            self._clean_locked()
            return self._data.get(key, (default,))[0]

    def keys(self) -> List[Any]:
        with self._lock:
            self._clean_locked()
            return list(self._data.keys())

    def items(self) -> List[Tuple[Any, Any]]:
        with self._lock:
            self._clean_locked()
            return [(k, v[0]) for k, v in self._data.items()]

    def values(self) -> List[Any]:
        with self._lock:
            self._clean_locked()
            return [v[0] for v in self._data.values()]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
