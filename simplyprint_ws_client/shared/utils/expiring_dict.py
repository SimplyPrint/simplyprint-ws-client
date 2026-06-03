from __future__ import annotations

import threading
from typing import Any, List, Optional, Tuple
from time import time


class ExpiringDict(dict):
    """
    A Dictionary with keys that have a TTL.
    This is a minimal implementation and does not proxy all methods.

    Thread-safe: the discovery backends write from their own loop thread while
    the web/onboarding and client threads read concurrently, so every public
    operation (and the lazy clean it triggers) is guarded by a single lock. The
    real hazard without it is ``_clean`` deleting entries while another thread
    iterates the same dict.
    """

    def __init__(self, ttl: int = 300, scan_interval: Optional[float | int] = None):
        super().__init__()
        self.ttl = ttl
        self.scan_interval = scan_interval or ttl / 2
        self.last_clean = time()
        self._lock = threading.Lock()

    def _clean_locked(self) -> None:
        # Caller must hold ``self._lock``.
        now = time()

        if now - self.last_clean < self.scan_interval:
            return

        self.last_clean = now

        for key, value in list(super().items()):
            if value[1] < now:
                super().__delitem__(key)

    def __setitem__(self, __key: Any, __value: Any) -> None:
        with self._lock:
            self._clean_locked()
            super().__setitem__(__key, (__value, time() + self.ttl))

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            self._clean_locked()
            return super().__getitem__(key)[0]

    def __contains__(self, __key: object) -> bool:
        with self._lock:
            self._clean_locked()
            return super().__contains__(__key)

    def get(self, key: Any, default=None) -> Any:
        with self._lock:
            self._clean_locked()
            return super().get(key, (default,))[0]

    def items(self) -> List[Tuple[Any, Any]]:
        with self._lock:
            self._clean_locked()
            return [(k, v[0]) for k, v in super().items()]

    def values(self) -> List[Any]:
        with self._lock:
            self._clean_locked()
            return [v[0] for v in super().values()]
