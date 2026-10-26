"""The store's coalescing write queue.

Callers never write rows themselves. They :meth:`~Writer.mark` a row dirty
with a ``produce`` function that returns its current value, and the store's
writer thread later calls every pending ``produce`` and commits the results as
one transaction: one fsync for a burst of 500 token changes instead of 500.

When the batch goes out:

* a chatty row waits :data:`DEBOUNCE` seconds after its last mark, so a burst
  collapses to one write;
* but never more than :data:`CAP` seconds after its first mark, so a row that
  changes every half second is still saved;
* an ``urgent`` mark (adding or removing a printer, saving an account, a
  settings change, an import) is due at once.

When anything is due, everything pending goes in the same batch: the fsync is
the cost, not the rows.

``mark`` only takes a short lock, so it is safe and cheap on the event loop.
The timing is computed from an injected clock, so it is tested without
sleeping.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Writer", "Pending", "Produce", "SKIP", "DEBOUNCE", "CAP"]

#: Seconds a row waits after its last mark.
DEBOUNCE = 1.0
#: Seconds a row waits at most after its first mark.
CAP = 5.0

#: Returns the row's value columns, ``None`` to delete the row, or
#: :data:`SKIP` to leave it as stored. Called on the writer thread, at write
#: time, so it reads the newest value.
Produce = Callable[[], Any]


class _Skip:
    def __repr__(self) -> str:
        return "SKIP"


#: What a ``produce`` returns when, by write time, it has nothing to say about
#: the row (its owner forgot it): the row is neither written nor deleted.
SKIP: Any = _Skip()

#: ``(table, key)``; keys are tuples of the table's key columns.
RowId = Tuple[str, Tuple[Any, ...]]


@dataclass
class Pending:
    produce: Produce
    first: float
    last: float
    urgent: bool

    def due_at(self, debounce: float, cap: float) -> float:
        if self.urgent:
            return self.first
        return min(self.last + debounce, self.first + cap)


class Writer:
    """The dirty map. Thread-safe; owns no thread itself (the store's runs it)."""

    def __init__(
        self,
        *,
        debounce: float = DEBOUNCE,
        cap: float = CAP,
        clock: Callable[[], float] = time.monotonic,
        wake: Optional[Callable[[], None]] = None,
    ) -> None:
        self.debounce = debounce
        self.cap = cap
        self.clock = clock
        self._lock = threading.Lock()
        self._dirty: Dict[RowId, Pending] = {}
        #: Called after a mark that may move the next deadline earlier, so a
        #: sleeping writer thread recomputes it.
        self._wake = wake
        #: Set by the store: runs a batch now, on the calling thread.
        self._flush: Optional[Callable[[], int]] = None
        #: Set by the store: a key as its table's key tuple (and a check that
        #: the table exists), so ``"uid"`` and ``("uid",)`` are one row.
        self._key_of: Optional[Callable[[str, Any], Tuple[Any, ...]]] = None

    def mark(
        self,
        table: str,
        key: Any,
        produce: Produce,
        *,
        urgent: bool = False,
    ) -> None:
        """Note that row ``key`` of ``table`` changed. The newest ``produce``
        wins; the first mark's time is kept, so the cap still applies."""
        if self._key_of is not None:
            key = self._key_of(table, key)
        now = self.clock()
        row = (table, key)
        with self._lock:
            pending = self._dirty.get(row)
            if pending is None:
                self._dirty[row] = Pending(produce, now, now, urgent)
            else:
                pending.produce = produce
                pending.last = now
                pending.urgent = pending.urgent or urgent
        if self._wake is not None:
            self._wake()

    @property
    def dirty(self) -> bool:
        with self._lock:
            return bool(self._dirty)

    def pending(self) -> List[RowId]:
        with self._lock:
            return list(self._dirty)

    def next_deadline(self) -> Optional[float]:
        """When the next batch is due (clock time), ``None`` when nothing is
        pending."""
        with self._lock:
            if not self._dirty:
                return None
            return min(
                p.due_at(self.debounce, self.cap) for p in self._dirty.values()
            )

    def is_due(self, now: Optional[float] = None) -> bool:
        deadline = self.next_deadline()
        if deadline is None:
            return False
        return (self.clock() if now is None else now) >= deadline

    def take(self) -> Dict[RowId, Pending]:
        """Hand every pending row to the caller (the batch) and start a fresh
        map; marks made while the batch is written go to the next one."""
        with self._lock:
            batch, self._dirty = self._dirty, {}
            return batch

    def restore(self, batch: Dict[RowId, Pending]) -> None:
        """Put back a batch that failed to commit. A row marked again since
        keeps its newer ``produce``, and the older first-mark time, so the cap
        still counts from the first change."""
        with self._lock:
            for row, pending in batch.items():
                newer = self._dirty.get(row)
                if newer is None:
                    self._dirty[row] = pending
                else:
                    newer.first = min(newer.first, pending.first)
                    newer.urgent = newer.urgent or pending.urgent

    def flush_now(self) -> int:
        """Write everything pending now, on the calling thread; returns the
        rows that changed. Blocks for the fsync: callers that need the row on
        disk before they return (adding a printer) use it, the event loop does
        not."""
        if self._flush is None:
            raise RuntimeError("writer is not attached to a store")
        return self._flush()
