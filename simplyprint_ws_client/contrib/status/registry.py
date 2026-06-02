"""The status registry: producers publish, the status endpoint reads.

A brand- and producer-agnostic store of the latest computed status for each
*section* of the app. Periodic tasks (health, OTA, discovery, account-token
checks, ...) each own a section keyed by a string and publish a
:class:`StatusEntry` into it; a unified ``/service/status`` endpoint reads the
:meth:`~StatusRegistry.snapshot` and rolls the sections up to one top-level
state -- never doing the expensive work itself (the "compute on a schedule,
serve from cache" pattern).

The registry names no section and inspects no ``detail`` payload: a producer
owns its section key and the shape of its detail. There is no ``if section ==
...`` here, by design -- the same no-special-case rule as the brand boundary --
so a new producer (OTA today, something else tomorrow) is added without touching
this module.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class StatusState(str, Enum):
    """Coarse health of one section; worst-wins when rolled up to a top-level
    state.

    A ``str`` enum, so a value compares equal to its plain string and serialises
    straight to it (``StatusState.OK == "ok"``, ``json`` emits ``"ok"``).
    """

    OK = "ok"
    DEGRADED = "degraded"
    """Working, but a soft signal is off -- flapping, slow, stale-ish, retrying."""
    FAILING = "failing"
    """A critical signal is red, with a known cause."""
    UNKNOWN = "unknown"
    """Never reported, or the last report is older than its ttl (see
    :meth:`StatusEntry.is_stale`)."""


@dataclass(frozen=True)
class StatusEntry:
    """One producer's latest word on its section.

    ``detail`` is opaque to the registry -- producer-shaped, JSON-able data the
    endpoint passes through and the UI renders. ``updated_at`` is stamped by the
    producer (``time.time()``); ``ttl`` (seconds), when set, lets a reader treat
    a stale entry as :attr:`StatusState.UNKNOWN` rather than trusting old data.
    """

    section: str
    state: StatusState
    detail: Mapping[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0
    ttl: Optional[float] = None

    def is_stale(self, now: Optional[float] = None) -> bool:
        """True when ``ttl`` is set and the entry is older than it."""
        if self.ttl is None:
            return False
        return (time.time() if now is None else now) - self.updated_at > self.ttl


class StatusRegistry:
    """Thread-safe latest-value store, one entry per section.

    Publishing is last-writer-wins per section; reading returns a consistent
    point-in-time copy. Safe to write from the scheduler's loop/threads and read
    from the web request handler.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, StatusEntry] = {}
        self._lock = threading.Lock()

    def publish(self, entry: StatusEntry) -> None:
        """Replace the entry for ``entry.section``."""
        with self._lock:
            self._entries[entry.section] = entry

    def snapshot(self) -> Dict[str, StatusEntry]:
        """A shallow copy of every section's latest entry. The status endpoint
        serialises this and applies the staleness / roll-up policy."""
        with self._lock:
            return dict(self._entries)
