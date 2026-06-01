"""Small concurrency helpers shared across integrations."""

from __future__ import annotations

__all__ = ["start_in_thread"]

import asyncio
import threading
from typing import Coroutine, Optional


def start_in_thread(coro: Coroutine, *, name: Optional[str] = None) -> threading.Thread:
    """Run ``coro`` to completion on a fresh event loop in a background thread.

    Used by the per-printer file flows to kick off a download/upload without
    blocking the caller. Returns the started thread so the caller can join or
    track it (threads are non-daemon so shutdown joins them).
    """
    thread = threading.Thread(target=asyncio.run, args=(coro,), name=name)
    thread.start()
    return thread
