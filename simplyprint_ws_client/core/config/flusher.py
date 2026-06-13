"""Coalesced, offloaded config persistence.

The ``ClientConfigChangedEvent`` listener used to call ``ConfigManager.flush()``
inline on the app loop -- a full re-serialize + ``fsync`` + ``os.replace`` per
field change. A burst (a reconnect minting tokens across N printers) meant N such
writes on the loop. :class:`ConfigFlusher` turns the listener into a cheap
:meth:`trigger`: a :class:`CoalescingTask` collapses the burst to one flush, run
on the ``io`` lane off the loop.

Only the chatty change path is coalesced. Registration (``add``/``remove``) keeps
its direct synchronous flush -- those are rare, user-driven, and must not be lost
to a crash inside the debounce window. The flush itself is unchanged (same atomic
tmp + ``os.replace`` + corrupt-file preservation); only *when* and *where* it runs
moves.
"""

from __future__ import annotations

from simplyprint_ws_client.common.asyncio.coalescing_task import CoalescingTask
from simplyprint_ws_client.common.asyncio.offload import Offload
from simplyprint_ws_client.core.config.manager import ConfigManager

__all__ = ["ConfigFlusher"]

#: Debounce window: a burst of change events within this collapses to one write.
DEFAULT_FLUSH_DELAY = 0.25


class ConfigFlusher:
    """One coalesced flusher for a single config manager."""

    def __init__(
        self,
        manager: ConfigManager,
        offload: Offload,
        *,
        delay: float = DEFAULT_FLUSH_DELAY,
        loop=None,
    ) -> None:
        self._manager = manager
        self._offload = offload
        self._task = CoalescingTask(self._flush_job, delay=delay, loop=loop)

    @property
    def dirty(self) -> bool:
        return self._task.dirty

    def trigger(self) -> None:
        """Mark the config dirty. Cheap, thread-safe; the listener calls this
        instead of flushing inline."""
        self._task.trigger()

    async def _flush_job(self) -> None:
        # Argument-less flush: every manager re-serializes its full set, so a
        # coalesced flush captures all pending changes in one atomic write.
        await self._offload.run_io(self._manager.flush)

    async def aclose(self) -> None:
        """Drain a pending flush once and stop. Call while the loop is alive."""
        await self._task.aclose()

    def flush_now_if_dirty(self) -> None:
        """Sync fallback for after the loop is dead (crash / never-ran): write any
        still-pending change on the caller thread so it is not lost."""
        if self._task.dirty:
            self._manager.flush()
