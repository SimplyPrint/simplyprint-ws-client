"""Generic worker/IPC substrate -- run producers in another world, deliver home.

The reusable engine the camera pool is being rebuilt on (and that any future
heavy producer can reuse): pick where a producer runs (:class:`ExecutionContext`)
and the substrate delivers its items back onto the consumer loop -- zero-copy
across a process boundary via :class:`SharedSlabChannel`, or via the courier for
a thread, or directly for an inline task.

This package is brand-free and imports only ``shared`` leaves, so it sits below
both ``core`` and ``contrib`` in the import DAG.
"""

from simplyprint_ws_client.common.worker.context import ExecutionContext, OverflowPolicy
from simplyprint_ws_client.common.worker.channel import SlabLease, SharedSlabChannel

__all__ = [
    "ExecutionContext",
    "OverflowPolicy",
    "SlabLease",
    "SharedSlabChannel",
]
