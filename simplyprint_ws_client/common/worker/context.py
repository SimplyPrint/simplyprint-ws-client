"""Where a producer runs -- the routing axis the camera/worker pool dispatches on.

The thing that forces a *subprocess* today is not "is the protocol async"; it is
"can this work cross a process boundary and does it burn CPU." So the substrate
routes on an explicit :class:`ExecutionContext`, derived from a protocol's
CPU-intensity + async-ness (with an explicit override), not from ``is_async``
alone:

* **PROCESS** -- CPU-heavy work (RTSP/OpenCV decode+encode). Its own subprocess;
  payloads ride zero-copy shared memory (see :mod:`.channel`).
* **THREAD** -- a light async (or blocking) producer in its own thread+loop;
  results are couriered back to the consumer loop.
* **INLINE** -- a light async producer run as a task *directly on* the consumer
  loop: no process, no thread, no IPC, no pickling.

Backpressure reuses the one :class:`OverflowPolicy` the courier defines, so
"latest wins" means the same thing on a frame channel as on an event queue.
"""

from __future__ import annotations

from enum import Enum, auto

from simplyprint_ws_client.common.asyncio.courier import OverflowPolicy

__all__ = ["ExecutionContext", "OverflowPolicy"]


class ExecutionContext(Enum):
    """The world a producer runs in. See the module docstring."""

    PROCESS = auto()
    THREAD = auto()
    INLINE = auto()
