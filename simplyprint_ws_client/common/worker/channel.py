"""Zero-copy frame transport between a worker subprocess and its parent.

A frame (a JPEG, tens-to-hundreds of KB) used to be pickled and copied through a
``multiprocessing.Queue`` -- measured at ~0.15 ms/frame, ~12x slower than it
needs to be. This channel instead keeps a ring of fixed-size **slabs** in one
``shared_memory`` block: the worker writes the JPEG straight into a free slab and
sends only a tiny fixed-width metadata record (slab index, length, producer id,
timestamp) down a SPSC :class:`~multiprocessing.Connection` pipe. The parent
wraps the slab as a :class:`SlabLease` -- a ``memoryview`` over the bytes, no
copy -- and frees the slab when done.

Lifetime rules that keep this safe:

* The **parent process owns** both the shared memory and the slab states, and is
  the only one that ``unlink`` s. A worker *attaches without tracking* the segment
  (see :func:`_attach_shm`) -- otherwise the child's ``resource_tracker`` would
  unlink the parent's segment at the child's exit (the well-known CPython
  ``shared_memory`` footgun) and spew "leaked" warnings.
* A slab is ``IN_USE`` from the moment the worker claims it until the parent's
  :meth:`SlabLease.release`; the worker never reclaims an in-use slab, so the
  parent's ``memoryview`` can never be overwritten underneath it.
* When no slab is free (the parent is briefly behind, holding leases), the worker
  **drops** the frame rather than block its decode loop or steal a held slab.
  Combined with the parent-side courier coalescing (latest-wins), the net
  behaviour is "newest frame wins" with bounded, copy-free memory.

A frame larger than a slab is rare; it falls back to a one-off ``send_bytes`` (a
single copy) and is logged so the slab size can be raised.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import struct
import threading
from multiprocessing import shared_memory
from typing import Any, Optional

__all__ = ["SlabLease", "SharedSlabChannel"]

_logger = logging.getLogger("worker.channel")

# slab_index(int32), length(uint32), producer_id(uint32), timestamp(float64)
_META = struct.Struct("<iIId")

# slab_index sentinels (never a real index)
_FAILED = -1  # the producer yielded no frame (error); data is None
_OVERSIZED = -2  # frame > slab_size; the payload bytes follow via send_bytes

_FREE = 0
_IN_USE = 1

_DEFAULT_SLABS = 24
_DEFAULT_SLAB_SIZE = 512 * 1024


def _attach_shm(name: str) -> "shared_memory.SharedMemory":
    """Open an EXISTING segment in a worker without tracking it.

    The parent owns the segment's lifetime and is the sole unlinker. A worker
    must not let its own ``resource_tracker`` learn about the segment -- otherwise
    that tracker would unlink the parent's segment at the worker's exit, and a
    second unregister (parent unlink + worker untrack of one name in the shared
    tracker daemon) raises a spurious ``KeyError``. So we prevent the worker from
    *registering* it at all, rather than registering then unregistering.

    Python 3.13+ exposes ``track=False`` for exactly this; on the 3.9 floor we
    neutralize ``resource_tracker.register`` for the duration of the attach.
    """
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # py3.13+
    except TypeError:
        pass

    from multiprocessing import resource_tracker

    original = resource_tracker.register

    def _skip_shared_memory(rname: str, rtype: str) -> None:
        if rtype == "shared_memory":
            return
        return original(rname, rtype)

    resource_tracker.register = _skip_shared_memory
    try:
        return shared_memory.SharedMemory(name=name)
    finally:
        resource_tracker.register = original


class SlabLease:
    """A borrowed view of one frame. Read :attr:`data`, then :meth:`release`.

    ``data`` is a zero-copy ``memoryview`` into a shared slab for the common case,
    or owned ``bytes`` for the oversized fallback, or ``None`` for a failed frame.
    Hold it only briefly: the slab cannot be reused until you release it.
    """

    __slots__ = (
        "producer_id",
        "timestamp",
        "_view",
        "_release_cb",
        "_data_bytes",
        "_released",
    )

    def __init__(
        self,
        producer_id: int,
        timestamp: float,
        *,
        view: Optional[memoryview] = None,
        release_cb: Optional[Any] = None,
        data_bytes: Optional[bytes] = None,
    ) -> None:
        self.producer_id = producer_id
        self.timestamp = timestamp
        self._view = view
        self._release_cb = release_cb
        self._data_bytes = data_bytes
        self._released = False

    @property
    def data(self):
        """The frame bytes (``memoryview`` / ``bytes``), or ``None`` if failed."""
        if self._view is not None:
            return self._view
        return self._data_bytes

    def to_bytes(self) -> Optional[bytes]:
        """A standalone copy of the frame (so the slab can be released)."""
        if self._view is not None:
            return bytes(self._view)
        return self._data_bytes

    def release(self) -> None:
        """Release the slab (idempotent). After this, :attr:`data` is invalid."""
        if self._released:
            return
        self._released = True
        if self._view is not None:
            try:
                self._view.release()
            except Exception:  # noqa: BLE001
                pass
            self._view = None
        if self._release_cb is not None:
            self._release_cb()
            self._release_cb = None

    def __len__(self) -> int:
        data = self.data
        return len(data) if data is not None else 0


class SharedSlabChannel:
    """A ring of shared-memory slabs + a SPSC pipe for one worker process.

    Build it in the parent with :meth:`create`, pass :meth:`child_args` to the
    worker, and rebuild it there with :meth:`attach`. The worker calls
    :meth:`send`; the parent's reader thread calls :meth:`recv`.
    """

    def __init__(
        self,
        *,
        owner: bool,
        data_shm: "shared_memory.SharedMemory",
        states: Any,  # multiprocessing.Array('b', n_slabs), has .get_lock()
        parent_conn: Any,
        child_conn: Any,
        n_slabs: int,
        slab_size: int,
    ) -> None:
        self._owner = owner
        self._data = data_shm
        self._states = states
        self._parent_conn = parent_conn
        self._child_conn = child_conn
        self._n_slabs = n_slabs
        self._slab_size = slab_size
        self._dropped = 0
        self._closed = False
        # Serializes pipe writes when several producer threads share one channel
        # (the camera worker's thread pool). Per-instance: the child's attached
        # channel has its own, independent of the parent's.
        self._send_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Construction / handoff.
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls, n_slabs: int = _DEFAULT_SLABS, slab_size: int = _DEFAULT_SLAB_SIZE
    ) -> "SharedSlabChannel":
        if n_slabs <= 0 or slab_size <= 0:
            raise ValueError("n_slabs and slab_size must be positive")
        data_shm = shared_memory.SharedMemory(create=True, size=n_slabs * slab_size)
        states = mp.Array("b", n_slabs)  # zero-initialised -> all FREE
        parent_conn, child_conn = mp.Pipe()  # duplex; worker sends, parent recvs
        return cls(
            owner=True,
            data_shm=data_shm,
            states=states,
            parent_conn=parent_conn,
            child_conn=child_conn,
            n_slabs=n_slabs,
            slab_size=slab_size,
        )

    def child_args(self) -> tuple:
        """A picklable bundle to rebuild this channel in the worker process."""
        return (
            self._data.name,
            self._states,
            self._child_conn,
            self._n_slabs,
            self._slab_size,
        )

    @classmethod
    def attach(cls, args: tuple) -> "SharedSlabChannel":
        """Rebuild the channel inside the worker process (the child side)."""
        data_name, states, child_conn, n_slabs, slab_size = args
        data_shm = _attach_shm(data_name)  # the parent owns this segment, not us
        return cls(
            owner=False,
            data_shm=data_shm,
            states=states,
            parent_conn=None,
            child_conn=child_conn,
            n_slabs=n_slabs,
            slab_size=slab_size,
        )

    @property
    def dropped(self) -> int:
        return self._dropped

    # ------------------------------------------------------------------ #
    # Worker side.
    # ------------------------------------------------------------------ #

    def send(self, producer_id: int, data: Optional[Any], timestamp: float) -> bool:
        """Hand a payload to the parent. ``False`` if it was dropped (slabs full).

        Safe to call concurrently from several producer threads in one process
        (the camera worker's thread pool): slab claims are serialized by the slab
        lock, and the pipe write is serialized by ``_send_lock`` -- a bare
        ``Connection.send_bytes`` is not safe under concurrent writers.
        """
        if data is None:
            with self._send_lock:
                self._child_conn.send_bytes(
                    _META.pack(_FAILED, 0, producer_id, timestamp)
                )
            return True

        length = len(data)
        if length > self._slab_size:
            _logger.warning(
                "payload %d bytes exceeds slab %d; using the one-off copy path",
                length,
                self._slab_size,
            )
            with self._send_lock:
                self._child_conn.send_bytes(
                    _META.pack(_OVERSIZED, length, producer_id, timestamp)
                )
                self._child_conn.send_bytes(bytes(data))
            return True

        idx = self._claim_slab()
        if idx is None:
            self._dropped += 1
            return False  # safe drop: parent-side coalescing keeps "latest wins"

        offset = idx * self._slab_size
        self._data.buf[offset : offset + length] = data  # parallel: distinct slabs
        with self._send_lock:
            self._child_conn.send_bytes(_META.pack(idx, length, producer_id, timestamp))
        return True

    def _claim_slab(self) -> Optional[int]:
        with self._states.get_lock():
            for i in range(self._n_slabs):
                if self._states[i] == _FREE:
                    self._states[i] = _IN_USE
                    return i
        return None

    # ------------------------------------------------------------------ #
    # Parent side.
    # ------------------------------------------------------------------ #

    def recv(self, timeout: Optional[float] = None) -> Optional[SlabLease]:
        """Block for the next frame and return a :class:`SlabLease`.

        Returns ``None`` on timeout, or when the pipe is closed / the worker has
        gone away (``EOFError`` / ``OSError``).
        """
        conn = self._parent_conn
        if conn is None:
            raise RuntimeError("recv is the parent side only")
        try:
            if timeout is not None and not conn.poll(timeout):
                return None
            meta = conn.recv_bytes()
        except (EOFError, OSError):
            return None

        slab, length, producer_id, timestamp = _META.unpack(meta)

        if slab == _FAILED:
            return SlabLease(producer_id, timestamp, data_bytes=None)

        if slab == _OVERSIZED:
            try:
                payload = conn.recv_bytes()
            except (EOFError, OSError):
                return None
            return SlabLease(producer_id, timestamp, data_bytes=payload)

        offset = slab * self._slab_size
        view = self._data.buf[offset : offset + length]
        return SlabLease(
            producer_id,
            timestamp,
            view=view,
            release_cb=lambda i=slab: self._free_slab(i),
        )

    def _free_slab(self, idx: int) -> None:
        with self._states.get_lock():
            self._states[idx] = _FREE

    # ------------------------------------------------------------------ #
    # Teardown.
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Release this end. The owner also unlinks the shared segment.

        All outstanding :class:`SlabLease` s must be released first (an alive
        ``memoryview`` blocks ``SharedMemory.close``).
        """
        if self._closed:
            return
        self._closed = True
        for conn in (self._parent_conn, self._child_conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self._data.close()
        except BufferError:
            # A lease's memoryview is still alive: close() refuses (so the buffer
            # is NOT unmapped -- the in-flight read stays valid). We must still
            # unlink below, or the segment leaks; the mapping is freed once the
            # last view is released and the object is finalized.
            _logger.warning("SharedSlabChannel.close: outstanding frame leases")
        if self._owner:
            try:
                self._data.unlink()  # always: the owner is the sole unlinker
            except FileNotFoundError:
                pass
