"""Brand-free tests for the zero-copy :class:`SharedSlabChannel`.

These pin the slab lifecycle the whole camera IPC win rests on: round-trip
fidelity, slab claim/recycle, drop-when-full (never block, never corrupt a held
lease), the oversized fallback, failed frames, no shared-memory leak after close,
and a real cross-process round-trip.
"""

import multiprocessing as mp
import os

import pytest

from simplyprint_ws_client.shared.worker.channel import SharedSlabChannel

_SHM_DIR = "/dev/shm"


def _shm_segments():
    if not os.path.isdir(_SHM_DIR):
        return None
    return {n for n in os.listdir(_SHM_DIR) if n.startswith("psm_")}


def test_roundtrip_preserves_frame_bytes():
    channel = SharedSlabChannel.create(n_slabs=4, slab_size=4096)
    try:
        payload = bytes(range(200))
        assert channel.send(producer_id=7, data=payload, timestamp=1.5) is True

        lease = channel.recv(timeout=1.0)
        assert lease is not None
        assert lease.producer_id == 7
        assert lease.timestamp == 1.5
        assert bytes(lease.data) == payload
        assert lease.to_bytes() == payload
        lease.release()
    finally:
        channel.close()


def test_slabs_are_recycled_across_many_frames():
    channel = SharedSlabChannel.create(n_slabs=2, slab_size=1024)
    try:
        for i in range(20):  # far more frames than slabs
            data = bytes([i % 256]) * 100
            assert channel.send(1, data, float(i)) is True
            lease = channel.recv(timeout=1.0)
            assert lease is not None
            assert bytes(lease.data) == data
            lease.release()  # frees the slab for the next iteration
        assert channel.dropped == 0
    finally:
        channel.close()


def test_drop_when_full_then_recover_after_release():
    channel = SharedSlabChannel.create(n_slabs=2, slab_size=1024)
    try:
        assert channel.send(1, b"a" * 10, 0.0) is True  # slab 0
        assert channel.send(1, b"b" * 10, 1.0) is True  # slab 1 -> full

        assert channel.send(1, b"c" * 10, 2.0) is False  # dropped
        assert channel.dropped == 1

        lease = channel.recv(timeout=1.0)  # frame "a"
        assert bytes(lease.data) == b"a" * 10
        lease.release()  # frees a slab

        assert channel.send(1, b"d" * 10, 3.0) is True  # room again
    finally:
        channel.close()


def test_oversized_frame_uses_the_copy_fallback():
    channel = SharedSlabChannel.create(n_slabs=2, slab_size=512)
    try:
        big = bytes(2000)  # larger than a slab
        assert channel.send(3, big, 9.0) is True

        lease = channel.recv(timeout=1.0)
        assert lease is not None
        assert lease.to_bytes() == big
        assert channel.dropped == 0  # the fallback is not a drop
        lease.release()  # no-op for a heap-backed lease
    finally:
        channel.close()


def test_failed_frame_yields_none():
    channel = SharedSlabChannel.create(n_slabs=2, slab_size=512)
    try:
        assert channel.send(5, None, 4.0) is True
        lease = channel.recv(timeout=1.0)
        assert lease is not None
        assert lease.producer_id == 5
        assert lease.data is None
        assert lease.to_bytes() is None
        lease.release()
    finally:
        channel.close()


@pytest.mark.skipif(_shm_segments() is None, reason="no /dev/shm to inspect")
def test_close_leaves_no_shared_memory_behind():
    before = _shm_segments()
    channel = SharedSlabChannel.create(n_slabs=4, slab_size=4096)
    during = _shm_segments()
    assert during - before  # a new segment exists while open

    lease = None
    channel.send(1, b"x" * 50, 0.0)
    lease = channel.recv(timeout=1.0)
    lease.release()
    channel.close()

    after = _shm_segments()
    assert after - before == set()  # nothing leaked


@pytest.mark.skipif(_shm_segments() is None, reason="no /dev/shm to inspect")
def test_close_unlinks_even_with_an_outstanding_lease():
    # Regression: close() must unlink the segment even when a lease still holds a
    # memoryview into it (close() raises BufferError; unlink must still run).
    before = _shm_segments()
    channel = SharedSlabChannel.create(n_slabs=4, slab_size=4096)
    channel.send(1, b"held" * 10, 0.0)
    lease = channel.recv(timeout=1.0)
    assert lease is not None

    channel.close()  # lease (and its memoryview) is still alive

    assert _shm_segments() - before == set()  # not leaked despite the live view
    lease.release()


def _frame_writer(child_args, count, slab_size):
    """Worker entrypoint: attach and write `count` distinct frames, then close."""
    channel = SharedSlabChannel.attach(child_args)
    for i in range(count):
        channel.send(42, bytes([i % 256]) * (slab_size // 2), float(i))
    channel.close()


def test_cross_process_roundtrip():
    count = 40
    slab_size = 2048
    # More slabs than frames so the worker never has to drop while the parent
    # consumes -- this test is about IPC fidelity, not backpressure.
    channel = SharedSlabChannel.create(n_slabs=count + 8, slab_size=slab_size)
    before = _shm_segments()
    proc = mp.Process(
        target=_frame_writer, args=(channel.child_args(), count, slab_size)
    )
    proc.start()
    try:
        received = []
        for _ in range(count):
            lease = channel.recv(timeout=5.0)
            assert lease is not None
            received.append(bytes(lease.data))
            lease.release()

        assert len(received) == count
        for i, data in enumerate(received):
            assert data == bytes([i % 256]) * (slab_size // 2)

        proc.join(5.0)
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
        channel.close()

    if before is not None:
        assert _shm_segments() - before == set()  # worker attach did not leak
