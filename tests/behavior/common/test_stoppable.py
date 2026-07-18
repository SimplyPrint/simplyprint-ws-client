import asyncio
import threading

import pytest

from simplyprint_ws_client.common.utils.stoppable import (
    AsyncStoppable,
    ProcessStoppable,
    SyncStoppable,
)


def _stop_after_delay(stoppable) -> threading.Timer:
    timer = threading.Timer(0.01, stoppable.stop)
    timer.start()
    return timer


def test_sync_stop_clear_wait_and_timeout():
    stoppable = SyncStoppable()

    assert stoppable.is_stopped() is False
    assert stoppable.wait(0.001) is False

    timer = _stop_after_delay(stoppable)
    assert stoppable.wait(1) is True
    timer.join()

    assert stoppable.is_stopped() is True
    stoppable.clear()
    assert stoppable.is_stopped() is False


@pytest.mark.asyncio
async def test_async_stop_clear_wait_and_timeout():
    stoppable = AsyncStoppable()

    assert stoppable.is_stopped() is False
    assert await stoppable.wait(0.001) is False

    waiter = asyncio.create_task(stoppable.wait())
    await asyncio.sleep(0)
    stoppable.stop()
    assert await waiter is True

    assert stoppable.is_stopped() is True
    stoppable.clear()
    assert stoppable.is_stopped() is False


@pytest.mark.asyncio
async def test_async_wait_propagates_cancellation():
    stoppable = AsyncStoppable()
    waiter = asyncio.create_task(stoppable.wait())
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert stoppable.is_stopped() is False


def test_process_stop_clear_wait_and_timeout():
    stoppable = ProcessStoppable()

    assert stoppable.is_stopped() is False
    assert stoppable.wait(0.001) is False

    timer = _stop_after_delay(stoppable)
    assert stoppable.wait(1) is True
    timer.join()

    assert stoppable.is_stopped() is True
    stoppable.clear()
    assert stoppable.is_stopped() is False
