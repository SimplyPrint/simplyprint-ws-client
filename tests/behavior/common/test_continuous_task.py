"""Regression coverage for continuous task lifecycle behavior."""

import asyncio

import pytest

from simplyprint_ws_client.common.asyncio.continuous_task import ContinuousTask


@pytest.mark.asyncio
async def test_discard_of_cancelled_task_does_not_raise():
    task = ContinuousTask(lambda: asyncio.sleep(60), factory=asyncio.get_running_loop)
    task.schedule()
    task.cancel()

    # Let the cancellation be delivered so the task settles as cancelled.
    await asyncio.sleep(0)
    assert task.cancelled()

    task.discard()
    assert task.task is None
