"""Regression coverage for the event-loop runner."""

import asyncio

import pytest

from simplyprint_ws_client.common.asyncio.event_loop_runner import Runner


def test_runner_propagates_exceptions():
    with pytest.raises(RuntimeError, match="boom"):
        with Runner():
            raise RuntimeError("boom")


def test_runner_returns_result():
    async def main():
        return 42

    with Runner() as runner:
        assert runner.run(main()) == 42


def test_runner_explicit_debug_false_is_respected():
    async def main():
        return asyncio.get_running_loop().get_debug()

    runner = Runner(debug=False)
    with runner:
        assert runner.run(main(), debug=False) is False
