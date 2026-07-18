import pytest

from simplyprint_ws_client.common.hardware import physical_machine
from simplyprint_ws_client.common.hardware.physical_machine import (
    make_host_telemetry_reader,
)


@pytest.mark.asyncio
async def test_host_telemetry_cache_is_owned_by_each_reader(monkeypatch):
    async def inline_to_thread(fn):
        return fn()

    monkeypatch.setattr(physical_machine.asyncio, "to_thread", inline_to_thread)
    now = {"first": 10.0, "second": 10.0}
    calls = {"first": 0, "second": 0}

    def usage(name, value):
        def read():
            calls[name] += 1
            return {"cpu": value, "memory": calls[name]}

        return read

    first = make_host_telemetry_reader(
        read_usage=usage("first", 11), clock=lambda: now["first"]
    )
    second = make_host_telemetry_reader(
        read_usage=usage("second", 22), clock=lambda: now["second"]
    )

    assert await first() == {"cpu": 11, "memory": 1}
    assert await first() == {"cpu": 11, "memory": 1}
    assert await second() == {"cpu": 22, "memory": 1}
    assert calls == {"first": 1, "second": 1}

    now["first"] += 5.0
    assert await first() == {"cpu": 11, "memory": 2}
    assert await second() == {"cpu": 22, "memory": 1}
    assert calls == {"first": 2, "second": 1}
