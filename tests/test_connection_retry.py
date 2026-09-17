import asyncio

import pytest

from simplyprint_ws_client.core.ws_protocol.connection import Connection


class _FailingSession:
    def __init__(self):
        self.connect_calls = 0
        self.closed = False

    async def ws_connect(self, *_args, **_kwargs):
        self.connect_calls += 1
        raise RuntimeError("cannot schedule new futures after shutdown")

    async def close(self):
        self.closed = True


class _BlockingSession(_FailingSession):
    async def ws_connect(self, *_args, **_kwargs):
        self.connect_calls += 1
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_unexpected_connection_error_uses_reconnect_backoff(monkeypatch):
    class ShortBackoff:
        def delay(self):
            return 0.1

        def reset(self):
            pass

    monkeypatch.setattr(
        "simplyprint_ws_client.core.ws_protocol.connection.ConstantBackoff",
        ShortBackoff,
    )
    session = _FailingSession()
    connection = Connection(session=session, loop=asyncio.get_running_loop())

    await connection.connect()
    await asyncio.sleep(0.02)

    assert session.connect_calls == 1

    connection.stop()
    await asyncio.wait_for(connection, timeout=0.2)
    assert session.closed


@pytest.mark.asyncio
async def test_connection_cancellation_exits_and_closes_session():
    session = _BlockingSession()
    connection = Connection(session=session, loop=asyncio.get_running_loop())

    await connection.connect()
    await asyncio.sleep(0)
    connection._loop_task.cancel()
    await asyncio.wait_for(connection, timeout=0.2)

    assert session.connect_calls == 1
    assert session.closed
