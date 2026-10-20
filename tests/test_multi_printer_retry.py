from datetime import datetime, timedelta

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.client import ClientState
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.messages import MultiPrinterAddedMsg
from simplyprint_ws_client.core.protocol.models import ServerMsgType


@pytest.mark.asyncio
async def test_rate_limit_retry_waits_for_server_delay(monkeypatch, caplog):
    class Clock:
        current = datetime(2026, 1, 1)

        @classmethod
        def now(cls):
            return cls.current

    client = Client(PrinterConfig.get_new(), context=ClientContext())
    monkeypatch.setattr("simplyprint_ws_client.core.client.datetime", Clock)
    client.state = ClientState.PENDING_ADDED
    sent = []

    async def capture(message, *_args, **_kwargs):
        sent.append(message)

    monkeypatch.setattr(client, "send", capture)

    await client.event_bus.emit(
        ServerMsgType.ADD_CONNECTION,
        MultiPrinterAddedMsg(
            type=ServerMsgType.ADD_CONNECTION,
            data={
                "status": False,
                "unique_id": str(client.unique_id),
                "reason": "Rate limited. Retry-After: 249s",
            },
        ),
    )

    assert client.state == ClientState.NOT_CONNECTED
    assert "server requested retry in 249.0s" in caplog.text
    assert not await client.ensure_added(ConnectionMode.MULTI)
    assert sent == []

    Clock.current += timedelta(seconds=248)
    assert not await client.ensure_added(ConnectionMode.MULTI)
    assert sent == []

    Clock.current += timedelta(seconds=1, microseconds=1)
    assert not await client.ensure_added(ConnectionMode.MULTI)
    assert len(sent) == 1
