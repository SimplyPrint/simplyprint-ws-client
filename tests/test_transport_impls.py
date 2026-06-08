"""Contract tests for every concrete :class:`WebSocket` impl, against a real server.

Both shipped implementations -- WebsocketsImpl (default) and AiohttpImpl -- must
satisfy the same contract: connect/send/recv/close roundtrip, connect failure ->
WebSocketError, peer close -> WebSocketClosed, idempotent close. Parametrizing over
both is what proves the abstraction is real.
"""

import pytest
from websockets.asyncio.server import serve

from simplyprint_ws_client.contrib.connection.websocket import (
    AiohttpImpl,
    WebsocketsImpl,
)
from simplyprint_ws_client.contrib.connection.websocket.base import (
    WebSocketClosed,
    WebSocketError,
)

TRANSPORTS = [WebsocketsImpl, AiohttpImpl]
IDS = ["websockets", "aiohttp"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_cls", TRANSPORTS, ids=IDS)
async def test_roundtrip_connect_send_recv_close(transport_cls):
    async def echo(ws):
        async for message in ws:
            await ws.send(message)

    async with serve(echo, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]

        transport = transport_cls()
        await transport.connect(f"ws://127.0.0.1:{port}")
        assert transport.is_open

        await transport.send("hello")
        assert await transport.recv() == "hello"

        await transport.close()
        assert not transport.is_open


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_cls", TRANSPORTS, ids=IDS)
async def test_connect_failure_raises_transport_error(transport_cls):
    transport = transport_cls()
    with pytest.raises(WebSocketError):
        # Nothing is listening on port 1.
        await transport.connect("ws://127.0.0.1:1", open_timeout=0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_cls", TRANSPORTS, ids=IDS)
async def test_recv_raises_transport_closed_on_peer_close(transport_cls):
    async def closer(ws):
        await ws.close()

    async with serve(closer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]

        transport = transport_cls()
        await transport.connect(f"ws://127.0.0.1:{port}")

        with pytest.raises(WebSocketClosed):
            # Drain until the close handshake surfaces.
            for _ in range(20):
                await transport.recv()

        await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_cls", TRANSPORTS, ids=IDS)
async def test_close_is_idempotent_and_never_raises(transport_cls):
    transport = transport_cls()
    # Never connected -> close is a no-op, not an error.
    await transport.close()
    await transport.close()
    assert not transport.is_open
