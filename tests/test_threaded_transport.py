"""``ThreadedWebSocketTransport`` -- the threaded, websocket-client-based printer
transport (distinct from the async :class:`WebSocketTransport` ABC).

websocket-client is an optional extra, so these skip when it isn't installed.
The behavioural depth lives in the integrations that compose this transport;
here we just pin the pre-connect contract without opening a real socket.
"""

import logging

import pytest

pytest.importorskip("websocket")

from simplyprint_ws_client.contrib.transport import (  # noqa: E402
    ThreadedWebSocketTransport,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState  # noqa: E402


def _make():
    return ThreadedWebSocketTransport(
        "ws://127.0.0.1:9/ws",
        logger=logging.getLogger("test-ws"),
        on_message=lambda _msg: None,
    )


def test_not_connected_before_start():
    t = _make()
    assert t.connected is False
    assert t.state is ConnectionState.OFFLINE


def test_send_returns_false_when_not_connected():
    t = _make()
    assert t.send("hello") is False


def test_close_is_idempotent_before_start():
    t = _make()
    # close is an alias for stop; both must be safe with no socket/threads yet.
    t.close()
    t.stop()
    assert t.connected is False
