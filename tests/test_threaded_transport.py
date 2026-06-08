"""``ThreadedImpl`` -- the threaded, websocket-client-based printer wire
(distinct from the async :class:`WebSocket` ABC).

websocket-client is an optional extra, so these skip when it isn't installed.
The behavioural depth lives in the integrations that compose this transport;
here we just pin the pre-connect contract without opening a real socket.
"""

import logging

import pytest

pytest.importorskip("websocket")

from simplyprint_ws_client.contrib.connection.websocket import (  # noqa: E402
    ThreadedImpl,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState  # noqa: E402


def _make():
    return ThreadedImpl(
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


def test_stop_is_idempotent_before_start():
    t = _make()
    # The ``close`` alias was removed per S-close; ``stop()`` is the only
    # public shutdown method, and it must be safe with no socket/threads yet.
    t.stop()
    t.stop()
    assert t.connected is False


def test_no_close_alias():
    t = _make()
    # The backwards-compat ``close = stop`` alias was deleted per S-close;
    # ``stop()`` is the sole canonical shutdown method. Guard against the
    # alias creeping back in.
    assert not hasattr(ThreadedImpl, "close")
    assert callable(t.stop)
