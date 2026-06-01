"""Tests that :class:`Connection` drives the transport seam correctly.

Connection must build its socket only through the injected transport factory and
rebuild a *fresh* transport on every (re)connect attempt -- the property that
keeps the socket library swappable and the reconnect/version logic in
``Connection`` rather than in the transport.
"""

import asyncio
from unittest.mock import patch

import pytest

from simplyprint_ws_client.core.ws_protocol.connection import (
    Connection,
    ConnectionHint,
    ConnectionMode,
)
from simplyprint_ws_client.core.ws_protocol.events import (
    ConnectionEstablishedEvent,
)
from simplyprint_ws_client.shared.utils.backoff import ConstantBackoff

from tests._fakes import FakeTransport


def _connection_with_recording_factory():
    """A Connection whose factory records every transport it builds."""
    built = []

    def factory(logger):
        transport = FakeTransport(logger)
        built.append(transport)
        return transport

    conn = Connection(
        transport_factory=factory,
        hint=ConnectionHint(mode=ConnectionMode.SINGLE),
    )
    conn.use_running_loop()
    return conn, built


@pytest.mark.asyncio
async def test_connect_builds_transport_via_factory():
    conn, built = _connection_with_recording_factory()
    established = []
    conn.event_bus.on(ConnectionEstablishedEvent, lambda e: established.append(e.v))

    with patch(
        "simplyprint_ws_client.core.ws_protocol.connection.WsFirstMessageTimeout",
        10.0,
    ):
        await conn.connect()
        for _ in range(50):
            await asyncio.sleep(0.05)
            if built and built[0].is_open and conn.v == 0:
                break

        assert len(built) == 1, "exactly one transport built for the first connect"
        assert built[0].is_open
        assert conn.connected
        assert established == [0]

        await conn.disconnect()

    conn.stop()


@pytest.mark.asyncio
async def test_reconnect_builds_a_fresh_transport():
    conn, built = _connection_with_recording_factory()

    with (
        patch(
            "simplyprint_ws_client.core.ws_protocol.connection.WsFirstMessageTimeout",
            10.0,
        ),
        patch.object(ConstantBackoff, "delay", return_value=0.01),
    ):
        await conn.connect()
        for _ in range(50):
            await asyncio.sleep(0.05)
            if built and built[0].is_open and conn.v == 0:
                break
        assert len(built) == 1 and conn.v == 0

        # Drop the live socket -> Connection must rebuild via the factory.
        built[0].queue_close()
        for _ in range(100):
            await asyncio.sleep(0.05)
            if len(built) >= 2 and built[1].is_open:
                break

        assert len(built) >= 2, "reconnect must build a fresh transport, not reuse"
        assert conn.v == 1, "exactly one version bump for the single drop"
        assert built[1].is_open

        await conn.disconnect()

    conn.stop()
