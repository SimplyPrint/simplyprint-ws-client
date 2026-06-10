"""Tests that :class:`CloudConnection` composes protocol over a transport."""

import asyncio
from unittest.mock import patch

import pytest

from simplyprint_ws_client.core.protocol import connection as conn_mod
from simplyprint_ws_client.core.protocol.connection import (
    CloudConnection,
    ConnectionHint,
    ConnectionMode,
)
from simplyprint_ws_client.core.protocol.events import CloudConnectionEstablishedEvent
from simplyprint_ws_client.common.utils.backoff import ConstantBackoff

from tests._fakes import FakeTransport


def _connection_with_recording_factory():
    built = []

    def factory(url, provider, logger):
        transport = FakeTransport(
            url,
            provider,
            logger,
            first_message_timeout=conn_mod.WsFirstMessageTimeout,
        )
        built.append(transport)
        return transport

    conn = CloudConnection(
        transport_factory=factory,
        hint=ConnectionHint(mode=ConnectionMode.SINGLE),
    )
    conn.use_running_loop()
    return conn, built


@pytest.mark.asyncio
async def test_connect_builds_transport_via_factory_once():
    conn, built = _connection_with_recording_factory()
    established = []
    conn.event_bus.on(
        CloudConnectionEstablishedEvent, lambda e: established.append(e.v)
    )

    with patch(
        "simplyprint_ws_client.core.protocol.connection.WsFirstMessageTimeout", 10.0
    ):
        await conn.connect()
        await asyncio.sleep(0.05)

        assert len(built) == 1
        assert built[0].connected
        assert conn.connected
        assert established == [0]

        await conn.disconnect()

    conn.stop()


@pytest.mark.asyncio
async def test_transport_reconnect_advances_protocol_version():
    conn, built = _connection_with_recording_factory()

    with (
        patch(
            "simplyprint_ws_client.core.protocol.connection.WsFirstMessageTimeout",
            10.0,
        ),
        patch.object(ConstantBackoff, "delay", return_value=0.01),
    ):
        await conn.connect()
        await asyncio.sleep(0.05)
        assert len(built) == 1 and conn.v == 0

        built[0].queue_close()
        for _ in range(100):
            await asyncio.sleep(0.05)
            if built[0].connect_calls >= 2 and conn.v == 1:
                break

        assert len(built) == 1
        assert built[0].connect_calls >= 2
        assert conn.v == 1
        assert built[0].connected

        await conn.disconnect()

    conn.stop()
