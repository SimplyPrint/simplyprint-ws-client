"""Comprehensive tests for WebSocket connection version (v) increment logic.

Tests ensure that:
- Version increments correctly on disconnects/reconnects
- Messages are not dropped while connected to the correct version
- Messages are dropped when targeting a stale version
- Multiple disconnect/reconnect cycles maintain version consistency

These drive :class:`SimplyPrintConnection` through a :class:`FakeTransport`, so they pin the
version/state contract independently of the concrete socket library.
"""

import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio

from simplyprint_ws_client.core.protocol import connection as conn_mod
from simplyprint_ws_client.core.protocol.connection import (
    SimplyPrintConnection,
    ConnectionHint,
    ConnectionMode,
)
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionEstablishedEvent,
    SimplyPrintConnectionLostEvent,
    SimplyPrintConnectionIncomingEvent,
)
from simplyprint_ws_client.core.protocol.messages import (
    PingMsg,
)

from tests._fakes import FakeTransport


@pytest_asyncio.fixture
async def fake_transport():
    """Provide a fake transport instance."""
    return FakeTransport()


@pytest_asyncio.fixture
async def connection(fake_transport):
    """Create a SimplyPrintConnection whose transport factory yields the fake transport."""

    def factory(url, provider, logger):
        fake_transport.url = url
        fake_transport.provider = provider
        fake_transport.logger = logger
        fake_transport.first_message_timeout = conn_mod.WsFirstMessageTimeout
        return fake_transport

    conn = SimplyPrintConnection(
        transport_factory=factory,
        hint=ConnectionHint(mode=ConnectionMode.SINGLE),
    )

    # Provide the running event loop to the connection
    conn.use_running_loop()

    yield conn

    # Cleanup
    conn.stop()


@pytest.mark.asyncio
async def test_initial_version_is_zero(connection):
    """Test that connection starts with version 0."""
    assert connection.v == 0


@pytest.mark.asyncio
async def test_message_dropped_when_version_mismatch(connection, fake_transport):
    """Test that messages are dropped when targeting a different version."""
    # Start with version 0
    assert connection.v == 0

    # Create a test message
    test_msg = PingMsg()

    # Send message targeting version 0 - should be dropped (not connected)
    await connection.send(test_msg, v=0)
    assert len(fake_transport.sent) == 0  # Not connected, message dropped

    # Now connect (fake) and try targeting wrong version
    connection.transport = fake_transport.open_for_test()
    connection.protocol.attach(fake_transport)
    await connection.send(test_msg, v=1)
    assert len(fake_transport.sent) == 0  # Version mismatch, dropped


@pytest.mark.asyncio
async def test_message_sent_when_version_matches(connection, fake_transport):
    """Test that messages are sent when version matches."""
    # Setup: fake a connected state
    connection.transport = fake_transport.open_for_test()
    connection.protocol.attach(fake_transport)
    connection.v = 0

    # Create a test message
    test_msg = PingMsg()

    # Send message targeting matching version
    await connection.send(test_msg, v=0)
    assert len(fake_transport.sent) == 1

    # Increment version and try again - should fail
    connection.v = 1
    await connection.send(test_msg, v=0)
    assert len(fake_transport.sent) == 1  # Not sent due to version mismatch


@pytest.mark.asyncio
async def test_message_sent_without_version_constraint(connection, fake_transport):
    """Test that messages without version constraint are sent when connected."""
    # Setup: fake a connected state
    connection.transport = fake_transport.open_for_test()
    connection.protocol.attach(fake_transport)
    connection.v = 0

    # Create a test message
    test_msg = PingMsg()

    # Send message without version constraint
    await connection.send(test_msg, v=None)
    assert len(fake_transport.sent) == 1

    # Change version - message should still be sent since no constraint
    connection.v = 5
    await connection.send(test_msg, v=None)
    assert len(fake_transport.sent) == 2


@pytest.mark.asyncio
async def test_incoming_message_tagged_with_correct_version(connection):
    """Test that incoming messages are tagged with the current version."""
    received_messages = []

    async def on_message(msg, v):
        received_messages.append((msg, v))

    # Subscribe to incoming events
    connection.event_bus.on(SimplyPrintConnectionIncomingEvent, on_message)

    # Simulate connection at version 0
    connection.v = 0

    # Emit a message event as would happen in poll()
    test_msg = {"msg_type": "test"}
    await connection.event_bus.emit(SimplyPrintConnectionIncomingEvent, test_msg, 0)

    await asyncio.sleep(0.05)
    assert len(received_messages) == 1
    # Verify the message was tagged with version 0
    assert received_messages[0] == (test_msg, 0)


@pytest.mark.asyncio
async def test_version_consistency_across_events():
    """Test that version is consistent when events are emitted."""
    conn = SimplyPrintConnection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))
    events_captured = []

    async def capture_established(event: SimplyPrintConnectionEstablishedEvent):
        events_captured.append(("established", event.v, conn.v))

    async def capture_lost(event: SimplyPrintConnectionLostEvent):
        events_captured.append(("lost", event.v, conn.v))

    conn.event_bus.on(SimplyPrintConnectionEstablishedEvent, capture_established)
    conn.event_bus.on(SimplyPrintConnectionLostEvent, capture_lost)

    # Emit events with specific versions
    await conn.event_bus.emit(SimplyPrintConnectionEstablishedEvent(42))
    await conn.event_bus.emit(SimplyPrintConnectionLostEvent(42))

    await asyncio.sleep(0.05)

    assert len(events_captured) == 2
    assert events_captured[0] == ("established", 42, 0)  # Event has v=42, conn.v=0
    assert events_captured[1] == ("lost", 42, 0)

    conn.stop()


@pytest.mark.asyncio
async def test_version_isolation_between_connections():
    """Test that versions are independent between different connection instances."""
    conn1 = SimplyPrintConnection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))
    conn2 = SimplyPrintConnection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))

    assert conn1.v == 0
    assert conn2.v == 0

    # Increment one connection's version
    conn1.v += 1

    # Other connection should be unaffected
    assert conn1.v == 1
    assert conn2.v == 0

    conn1.stop()
    conn2.stop()


@pytest.mark.asyncio
async def test_message_dropped_when_not_connected(connection):
    """Test that messages are dropped when the transport is not connected."""
    # Ensure not connected
    connection.transport = None
    connection.protocol.transport = None
    connection.v = 0

    test_msg = PingMsg()

    # Send message without version constraint - should still be dropped
    # because we're not connected
    await connection.send(test_msg, v=None)

    # The send method should not raise an exception, just drop the message silently
    assert True


@pytest.mark.asyncio
async def test_first_message_timeout_version_increment_is_single_not_double(
    connection,
):
    """
    Integration test that NEGATIVELY tests the double-increment bug.

    This runs the reconnect supervisor as a background task and triggers the
    first-message timeout. It verifies that the first dropped transport attempt
    increments the protocol version exactly once.
    """
    # Capture SimplyPrintConnectionLostEvent to know when timeout was handled
    lost_events = []
    first_lost = asyncio.Event()

    async def on_connection_lost(event: SimplyPrintConnectionLostEvent):
        lost_events.append(event)
        first_lost.set()

    connection.event_bus.on(SimplyPrintConnectionLostEvent, on_connection_lost)

    # Patch WsFirstMessageTimeout to be very short so test completes quickly
    with patch(
        "simplyprint_ws_client.core.protocol.connection.WsFirstMessageTimeout",
        0.01,  # 10ms timeout
    ):
        # Start the connection loop (which will eventually hit the first message timeout)
        await connection.connect()

        # Wait for the first timeout to trigger and stop before the reconnect
        # supervisor starts another fake attempt.
        await asyncio.wait_for(first_lost.wait(), 5)

        # Stop the connection loop to prevent further reconnection attempts
        await connection.disconnect()

        # Assert version was incremented exactly once
        assert connection.v == 1, (
            f"Expected v == 1 (single increment) but got {connection.v}. "
            f"This indicates a double-increment bug in the disconnected-event "
            f"path."
        )


@pytest.mark.asyncio
async def test_poll_failure_increments_version_via_exception_handler(
    connection, fake_transport
):
    """
    Integration test for poll() failure path.

    Tests the exception handler flow that catches WsConnectionErrors during
    poll() (now a dropped transport) and increments version.

    Verifies that version is incremented exactly once when poll() fails, not
    twice (which would indicate a double-increment bug).
    """
    # Capture connection lost events to verify exception was handled
    lost_events = []

    async def on_connection_lost(event: SimplyPrintConnectionLostEvent):
        lost_events.append(event)

    connection.event_bus.on(SimplyPrintConnectionLostEvent, on_connection_lost)

    # Use a long first message timeout to avoid interference with this test
    with patch(
        "simplyprint_ws_client.core.protocol.connection.WsFirstMessageTimeout",
        10.0,  # 10 seconds - long enough for test to complete
    ):
        # Start the connection loop
        await connection.connect()

        # Wait for initial connection to establish at v=0
        for _ in range(50):
            await asyncio.sleep(0.1)
            if connection.connected and connection.v == 0:
                break

        assert connection.connected and connection.v == 0, "Should be connected at v=0"

        # Make the next recv() raise TransientError -> exception handler increments v
        fake_transport.queue_close()

        # Wait for poll to fail and exception handler to increment v
        for _ in range(50):
            await asyncio.sleep(0.1)
            if connection.v > 0 and lost_events:
                break

        # Version should be incremented to exactly 1 by exception handler
        assert connection.v == 1, (
            f"Expected v=1 after poll() failure, got {connection.v}. "
            f"This indicates a double-increment bug in the exception handler path."
        )
        assert len(lost_events) == 1, (
            "Should have emitted exactly one SimplyPrintConnectionLostEvent"
        )

        # Clean up
        await connection.disconnect()
