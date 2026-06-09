"""Comprehensive tests for WebSocket connection version (v) increment logic.

Tests ensure that:
- Version increments correctly on disconnects/reconnects
- Messages are not dropped while connected to the correct version
- Messages are dropped when targeting a stale version
- Multiple disconnect/reconnect cycles maintain version consistency

These drive :class:`Connection` through a :class:`FakeBackend`, so they pin the
version/state contract independently of the concrete socket library.
"""

import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio

from simplyprint_ws_client.core.ws_protocol.connection import (
    Connection,
    ConnectionHint,
    ConnectionMode,
)
from simplyprint_ws_client.core.ws_protocol.events import (
    ConnectionEstablishedEvent,
    ConnectionLostEvent,
    ConnectionIncomingEvent,
)
from simplyprint_ws_client.core.ws_protocol.messages import (
    PingMsg,
)

from tests._fakes import FakeBackend


@pytest_asyncio.fixture
async def fake_backend():
    """Provide a fake backend instance."""
    return FakeBackend()


@pytest_asyncio.fixture
async def connection(fake_backend):
    """Create a Connection whose backend factory yields the fake backend."""
    conn = Connection(
        backend_factory=lambda logger: fake_backend,
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
async def test_message_dropped_when_version_mismatch(connection, fake_backend):
    """Test that messages are dropped when targeting a different version."""
    # Start with version 0
    assert connection.v == 0

    # Create a test message
    test_msg = PingMsg()

    # Send message targeting version 0 - should be dropped (not connected)
    await connection.send(test_msg, v=0)
    assert len(fake_backend.sent) == 0  # Not connected, message dropped

    # Now connect (fake) and try targeting wrong version
    connection.backend = fake_backend.open()
    await connection.send(test_msg, v=1)
    assert len(fake_backend.sent) == 0  # Version mismatch, dropped


@pytest.mark.asyncio
async def test_message_sent_when_version_matches(connection, fake_backend):
    """Test that messages are sent when version matches."""
    # Setup: fake a connected state
    connection.backend = fake_backend.open()
    connection.v = 0

    # Create a test message
    test_msg = PingMsg()

    # Send message targeting matching version
    await connection.send(test_msg, v=0)
    assert len(fake_backend.sent) == 1

    # Increment version and try again - should fail
    connection.v = 1
    await connection.send(test_msg, v=0)
    assert len(fake_backend.sent) == 1  # Not sent due to version mismatch


@pytest.mark.asyncio
async def test_message_sent_without_version_constraint(connection, fake_backend):
    """Test that messages without version constraint are sent when connected."""
    # Setup: fake a connected state
    connection.backend = fake_backend.open()
    connection.v = 0

    # Create a test message
    test_msg = PingMsg()

    # Send message without version constraint
    await connection.send(test_msg, v=None)
    assert len(fake_backend.sent) == 1

    # Change version - message should still be sent since no constraint
    connection.v = 5
    await connection.send(test_msg, v=None)
    assert len(fake_backend.sent) == 2


@pytest.mark.asyncio
async def test_incoming_message_tagged_with_correct_version(connection):
    """Test that incoming messages are tagged with the current version."""
    received_messages = []

    async def on_message(msg, v):
        received_messages.append((msg, v))

    # Subscribe to incoming events
    connection.event_bus.on(ConnectionIncomingEvent, on_message)

    # Simulate connection at version 0
    connection.v = 0

    # Emit a message event as would happen in poll()
    test_msg = {"msg_type": "test"}
    await connection.event_bus.emit(ConnectionIncomingEvent, test_msg, 0)

    await asyncio.sleep(0.05)
    assert len(received_messages) == 1
    # Verify the message was tagged with version 0
    assert received_messages[0] == (test_msg, 0)


@pytest.mark.asyncio
async def test_version_consistency_across_events():
    """Test that version is consistent when events are emitted."""
    conn = Connection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))
    events_captured = []

    async def capture_established(event: ConnectionEstablishedEvent):
        events_captured.append(("established", event.v, conn.v))

    async def capture_lost(event: ConnectionLostEvent):
        events_captured.append(("lost", event.v, conn.v))

    conn.event_bus.on(ConnectionEstablishedEvent, capture_established)
    conn.event_bus.on(ConnectionLostEvent, capture_lost)

    # Emit events with specific versions
    await conn.event_bus.emit(ConnectionEstablishedEvent(42))
    await conn.event_bus.emit(ConnectionLostEvent(42))

    await asyncio.sleep(0.05)

    assert len(events_captured) == 2
    assert events_captured[0] == ("established", 42, 0)  # Event has v=42, conn.v=0
    assert events_captured[1] == ("lost", 42, 0)

    conn.stop()


@pytest.mark.asyncio
async def test_version_isolation_between_connections():
    """Test that versions are independent between different connection instances."""
    conn1 = Connection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))
    conn2 = Connection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))

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
    """Test that messages are dropped when the backend is not connected."""
    # Ensure not connected
    connection.backend = None
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

    This runs the actual _loop() as a background task and triggers the first
    message timeout. It verifies that version is incremented exactly once.

    Bug behavior: _close_ws() increments v, then raise, then exception handler
    increments v again -> v becomes 2.
    Fixed behavior: _close_ws() increments v once -> v becomes 1.
    """
    # Capture ConnectionLostEvent to know when timeout was handled
    lost_events = []

    async def on_connection_lost(event: ConnectionLostEvent):
        lost_events.append(event)

    connection.event_bus.on(ConnectionLostEvent, on_connection_lost)

    # Patch WsFirstMessageTimeout to be very short so test completes quickly
    with patch(
        "simplyprint_ws_client.core.ws_protocol.connection.WsFirstMessageTimeout",
        0.01,  # 10ms timeout
    ):
        # Start the connection loop (which will eventually hit the first message timeout)
        await connection.connect()

        # Wait for the timeout to trigger and be handled
        # The loop will: connect -> wait for first message -> timeout -> close -> v += 1
        for _ in range(50):  # Try 50 times with 100ms sleep = 5 seconds max wait
            await asyncio.sleep(0.1)
            if lost_events:  # ConnectionLostEvent was emitted
                break

        # Stop the connection loop to prevent further reconnection attempts
        await connection.disconnect()

        # Assert version was incremented exactly once
        assert connection.v == 1, (
            f"Expected v == 1 (single increment) but got {connection.v}. "
            f"This indicates a double-increment bug: _close_ws() increments v, "
            f"then raises, then the exception handler increments v again."
        )


@pytest.mark.asyncio
async def test_poll_failure_increments_version_via_exception_handler(
    connection, fake_backend
):
    """
    Integration test for poll() failure path.

    Tests the exception handler flow that catches WsConnectionErrors during
    poll() (now a dropped backend -> BackendClosed) and increments version.

    Verifies that version is incremented exactly once when poll() fails, not
    twice (which would indicate a double-increment bug).
    """
    # Capture connection lost events to verify exception was handled
    lost_events = []

    async def on_connection_lost(event: ConnectionLostEvent):
        lost_events.append(event)

    connection.event_bus.on(ConnectionLostEvent, on_connection_lost)

    # Use a long first message timeout to avoid interference with this test
    with patch(
        "simplyprint_ws_client.core.ws_protocol.connection.WsFirstMessageTimeout",
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

        # Make the next recv() raise BackendClosed -> exception handler increments v
        fake_backend.queue_close()

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
            "Should have emitted exactly one ConnectionLostEvent"
        )

        # Clean up
        await connection.disconnect()
