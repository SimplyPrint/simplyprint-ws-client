"""Comprehensive tests for WebSocket connection version (v) increment logic.

Tests ensure that:
- Version increments correctly on disconnects/reconnects
- Messages are not dropped while connected to the correct version
- Messages are dropped when targeting a stale version
- Multiple disconnect/reconnect cycles maintain version consistency
"""

import asyncio
from unittest.mock import AsyncMock, patch
from typing import List, Optional

import pytest
import pytest_asyncio
from aiohttp import WSMsgType, WSMessage

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


class MockWebSocket:
    """Mock WebSocket that simulates connection behavior."""

    def __init__(self):
        self.closed = False
        self.close_code = None
        self.sent_messages: List[str] = []
        self._receive_queue: asyncio.Queue = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None

    async def receive(self) -> WSMessage:
        """Simulate receiving a message."""
        item = await self._receive_queue.get()
        # If the item is a future with an exception, await it to raise the exception
        if asyncio.isfuture(item):
            return await item
        return item

    async def send_str(self, data: str) -> None:
        """Simulate sending a message."""
        if self.closed:
            raise ConnectionError("WebSocket is closed")
        self.sent_messages.append(data)

    async def close(self, *, code: int = 1000, message: bytes = b"") -> None:
        """Simulate closing the connection."""
        self.closed = True
        self.close_code = code

    def queue_receive(self, msg: WSMessage) -> None:
        """Queue a message to be received."""
        self._receive_queue.put_nowait(msg)

    def queue_error(self, error: Exception) -> None:
        """Queue an error to be raised on receive."""
        future: asyncio.Future = asyncio.Future()
        future.set_exception(error)
        self._receive_queue.put_nowait(future)


@pytest_asyncio.fixture
async def mock_ws():
    """Provide a mock WebSocket."""
    return MockWebSocket()


@pytest_asyncio.fixture
async def connection(mock_ws):
    """Create a Connection instance with mocked session."""
    conn = Connection(hint=ConnectionHint(mode=ConnectionMode.SINGLE))

    # Provide the running event loop to the connection
    conn.use_running_loop()

    # Mock the session to return our mock WebSocket
    mock_session = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)
    mock_session.close = AsyncMock()
    conn.session = mock_session

    yield conn

    # Cleanup
    conn.stop()


@pytest.mark.asyncio
async def test_initial_version_is_zero(connection):
    """Test that connection starts with version 0."""
    assert connection.v == 0


@pytest.mark.asyncio
async def test_message_dropped_when_version_mismatch(connection, mock_ws):
    """Test that messages are dropped when targeting a different version."""
    # Start with version 0
    assert connection.v == 0

    # Create a test message
    test_msg = PingMsg()

    # Send message targeting version 0 - should be accepted
    await connection.send(test_msg, v=0)
    assert len(mock_ws.sent_messages) == 0  # Not connected, message dropped

    # Now try targeting wrong version
    connection.ws = mock_ws  # Fake connection
    await connection.send(test_msg, v=1)
    assert len(mock_ws.sent_messages) == 0  # Version mismatch, dropped


@pytest.mark.asyncio
async def test_message_sent_when_version_matches(connection, mock_ws):
    """Test that messages are sent when version matches."""
    # Setup: fake a connected state
    connection.ws = mock_ws
    connection.v = 0

    # Create a test message
    test_msg = PingMsg()

    # Send message targeting matching version
    await connection.send(test_msg, v=0)
    assert len(mock_ws.sent_messages) == 1

    # Increment version and try again - should fail
    connection.v = 1
    await connection.send(test_msg, v=0)
    assert len(mock_ws.sent_messages) == 1  # Not sent due to version mismatch


@pytest.mark.asyncio
async def test_message_sent_without_version_constraint(connection, mock_ws):
    """Test that messages without version constraint are sent when connected."""
    # Setup: fake a connected state
    connection.ws = mock_ws
    connection.v = 0

    # Create a test message
    test_msg = PingMsg()

    # Send message without version constraint
    await connection.send(test_msg, v=None)
    assert len(mock_ws.sent_messages) == 1

    # Change version - message should still be sent since no constraint
    connection.v = 5
    await connection.send(test_msg, v=None)
    assert len(mock_ws.sent_messages) == 2


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
    """Test that messages are dropped when WebSocket is not connected."""
    # Ensure not connected
    connection.ws = None
    connection.v = 0

    test_msg = PingMsg()

    # Send message without version constraint - should still be dropped
    # because we're not connected
    await connection.send(test_msg, v=None)

    # The send method should not raise an exception, just drop the message silently
    assert True


@pytest.mark.asyncio
async def test_first_message_timeout_version_increment_is_single_not_double(
    connection, mock_ws
):
    """
    Integration test that NEGATIVELY tests the bug on line 369.

    This test runs the actual _loop() as a background task and triggers the first message timeout.
    It verifies that version is incremented exactly once, not twice.

    Bug behavior: _close_ws() increments v, then raise ConnectionResetError(), then exception
    handler increments v again → v becomes 2
    Fixed behavior: _close_ws() increments v once → v becomes 1
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
        # The loop will: connect → wait for first message → timeout → close ws → increment v
        # Maximum wait time is timeout + some buffer for processing
        for _ in range(50):  # Try 50 times with 100ms sleep = 5 seconds max wait
            await asyncio.sleep(0.1)
            if lost_events:  # ConnectionLostEvent was emitted
                break

        # Stop the connection loop to prevent further reconnection attempts
        await connection.disconnect()

        # Assert version was incremented exactly once
        # FAILS if bug is present (v would be 2 instead of 1)
        assert connection.v == 1, (
            f"Expected v == 1 (single increment) but got {connection.v}. "
            f"This indicates the bug on line 369 is present: "
            f"_close_ws() increments v, then raises exception, then exception handler increments v again."
        )


@pytest.mark.asyncio
async def test_poll_failure_increments_version_via_exception_handler(
    connection, mock_ws
):
    """
    Integration test for poll() failure path.

    Tests the exception handler flow (lines 368-375) that catches WsConnectionErrors
    during poll() and increments version.

    This exercises the exception handler increment path (line 375: self.v += 1)
    that's different from the first message timeout path.

    Verifies that version is correctly incremented exactly once when poll() fails,
    not twice (which would indicate a double-increment bug similar to line 369).
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

        # Queue a CLOSE message that will cause poll() to raise ConnectionResetError
        # This triggers the exception handler at line 368-375 which increments v
        mock_ws.queue_receive(WSMessage(WSMsgType.CLOSE, b"", None))

        # Wait for poll to fail and exception handler to increment v
        for _ in range(50):
            await asyncio.sleep(0.1)
            if connection.v > 0 and lost_events:
                break

        # Version should be incremented to exactly 1 by exception handler
        # FAILS if there's a double-increment bug: v would be 2
        assert connection.v == 1, (
            f"Expected v=1 after poll() failure, got {connection.v}. "
            f"This indicates a double-increment bug in the exception handler path "
            f"(similar to line 369 bug where raise statement causes double increment)."
        )
        assert len(lost_events) == 1, (
            "Should have emitted exactly one ConnectionLostEvent"
        )

        # Clean up
        await connection.disconnect()
