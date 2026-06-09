"""The three states a connection can be in.

Every transport reports exactly one of these at a time, and publishes a
:class:`~simplyprint_ws_client.contrib.connection.events.ConnectionEvent` whenever it
moves between them. The values are stable strings so a state survives logging,
serialization, and the wire untouched.
"""

from __future__ import annotations

from enum import StrEnum


class ConnectionState(StrEnum):
    """Where a link is in its lifecycle.

    ``CONNECTING`` is the active reach for a wire (the first attempt or any
    recovery); ``CONNECTED`` is a live wire able to carry messages;
    ``DISCONNECTED`` is no live wire -- either between retry attempts or after the
    reconnect loop has permanently given up.
    """

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
