"""Brand-agnostic connection awareness.

A single vocabulary for "is this printer reachable" that every transport
(MQTT / WebSocket / HTTP-polling) can report, independent of brand. This
replaces the ad-hoc ``is_connected`` booleans + scattered events each client
grew on its own.
"""

from __future__ import annotations

from enum import Enum


class ConnectionState(str, Enum):
    """High-level reachability of a printer connection."""

    #: Transport is established and the printer is responding.
    ONLINE = "online"
    #: Transport is down (network error, timeout, lost connection).
    OFFLINE = "offline"
    #: Credentials were rejected by the printer/broker.
    AUTH_FAILED = "auth_failed"
    #: The stored config cannot produce a valid connection (e.g. missing host).
    CONFIG_INVALID = "config_invalid"

    @property
    def is_usable(self) -> bool:
        """Whether commands can currently be sent over this connection."""
        return self is ConnectionState.ONLINE
