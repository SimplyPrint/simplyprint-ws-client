"""Neutral connection options shared by MQTT and WebSocket front doors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

from simplyprint_ws_client.contrib.connection.keepalive import Keepalive

if TYPE_CHECKING:
    from simplyprint_ws_client.contrib.connection.policy import RetryPolicy
    from simplyprint_ws_client.common.asyncio.event_loop_provider import (
        EventLoopProvider,
    )


@dataclass(frozen=True)
class WireKeepalive:
    """Transport-native ping/keepalive knobs.

    The front door maps these to the selected wire implementation: paho MQTT's
    CONNECT keepalive, websockets' ``ping_interval`` / ``ping_timeout``, aiohttp's
    heartbeat, etc. Unsupported values are ignored by transports that cannot use
    them.
    """

    interval: Optional[float] = None
    timeout: Optional[float] = None


@dataclass(frozen=True)
class ConnectionOptions:
    """Options that are meaningful across connection families.

    ``wire_keepalive`` configures the native transport. ``app_keepalive`` configures
    lease-level application probes. ``provider`` binds the pool and lease events to
    a specific loop. ``retry`` is used by supervised async transports.
    """

    provider: Optional["EventLoopProvider"] = None
    retry: Optional["RetryPolicy"] = None
    wire_keepalive: Optional[WireKeepalive] = None
    app_keepalive: Optional[Keepalive] = None

    def with_provider(self, provider: "EventLoopProvider") -> "ConnectionOptions":
        return replace(self, provider=provider)
