"""Neutral connection options shared by MQTT and WebSocket front doors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from simplyprint_ws_client.wire.keepalive import Keepalive

if TYPE_CHECKING:
    import logging

    from simplyprint_ws_client.wire.policy import RetryPolicy
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
    a specific loop. ``retry`` paces reconnects (supervised async transports use it
    fully; paho maps its backoff onto ``reconnect_delay_set``). ``verify_tls``
    controls broker certificate verification for ``mqtts://`` - off by default
    because printer fleets routinely present self-signed certificates, but an
    explicit, opt-in knob. ``logger`` names where the transport's lifecycle log
    lines land; the device drivers default it to the printer's own child logger
    so wire events end up in that printer's log files. Endpoints are pooled, so
    the logger of the FIRST lease that builds a transport owns its log lines.
    """

    provider: Optional["EventLoopProvider"] = None
    retry: Optional["RetryPolicy"] = None
    wire_keepalive: Optional[WireKeepalive] = None
    app_keepalive: Optional[Keepalive] = None
    verify_tls: bool = False
    logger: Optional["logging.Logger"] = None
    #: Bound on one connect attempt for the supervised async transports
    #: (``None`` = the transport's own default; paho owns its own timeouts).
    open_timeout: Optional[float] = None
