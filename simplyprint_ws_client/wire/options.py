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
class TlsClientAuth:
    """Mutual-TLS material issued by a printer at pairing time.

    The printer is its own CA; ``ca_pem`` is the printer's root certificate
    used to verify the broker's server cert. ``cert_pem`` / ``key_pem`` are
    the client credential the broker requires to accept the connection.
    """

    ca_pem: str
    cert_pem: str
    key_pem: str


@dataclass(frozen=True)
class WireKeepalive:
    """Transport-native ping/keepalive knobs.

    The front doors map these to paho MQTT's CONNECT keepalive and websockets'
    ``ping_interval`` / ``ping_timeout``. Unsupported values are ignored by
    transports that cannot use them.
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
    explicit, opt-in knob. ``tls_client_auth`` supplies mutual-TLS credentials
    issued by the printer at pairing time; when set it takes precedence over
    ``verify_tls``. ``logger`` names where
    the transport's lifecycle log lines land; the device drivers default it to the
    printer's own child logger so wire events end up in that printer's log files.
    Endpoints are pooled, so the logger of the FIRST lease that builds a transport
    owns its log lines.
    """

    provider: Optional["EventLoopProvider"] = None
    retry: Optional["RetryPolicy"] = None
    wire_keepalive: Optional[WireKeepalive] = None
    app_keepalive: Optional[Keepalive] = None
    verify_tls: bool = False
    tls_client_auth: Optional[TlsClientAuth] = None
    logger: Optional["logging.Logger"] = None
    #: Bound on one WebSocket connect attempt (paho owns its own timeouts).
    open_timeout: Optional[float] = None
