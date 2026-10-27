"""Deterministic pacing for a connection that never gives up."""

from __future__ import annotations

from dataclasses import dataclass, field

from simplyprint_ws_client.common.utils.backoff import (
    Backoff,
    ConstantBackoff,
    DecorrelatedJitterBackoff,
)

#: A device link's first retry comes within [0.5, 1.5] s (a printer that
#: rebooted is back fast), later ones within 5 s.
DEVICE_RETRY_BASE = 0.5
DEVICE_RETRY_MAX = 5.0


@dataclass
class RetryPolicy:
    """The delay between disposable connection attempts."""

    backoff: Backoff = field(default_factory=ConstantBackoff)

    def delay(self) -> float:
        return self.backoff.delay()

    def reset(self) -> None:
        self.backoff.reset()


def device_retry_policy() -> RetryPolicy:
    """The pace of a printer's own link (MQTT, WebSocket): decorrelated
    jitter, so links that dropped together do not retry together."""
    return RetryPolicy(DecorrelatedJitterBackoff(DEVICE_RETRY_BASE, DEVICE_RETRY_MAX))
