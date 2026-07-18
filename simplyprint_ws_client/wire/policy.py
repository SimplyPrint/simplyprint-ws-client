"""Deterministic pacing for a connection that never gives up."""

from __future__ import annotations

from dataclasses import dataclass, field

from simplyprint_ws_client.common.utils.backoff import Backoff, ConstantBackoff


@dataclass
class RetryPolicy:
    """The delay between disposable connection attempts."""

    backoff: Backoff = field(default_factory=ConstantBackoff)

    def delay(self) -> float:
        return self.backoff.delay()

    def reset(self) -> None:
        self.backoff.reset()
