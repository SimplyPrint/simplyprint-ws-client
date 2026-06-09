"""How long the reconnect loop keeps reaching for an endpoint, and how it paces retries.

A :class:`RetryPolicy` is the one knob the reconnect loop consults between
attempts: it owns the backoff schedule and the two give-up bounds (a max attempt
count and a wall-clock deadline). The loop asks it, per failed attempt, for the
next delay and whether it should stop -- the policy holds the counting so the
loop stays a clean state machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from simplyprint_ws_client.shared.utils.backoff import Backoff, ConstantBackoff


@dataclass
class RetryPolicy:
    """The retry schedule and give-up bounds for one supervised link.

    With both bounds ``None`` (the default) the reconnect loop retries forever at the
    backoff's pace. Set ``max_attempts`` to cap the number of failed attempts, or
    ``give_up_after`` to cap the elapsed wall-clock seconds since the first
    attempt; whichever is hit first ends the supervision and leaves the link
    permanently ``DISCONNECTED``.
    """

    backoff: Backoff = field(default_factory=ConstantBackoff)
    #: Maximum number of failed attempts before giving up; ``None`` = forever.
    max_attempts: Optional[int] = None
    #: Maximum elapsed seconds since the first attempt before giving up;
    #: ``None`` = forever.
    give_up_after: Optional[float] = None

    def attempt(self, *, started_at: Optional[float] = None) -> RetryAttempt:
        """Begin tracking a fresh supervision run.

        ``started_at`` is the monotonic clock reading of the first attempt;
        defaults to now. The returned :class:`RetryAttempt` carries the per-run
        state (attempt count, deadline) the loop threads through each retry.
        """
        return RetryAttempt(
            self, time.monotonic() if started_at is None else started_at
        )


class RetryAttempt:
    """Per-run bookkeeping for one :class:`RetryPolicy`.

    One of these lives for the duration of a single supervision loop. After each
    failed attempt the loop calls :meth:`next_delay`; it returns the seconds to
    sleep before the next try, or ``None`` when a give-up bound is exhausted (the
    loop then stops and stays ``DISCONNECTED``).
    """

    def __init__(self, policy: RetryPolicy, started_at: float) -> None:
        self.policy = policy
        self.started_at = started_at
        self.attempts = 0

    def next_delay(self) -> Optional[float]:
        """Account for one failed attempt; return the delay before the next, or
        ``None`` if the policy says to give up now."""
        self.attempts += 1

        if (
            self.policy.max_attempts is not None
            and self.attempts >= self.policy.max_attempts
        ):
            return None

        delay = self.policy.backoff.delay()

        if self.policy.give_up_after is not None:
            elapsed = time.monotonic() - self.started_at
            if elapsed + delay >= self.policy.give_up_after:
                return None

        return delay

    def reset(self) -> None:
        """Forget the failure history after a successful connect, so a later drop
        retries from a fresh backoff and bound."""
        self.attempts = 0
        self.started_at = time.monotonic()
        self.policy.backoff.reset()
