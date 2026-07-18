"""Configurable application-level keepalive for a connection lease."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from simplyprint_ws_client.wire.events import ActivityTimeout, Connected, Disconnected
from simplyprint_ws_client.wire.errors import TransportError

if TYPE_CHECKING:
    from simplyprint_ws_client.wire.lease import Lease

KeepaliveProbe = Callable[["Lease"], object]


class KeepaliveTimeout(TransportError):
    """The lease saw no inbound activity after its configured probes."""


@dataclass(frozen=True)
class Keepalive:
    """Application-level liveness policy for one connection lease.

    ``probe`` is protocol-specific and may be sync or async. The lease handles the
    generic parts: inbound activity resets the miss counter, stale intervals
    call the probe, and exhausted probes report application inactivity without
    mutating the transport.
    """

    interval: float
    max_misses: int = 3
    probe: Optional[KeepaliveProbe] = None
    timeout_message: str = "Keepalive failed enough times"


class ConnectionKeepalive:
    """Runs a :class:`Keepalive` policy for one connection lease."""

    def __init__(
        self,
        connection: "Lease",
        policy: Keepalive,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.connection = connection
        self.policy = policy
        self.logger = logger or logging.getLogger("wire.keepalive")
        self.misses = 0
        self.timed_out_activity: Optional[float] = None
        self.task: Optional[asyncio.Task] = None

    def start(self) -> "ConnectionKeepalive":
        if self.task is not None and not self.task.done():
            return self

        self.connection.note_activity()
        self.connection.event_bus.on(Connected, self._on_activity)
        self.connection.event_bus.on(Disconnected, self._on_disconnected)
        self.connection.on_close(self.stop)
        self.task = self.connection.create_task(self._run())
        return self

    def stop(self) -> None:
        self.connection.off_close(self.stop)
        self.connection.event_bus.off(Connected, self._on_activity)
        self.connection.event_bus.off(Disconnected, self._on_disconnected)
        task = self.task
        self.task = None
        if task is not None:
            task.cancel()

    async def _on_activity(self, _event) -> None:
        self.connection.note_activity()
        self.misses = 0
        self.timed_out_activity = None

    async def _on_disconnected(self, _event: Disconnected) -> None:
        self.misses = 0

    async def _run(self) -> None:
        try:
            while not self.connection.closed:
                await asyncio.sleep(self.policy.interval)
                if self.connection.closed:
                    return
                await self._check()
        except asyncio.CancelledError:
            return

    async def _check(self) -> None:
        if not self.connection.connected:
            return

        transport = self.connection.transport
        generation = transport.generation
        activity = self.connection.last_activity
        now = self.connection.provider.event_loop.time()
        if now - activity < self.policy.interval:
            self.misses = 0
            self.timed_out_activity = None
            return
        if self.timed_out_activity == activity:
            await self._probe()
            return
        if self.timed_out_activity is not None:
            self.misses = 0
            self.timed_out_activity = None

        if self.misses >= self.policy.max_misses:
            reason = KeepaliveTimeout(self.policy.timeout_message)
            self.timed_out_activity = activity
            self.connection.deliver(
                ActivityTimeout(generation, code=reason, last_activity=activity)
            )
            await self._probe()
            return

        await self._probe()
        if (
            not self.connection.connected
            or transport.generation != generation
            or self.connection.last_activity != activity
        ):
            self.misses = 0
            self.timed_out_activity = None
            return
        self.misses += 1

    async def _probe(self) -> None:
        probe = self.policy.probe
        if probe is None:
            return
        try:
            result = probe(self.connection)
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 -- probe failure counts as a missed reply
            self.logger.warning("connection keepalive probe failed", exc_info=True)
