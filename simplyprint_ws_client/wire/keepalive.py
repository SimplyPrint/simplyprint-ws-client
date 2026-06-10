"""Configurable application-level keepalive for a connection lease."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from simplyprint_ws_client.wire.events import (
    Connected,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.transport import TransientError

if TYPE_CHECKING:
    from simplyprint_ws_client.wire.lease import Lease

KeepaliveProbe = Callable[["Lease"], object]


class KeepaliveTimeout(TransientError):
    """The lease saw no inbound activity after its configured probes."""


@dataclass(frozen=True)
class Keepalive:
    """Application-level liveness policy for one connection lease.

    ``probe`` is protocol-specific and may be sync or async. The lease handles the
    generic parts: inbound activity resets the miss counter, stale intervals call
    the probe, and too many missed probes emits a generic ``Disconnected`` event on
    the lease bus.
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
        self.logger = logger or logging.getLogger("conn.keepalive")
        self.misses = 0
        self.timed_out = False
        self.last_activity = 0.0
        self.task: Optional[asyncio.Task] = None

    def start(self) -> "ConnectionKeepalive":
        if self.task is not None and not self.task.done():
            return self

        loop = self.connection.provider.event_loop
        self.last_activity = loop.time()
        self.connection.event_bus.on(Connected, self._on_activity)
        self.connection.event_bus.on(MessageReceived, self._on_activity)
        self.connection.event_bus.on(Disconnected, self._on_disconnected)
        self.connection.on_close(self.stop)
        self.task = self.connection.create_task(self._run())
        return self

    def stop(self) -> None:
        self.connection.off_close(self.stop)
        self.connection.event_bus.off(Connected, self._on_activity)
        self.connection.event_bus.off(MessageReceived, self._on_activity)
        self.connection.event_bus.off(Disconnected, self._on_disconnected)
        task = self.task
        self.task = None
        if task is not None:
            task.cancel()

    async def _on_activity(self, _event) -> None:
        self.last_activity = self.connection.provider.event_loop.time()
        self.misses = 0
        self.timed_out = False

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
        if self.timed_out or not self.connection.connected:
            return

        now = self.connection.provider.event_loop.time()
        if now - self.last_activity < self.policy.interval:
            return

        if self.misses >= self.policy.max_misses:
            self.timed_out = True
            self.misses = 0
            await self.connection.event_bus.emit(
                Disconnected(
                    self.connection.generation,
                    code=KeepaliveTimeout(self.policy.timeout_message),
                )
            )
            return

        await self._probe()
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
