"""Reconnection as a composable transport.

A :class:`ReconnectingTransport` transport keeps a live link to one endpoint by rebuilding
it from a factory whenever it drops -- backoff, a generation counter, liveness, and
crash recovery -- so a wire that is merely "one connection" (open / recv / send /
close) becomes self-healing without anyone writing the loop. Wires that already
self-heal (paho's network thread, websocket-client's ``run_forever``) do NOT use
this; they are complete transports on their own.

Brand-free: a :class:`Link` yields opaque payloads and this module knows nothing of
any protocol, route, URL, or message shape. Reconnection and routing/pooling are
orthogonal -- a :class:`ReconnectingTransport` is used directly for a 1:1 link, or stored in a
:class:`~.pool.Pool` like any other transport when many clients share one socket.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.utils.backoff import Backoff, ConstantBackoff
from simplyprint_ws_client.shared.utils.bounded_variable import BoundedInterval

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    ConnectionSuspect,
    Disconnected,
    MessageReceived,
    StateChanged,
    TransportEvent,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import AsyncTransport, TParams

__all__ = ["Link", "LinkFactory", "ReconnectingTransport"]


class Link(ABC):
    """One live connection -- the per-attempt wire a :class:`ReconnectingTransport` drives.

    A fresh ``Link`` is built per attempt and is single-use (``open`` -> ``recv``\\*
    / ``send`` -> ``close``); it owns no reconnect / backoff / generation logic. The
    existing raw WebSocket socket already has this shape; an async MQTT session wraps
    aiomqtt to match.
    """

    @abstractmethod
    async def open(self) -> None:
        """Establish the connection. Raise on failure (drives the reconnect path)."""

    @abstractmethod
    async def recv(self) -> Optional[Any]:
        """Return the next inbound payload (``None`` skips an uninteresting frame).

        Raise when the link drops -- that ends the attempt and triggers a reconnect.
        """

    @abstractmethod
    async def send(self, payload: Any) -> None:
        """Send ``payload`` on the live link; raise if it is unusable."""

    @abstractmethod
    async def close(self) -> None:
        """Tear the connection down. Idempotent; never raises."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the link currently holds a live connection."""

    async def on_connected(self) -> None:
        """Optional post-open work on the live link before consuming begins -- e.g.
        an MQTT session (re)subscribes its topics here. Default: nothing."""


#: Builds a fresh, fully-configured :class:`Link` for one connection attempt.
LinkFactory = Callable[[], Link]


class ReconnectingTransport(AsyncTransport[TParams]):
    """An :class:`AsyncTransport` that keeps a live :class:`Link`, rebuilding on drop.

    Per attempt: build a fresh link, ``open`` it, go ``ONLINE`` + emit
    :class:`Connected` + reset backoff + ``on_connected``, then stream ``recv()`` as
    :class:`MessageReceived` until the link drops. On any drop / error / idle-timeout:
    ``close`` the link, bump :attr:`generation` **once**, go ``OFFLINE`` + emit
    :class:`Disconnected` (``transient``) and -- if a ``suspect_after`` bound is set --
    a periodic :class:`ConnectionSuspect`, then back off and retry. Crash-safe (any error
    reconnects), cancel-safe (``stop`` cancels the task), idempotent ``start``/``stop``.
    """

    def __init__(
        self,
        params: TParams,
        link_factory: LinkFactory,
        *,
        logger: Optional[logging.Logger] = None,
        backoff: Optional[Backoff] = None,
        suspect_after: Optional[BoundedInterval[int]] = None,
        first_message_timeout: Optional[float] = None,
    ) -> None:
        self.params = params
        self.events: EventBus[TransportEvent] = EventBus()
        self.state = ConnectionState.OFFLINE
        self.generation = 0
        self._link_factory = link_factory
        self._logger = logger or logging.getLogger("transport.reconnect")
        self._backoff = backoff or ConstantBackoff()
        self._suspect_after = suspect_after
        self._first_message_timeout = first_message_timeout
        self._link: Optional[Link] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    @property
    def connected(self) -> bool:
        link = self._link
        return (
            self.state is ConnectionState.ONLINE and link is not None and link.is_open
        )

    @property
    def task(self) -> Optional[asyncio.Task]:
        """The supervision task, while running -- so an owner can await its
        wind-down at teardown. ``None`` before :meth:`start` / after :meth:`stop`."""
        return self._task

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop = False
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        self._stop = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def send(self, payload: Any) -> None:
        link = self._link
        if link is None or self.state is not ConnectionState.ONLINE:
            raise ConnectionError("transport not connected")
        await link.send(payload)

    def _set_state(self, state: ConnectionState) -> None:
        if state is not self.state:
            self.state = state
            self._emit(StateChanged(state))

    async def _run(self) -> None:
        suspect = (
            self._suspect_after.create_variable(0)
            if self._suspect_after is not None
            else None
        )
        while not self._stop:
            link = self._link_factory()
            was_connected = False
            try:
                await link.open()
                self._link = link
                was_connected = True
                self._set_state(ConnectionState.ONLINE)
                self._emit(Connected())
                self._backoff.reset()
                if suspect is not None:
                    suspect.reset()
                await link.on_connected()
                await self._consume(link)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 -- supervised: any error reconnects
                self._logger.debug("link %s dropped: %s", self.params, e)
                if not self._stop and (suspect is None or suspect.guard_until_bound()):
                    self._emit(ConnectionSuspect(error=e))
            finally:
                # Single generation-bump site: exactly once per ended attempt,
                # regardless of how it ended (clean drop, error, or liveness timeout).
                self._link = None
                try:
                    await link.close()
                finally:
                    if was_connected or not self._stop:
                        self.generation += 1
                        self._set_state(ConnectionState.OFFLINE)
                        self._emit(Disconnected(reason="link down", transient=True))
            if not self._stop:
                await asyncio.sleep(self._backoff.delay())

    async def _consume(self, link: Link) -> None:
        # A liveness deadline applies only until the first real message arrives; if
        # none does in time the attempt is dropped (and reconnected) like any other.
        if self._first_message_timeout is not None:
            await asyncio.wait_for(self._await_first(link), self._first_message_timeout)
        while not self._stop:
            payload = await link.recv()
            if payload is not None:
                self._emit(MessageReceived(payload))

    async def _await_first(self, link: Link) -> None:
        while not self._stop:
            payload = await link.recv()
            if payload is not None:
                self._emit(MessageReceived(payload))
                return
