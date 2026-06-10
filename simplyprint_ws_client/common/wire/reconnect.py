"""The supervised reconnect loop -- one asyncio task that keeps a wire alive.

This is the heart of the async wire families. A :class:`Reconnecting` transport is
itself an asyncio-native wire: a concrete subclass fills FOUR hooks on itself --
:meth:`~Reconnecting.open`, :meth:`~Reconnecting.recv`, :meth:`~Reconnecting.write`,
:meth:`~Reconnecting.aclose` -- and this base supplies everything around them: a
single supervision task, the connect/consume/drop/backoff cycle, the generation
counter, state, and the lifecycle events. There is no separate "link" object; the
hooks are methods on the concrete impl, so a wire is one class top to bottom.

The loop always keeps retrying. The :class:`~simplyprint_ws_client.common.wire.policy.RetryPolicy`
only decides the pace and, optionally, when to give up entirely -- at which point
the loop stops and the transport stays ``DISCONNECTED``. ``open`` and ``recv`` may
raise to end an attempt; the kind of exception is carried through to
``Disconnected.code`` but never changes the decision to retry.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from typing import Optional

import yarl

from simplyprint_ws_client.common.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.common.wire.events import (
    Connected,
    Connecting,
    WireEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.common.wire.errors import TransportError
from simplyprint_ws_client.common.wire.messages import message_qos
from simplyprint_ws_client.common.wire.policy import RetryPolicy
from simplyprint_ws_client.common.wire.state import ConnectionState
from simplyprint_ws_client.common.wire.transport import (
    NotConnected,
    TransientError,
    Transport,
)

__all__ = ["Reconnecting"]


class Reconnecting(Transport):
    """A :class:`Transport` that keeps one wire alive on a single asyncio task.

    Subclass it and implement the four wire hooks; the base owns the loop. Per
    attempt it emits :class:`Connecting`, calls :meth:`open`, then on success bumps
    the generation, goes ``CONNECTED``, emits :class:`Connected`, resets the
    backoff, and streams :meth:`recv` as :class:`MessageReceived` until the wire
    drops. On any drop it tears the wire down with :meth:`aclose`, goes
    ``DISCONNECTED`` with a :class:`Disconnected` tagged by the failure, then backs
    off and retries -- unless the policy's give-up bound is exhausted, in which case
    it stops and stays ``DISCONNECTED``.
    """

    def __init__(
        self,
        url: yarl.URL,
        policy: Optional[RetryPolicy] = None,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        first_message_timeout: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[WireEvent] = EventBus()
        self.policy = policy or RetryPolicy()
        self.provider = provider or EventLoopProvider.default()
        self.first_message_timeout = first_message_timeout
        self.logger = logger or logging.getLogger("conn.reconnect")
        self.live = False
        self.stopped = False
        self.gave_up = False
        self.task: Optional[asyncio.Task] = None

    @abstractmethod
    async def open(self) -> None:
        """Establish the live wire. Raise to end the attempt and trigger a retry.

        Raise :class:`~simplyprint_ws_client.common.wire.transport.TransientError`
        / :class:`~simplyprint_ws_client.common.wire.transport.FatalError` to tag
        the reason on ``Disconnected.code``; any other exception is treated the
        same (the reconnect loop still retries).
        """

    @abstractmethod
    async def recv(self) -> Optional[object]:
        """Return the next inbound message (``None`` to skip an uninteresting
        frame). Raise when the wire drops -- that ends the attempt."""

    @abstractmethod
    async def write(self, message: object) -> None:
        """Put one ``message`` on the live wire."""

    @abstractmethod
    async def aclose(self) -> None:
        """Tear the live wire down. Idempotent; must never raise."""

    @property
    def connected(self) -> bool:
        return self.state is ConnectionState.CONNECTED and self.live

    def supervising(self) -> bool:
        """``True`` while the loop will still try to reconnect; ``False`` once it
        has permanently given up (retry policy exhausted), been stopped, or its
        supervision task has otherwise finished.

        The last clause keeps the answer honest if the task dies for a reason the
        flags do not capture -- e.g. a wire hook that violates the contract by
        raising :class:`asyncio.CancelledError` itself. A finished task is not
        going to reconnect, so a waiter must not believe one is still in flight."""
        if self.gave_up or self.stopped:
            return False
        return self.task is None or not self.task.done()

    def start(self) -> None:
        if self.task is not None and not self.task.done():
            return
        self.stopped = False
        self.gave_up = False
        self.task = self.provider.event_loop.create_task(self.supervise())

    async def stop(self) -> None:
        self.stopped = True
        task = self.task
        self.task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def send(self, message: object) -> None:
        if not self.connected:
            raise NotConnected("transport not connected")
        await self.write(message)

    async def supervise(self) -> None:
        """The one task: keep a live wire up, retrying per the policy.

        Each iteration is one connection attempt. The generation is bumped exactly
        once -- on a successful open -- so every :class:`Connected` and
        :class:`MessageReceived` for that link shares one epoch.
        """
        attempt = self.policy.attempt()
        try:
            while not self.stopped:
                await self.mark_connecting()
                code: Optional[TransportError] = None
                try:
                    await self.open()
                    await self.mark_connected()
                    attempt.reset()
                    await self.consume()
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 -- supervised: any error retries
                    code = self.transport_error(error)
                    self.logger.debug("wire %s dropped: %s", self.url, error)
                finally:
                    self.live = False
                    await self.teardown()

                if self.stopped:
                    break

                # Decide whether to retry *before* announcing the drop, so the
                # Disconnected a waiter sees already carries the terminal verdict
                # (``supervising()`` is False on a give-up).
                delay = attempt.next_delay()
                self.gave_up = delay is None
                await self.mark_disconnected(code)

                if delay is None:
                    self.logger.debug("wire %s gave up retrying", self.url)
                    break
                await asyncio.sleep(delay)
        finally:
            # A stopped or given-up link has no live wire -- settle the public
            # state to DISCONNECTED on EVERY exit, including a stop() that cancels
            # this task mid-open or mid-recv (the cancellation re-raises past the
            # loop body, so this finally is the only spot that always runs).
            self.set_state(ConnectionState.DISCONNECTED)

    def set_state(self, state: ConnectionState) -> None:
        self.state = state

    async def transition(self, state: ConnectionState, event: WireEvent) -> None:
        self.set_state(state)
        await self.events.emit(event)

    async def mark_connecting(self) -> None:
        await self.transition(ConnectionState.CONNECTING, Connecting(self.generation))

    async def mark_connected(self) -> None:
        self.generation += 1
        self.live = True
        await self.transition(ConnectionState.CONNECTED, Connected(self.generation))

    async def mark_disconnected(self, code: Optional[TransportError]) -> None:
        await self.transition(
            ConnectionState.DISCONNECTED,
            Disconnected(self.generation, code=code),
        )

    @staticmethod
    def transport_error(error: Exception) -> TransportError:
        if isinstance(error, TransportError):
            return error
        return TransientError.wrap(error)

    async def consume(self) -> None:
        """Stream :meth:`recv` as :class:`MessageReceived` until the wire drops."""
        if self.first_message_timeout is not None:
            await asyncio.wait_for(self.consume_one(), self.first_message_timeout)
        while not self.stopped:
            await self.consume_one()

    async def consume_one(self) -> None:
        message = await self.recv()
        if message is not None:
            await self.events.emit(
                MessageReceived(self.generation, message, message_qos(message))
            )

    async def teardown(self) -> None:
        """Close the wire without letting :meth:`aclose` break the loop."""
        try:
            await self.aclose()
        except Exception:  # noqa: BLE001 -- aclose must never break supervision
            self.logger.debug("wire %s aclose failed", self.url, exc_info=True)
