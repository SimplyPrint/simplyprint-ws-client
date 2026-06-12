"""The supervised reconnect loop -- one asyncio task that keeps a wire alive.

This is the heart of the async wire families. A :class:`Reconnecting` transport is
itself an asyncio-native wire: a concrete subclass fills FOUR hooks on itself --
:meth:`~Reconnecting.open`, :meth:`~Reconnecting.recv`, :meth:`~Reconnecting.write`,
:meth:`~Reconnecting.aclose` -- and this base supplies everything around them: a
single supervision task, the connect/consume/drop/backoff cycle, the generation
counter, state, and the lifecycle events. There is no separate "link" object; the
hooks are methods on the concrete impl, so a wire is one class top to bottom.

The loop always keeps retrying. The :class:`~simplyprint_ws_client.wire.policy.RetryPolicy`
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

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.wire.events import (
    Connected,
    Connecting,
    WireEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.errors import TransportError
from simplyprint_ws_client.wire.messages import message_qos
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.state import ConnectionState
from simplyprint_ws_client.wire.transport import (
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

    #: Default bound on one :meth:`open` attempt. Wire libraries bound their
    #: own connects inconsistently (some not at all); this is the supervise-
    #: level backstop so a hung open is a failed attempt, never CONNECTING
    #: forever.
    DEFAULT_OPEN_TIMEOUT = 60.0

    def __init__(
        self,
        url: yarl.URL,
        policy: Optional[RetryPolicy] = None,
        provider: Optional[EventLoopProvider[asyncio.AbstractEventLoop]] = None,
        *,
        first_message_timeout: Optional[float] = None,
        open_timeout: Optional[float] = DEFAULT_OPEN_TIMEOUT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.url = url
        self.state = ConnectionState.DISCONNECTED
        self.generation = 0
        self.events: EventBus[WireEvent] = EventBus()
        self.policy = policy or RetryPolicy()
        self.provider = provider or EventLoopProvider.default()
        self.first_message_timeout = first_message_timeout
        self.open_timeout = open_timeout
        self.logger = logger or logging.getLogger("wire.reconnect")
        self.live = False
        self.stopped = False
        self.gave_up = False
        self.task: Optional[asyncio.Task] = None
        #: Armed while an attempt's consume loop runs; a failed send sets it via
        #: :meth:`trip` so the attempt ends even when recv never gets to raise.
        self._tripped: Optional[asyncio.Event] = None
        self._trip_reason: Optional[Exception] = None

    @abstractmethod
    async def open(self) -> None:
        """Establish the live wire. Raise to end the attempt and trigger a retry.

        Raise :class:`~simplyprint_ws_client.wire.transport.TransientError`
        / :class:`~simplyprint_ws_client.wire.transport.FatalError` to tag
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
        loop = self.provider.event_loop

        def spawn() -> None:
            if self.stopped or (self.task is not None and not self.task.done()):
                return
            self.task = loop.create_task(self.supervise())
            # A supervision task must never die of an exception; if one ever
            # does, it must at least say so -- a silent death is a permanent,
            # invisible outage (``supervising()`` turns honest only when asked).
            self.task.add_done_callback(self._log_supervise_death)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is loop:
            spawn()
        else:
            # ``create_task`` is not thread-safe; off-loop callers (sync front
            # doors, the paho thread) hop home first.
            loop.call_soon_threadsafe(spawn)

    def _log_supervise_death(self, task: "asyncio.Task") -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.logger.error(
                "wire %s supervision died unexpectedly", self.url, exc_info=error
            )

    async def stop(self) -> None:
        # An intentional stop is silent - no Disconnected is emitted (only a
        # *drop* speaks). Lease.close() settles its own ready() waiters.
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
        generation = self.generation
        try:
            await self.write(message)
        except Exception as error:
            # A wire that fails a write is dead or dying. The supervise task
            # normally notices via recv, but if it is busy dispatching an
            # inbound message it would never look -- trip it so the attempt
            # tears down and the loop reconnects.
            self.trip(generation, error)
            raise

    def trip(self, generation: int, reason: Exception) -> None:
        """Report a dead wire from outside the supervise task (a failed send).

        Ends the current attempt -- including one wedged in inbound dispatch,
        whose in-flight handler is cancelled -- so the loop tears down and
        reconnects. ``generation`` guards against a stale report: a send that
        captured a previous attempt's wire must not kill the new, healthy one.
        Thread-safe via the same home-loop hop as :meth:`start`.
        """

        def fire() -> None:
            tripped = self._tripped
            if tripped is None or generation != self.generation or self.stopped:
                return
            self._trip_reason = reason
            tripped.set()

        loop = self.provider.event_loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            fire()
        else:
            try:
                loop.call_soon_threadsafe(fire)
            except RuntimeError:
                pass  # loop already closed (shutdown); the wire is done anyway

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
                    if self.open_timeout is not None:
                        try:
                            await asyncio.wait_for(self.open(), self.open_timeout)
                        except asyncio.TimeoutError:
                            raise TransientError(
                                f"open timed out after {self.open_timeout:.0f}s"
                            )
                    else:
                        await self.open()
                    # Arm the tripwire BEFORE announcing Connected: a send made
                    # from inside a Connected handler (the classic "hello on
                    # connect") must be able to trip this very attempt.
                    self._trip_reason = None
                    self._tripped = asyncio.Event()
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
                    self._tripped = None
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
        try:
            await self.events.emit(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a listener bug is not a wire failure
            # A raising Connecting/Disconnected listener must neither kill the
            # supervision task (a silent, permanent outage) nor count as a wire
            # drop (churning a healthy link). Log it and keep supervising.
            self.logger.exception(
                "wire %s %s listener failed", self.url, type(event).__name__
            )

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
        """Stream :meth:`recv` as :class:`MessageReceived` until the wire drops
        or a failed send :meth:`trip` s the attempt.

        The stream and the tripwire race: a trip cancels the stream task --
        including a dispatch handler stuck mid-await, which would otherwise
        keep a dead wire looking ``CONNECTED`` forever -- and ends the attempt
        with the send's failure, so the loop tears down and reconnects.

        The tripwire is armed by :meth:`supervise` before ``Connected`` is
        announced, so even a send made from a Connected handler can trip.
        """
        tripped = self._tripped
        if tripped is None:  # pragma: no cover -- supervise always arms first
            self._tripped = tripped = asyncio.Event()
        stream = asyncio.ensure_future(self.consume_stream())
        trip = asyncio.ensure_future(tripped.wait())
        try:
            done, _pending = await asyncio.wait(
                {stream, trip}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            self._tripped = None
            stream.cancel()
            trip.cancel()
            # Settle both so a cancellation or late exception is never
            # left unobserved (the gather itself must survive our own
            # cancellation having already hit the children).
            await asyncio.gather(stream, trip, return_exceptions=True)
        if stream in done:
            stream.result()  # re-raise the drop that ended the stream
            return
        raise self.transport_error(
            self._trip_reason or TransientError("send failed on a dead wire")
        )

    async def consume_stream(self) -> None:
        """The inbound stream: :meth:`consume_one` until stop or drop."""
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
