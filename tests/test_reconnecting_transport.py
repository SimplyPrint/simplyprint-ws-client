"""Brand-free tests for the composable ``ReconnectingTransport`` transport.

A ``_FakeLink`` lets a test drive one connection attempt deterministically (no real
socket): ``open`` may fail, ``recv`` waits on an inbox the test feeds (a queued
exception simulates a drop), ``send`` records. These pin the reconnection state
machine: connect/stream/drop, a fresh link per attempt, exactly one generation bump
per ended attempt, bounded suspect, first-message liveness, and clean stop.
"""

import asyncio
from typing import Any, Optional

import pytest

from simplyprint_ws_client.contrib.connection.events import (
    Connected,
    ConnectionSuspect,
    Disconnected,
    MessageReceived,
    StateChanged,
)
from simplyprint_ws_client.contrib.connection.reconnect import Link, ReconnectingTransport
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.shared.utils.backoff import ConstantBackoff
from simplyprint_ws_client.shared.utils.bounded_variable import BoundedInterval


class _FakeLink(Link):
    def __init__(self, *, fail_open: bool = False) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.opened = 0
        self.closed = 0
        self.sent = []
        self._open = False
        self._fail_open = fail_open

    async def open(self) -> None:
        self.opened += 1
        if self._fail_open:
            raise OSError("open failed")
        self._open = True

    async def recv(self) -> Optional[Any]:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            self._open = False
            raise item
        return item

    async def send(self, payload: Any) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed += 1
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open


def _factory(links: list, **kw):
    def make() -> _FakeLink:
        link = _FakeLink(**kw)
        links.append(link)
        return link

    return make


async def _wait(pred, timeout: float = 2.0) -> None:
    for _ in range(int(timeout / 0.005)):
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


@pytest.mark.asyncio
async def test_connects_streams_and_emits():
    links: list = []
    t = ReconnectingTransport("x", _factory(links), backoff=ConstantBackoff(0))
    seen: list = []
    t.events.on(Connected, lambda e: seen.append("up"))
    t.events.on(StateChanged, lambda e: seen.append(("state", e.state)))
    t.events.on(MessageReceived, lambda e: seen.append(("msg", e.payload)))
    t.start()
    try:
        await _wait(lambda: bool(links) and links[0].is_open)
        links[0].inbox.put_nowait("hello")
        await _wait(lambda: ("msg", "hello") in seen)
        assert "up" in seen
        assert ("state", ConnectionState.ONLINE) in seen
        assert t.connected
        assert t.generation == 0  # connected, not yet dropped
        await t.send(b"cmd")
        assert links[0].sent == [b"cmd"]
    finally:
        t.stop()
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_drop_reconnects_fresh_and_bumps_generation_once():
    links: list = []
    t = ReconnectingTransport("x", _factory(links), backoff=ConstantBackoff(0))
    downs: list = []
    t.events.on(Disconnected, lambda e: downs.append(e.transient))
    t.start()
    try:
        await _wait(lambda: bool(links) and links[0].is_open)
        links[0].inbox.put_nowait(RuntimeError("drop"))  # end attempt 1
        await _wait(lambda: len(links) >= 2 and links[1].is_open)
        assert downs and downs[0] is True  # transient
        assert t.generation == 1  # exactly one bump for one drop
        assert links[0].closed == 1  # fresh link per attempt; old one closed
        assert links[1] is not links[0]
    finally:
        t.stop()
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_first_message_timeout_drops_and_reconnects():
    links: list = []
    t = ReconnectingTransport(
        "x", _factory(links), backoff=ConstantBackoff(0), first_message_timeout=0.05
    )
    t.start()
    try:
        # No message is ever fed; each attempt opens fine, then the first-message
        # timeout drops it -> reconnect.
        await _wait(lambda: len(links) >= 2)
        assert t.generation >= 1
        assert (
            links[0].opened == 1 and links[0].closed == 1
        )  # opened, then dropped by liveness
    finally:
        t.stop()
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_connect_suspect_is_bounded():
    links: list = []
    t = ReconnectingTransport(
        "x",
        _factory(links, fail_open=True),
        backoff=ConstantBackoff(0.01),
        suspect_after=BoundedInterval(3, 1, 0),
    )
    suspects: list = []
    t.events.on(ConnectionSuspect, lambda e: suspects.append(1))
    t.start()
    try:
        await _wait(lambda: len(links) >= 8)  # many failed attempts
        t.stop()
        await asyncio.sleep(0.02)
        assert len(suspects) >= 1  # fires periodically
        assert len(suspects) < len(links)  # but bounded, not on every drop
    finally:
        t.stop()
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_open_failure_reconnects():
    links: list = []
    t = ReconnectingTransport(
        "x", _factory(links, fail_open=True), backoff=ConstantBackoff(0.01)
    )
    t.start()
    try:
        await _wait(lambda: len(links) >= 2)  # keeps retrying despite open failing
        assert not t.connected
        assert t.generation >= 1
    finally:
        t.stop()
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_stop_cancels_and_halts():
    links: list = []
    t = ReconnectingTransport("x", _factory(links), backoff=ConstantBackoff(0))
    t.start()
    await _wait(lambda: bool(links) and links[0].is_open)
    t.stop()
    await asyncio.sleep(0.03)
    n = len(links)
    await asyncio.sleep(0.03)
    assert len(links) == n  # no new attempts after stop
    assert not t.connected
