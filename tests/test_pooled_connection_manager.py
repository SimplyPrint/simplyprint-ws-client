"""Brand-free tests for PooledConnectionManager (the v2 manager on the lease).

A fake Pool hands out fake Connection leases; the manager must wire each lease's
on_message/on_connected/on_disconnected to the client's event bus, keep deferred
registrations retryable, poke idle clients on keepalive (and declare a disconnect
after the bound), and close leases on remove/stop.
"""

import logging

import pytest

from simplyprint_ws_client.contrib.connection.manager import (
    KEEPALIVE_TIMEOUT_MS,
    ConnectionAttemptsBoundedInterval,
    PooledConnectionManager,
    now_ms,
)
from simplyprint_ws_client.contrib.connection.transport import (
    ConnectSuspect,
    Disconnected,
)
from simplyprint_ws_client.events import EventBus


class FakeConnection:
    def __init__(self, params, route):
        self.params = params
        self.route = route
        self.closed = False
        self._connected = False
        self._msg, self._con, self._dis, self._sus = [], [], [], []

    def on_message(self, handler):
        self._msg.append(handler)
        return lambda: self._msg.remove(handler)

    def on_connected(self, handler):
        self._con.append(handler)
        return lambda: self._con.remove(handler)

    def on_disconnected(self, handler):
        self._dis.append(handler)
        return lambda: self._dis.remove(handler)

    def on_suspect(self, handler):
        self._sus.append(handler)
        return lambda: self._sus.remove(handler)

    def fire_suspect(self):
        for h in list(self._sus):
            h(ConnectSuspect())

    @property
    def connected(self):
        return self._connected

    def close(self):
        self.closed = True

    # -- test drivers --
    def fire_connected(self):
        self._connected = True
        for h in list(self._con):
            h()

    def fire_message(self, msg):
        for h in list(self._msg):
            h(msg)

    def fire_disconnected(self, reason="dropped"):
        self._connected = False
        for h in list(self._dis):
            h(Disconnected(reason=reason))


class FakePool:
    def __init__(self):
        self.conns = []
        self.stopped = False
        self.submitted = []

    def connect(self, params, *, route=None):
        conn = FakeConnection(params, route)
        self.conns.append(conn)
        return conn

    def submit_to_consumer(self, coro_factory, *, coalesce_key=None):
        self.submitted.append((coro_factory, coalesce_key))

    def stop(self):
        self.stopped = True


class FakeClient:
    def __init__(self, topic, params, connected=False):
        self.config = {"params": params}
        self.logger = logging.getLogger("test-pool-client")
        self.event_bus = EventBus()
        self.last_message_at = 0
        self.keepalive_attempts = ConnectionAttemptsBoundedInterval.create_variable(0)
        self.keepalives = []
        self._topic = topic
        self._connected = connected

    @property
    def report_topic(self):
        return self._topic

    @property
    def connected(self):
        return self._connected


def _manager(pool):
    class _Mgr(PooledConnectionManager):
        connected_event = "connected"
        disconnected_event = "disconnected"
        message_event = "message"
        params_factory = staticmethod(lambda config: config.get("params"))

        def _make_pool(self):
            return pool

        def _send_keepalive(self, client):
            client.keepalives.append(1)

    return _Mgr()


def test_add_client_routes_messages_and_lifecycle():
    pool = FakePool()
    mgr = _manager(pool)
    client = FakeClient("printer/a/report", params=("broker", 8883))

    seen = []
    client.event_bus.on("connected", lambda: seen.append("up"))
    client.event_bus.on("message", seen.append)
    client.event_bus.on("disconnected", lambda r: seen.append(("down", r)))

    mgr.add_client(client)
    conn = pool.conns[0]
    assert conn.route == "printer/a/report"  # leased scoped to the client's topic

    conn.fire_connected()
    conn.fire_message("m1")
    conn.fire_disconnected("bye")

    assert seen == ["up", "m1", ("down", "bye")]


def test_deferred_registration_retries_until_params_ready():
    pool = FakePool()
    mgr = _manager(pool)
    client = FakeClient("t/a", params=None)  # params_factory -> None -> deferred

    mgr.request_registration(client)
    assert client not in mgr._leases  # deferred, not leased
    assert pool.conns == []

    mgr.reconcile_registrations()
    assert client not in mgr._leases  # still not ready

    client.config["params"] = ("broker", 8883)
    mgr.reconcile_registrations()
    assert client in mgr._leases  # landed once params became valid


def test_keepalive_pokes_idle_then_disconnects_after_bound():
    pool = FakePool()
    mgr = _manager(pool)
    client = FakeClient("t/a", params=("b", 1), connected=True)

    disconnects = []
    client.event_bus.on("disconnected", disconnects.append)
    mgr.add_client(client)
    pool.conns[0]._connected = True

    for _ in range(10):
        # Clearly idle, relative to now_ms() (robust to any time source).
        client.last_message_at = now_ms() - 10 * KEEPALIVE_TIMEOUT_MS
        mgr.keepalive_check()
        if disconnects:
            break

    assert client.keepalives  # poked the idle client before giving up
    assert disconnects == [
        "Keepalive failed enough times"
    ]  # then declared a disconnect


def test_suspect_reaches_the_brand_hook():
    pool = FakePool()
    suspects = []

    class _Mgr(PooledConnectionManager):
        connected_event = "connected"
        disconnected_event = "disconnected"
        message_event = "message"
        params_factory = staticmethod(lambda config: config.get("params"))

        def _make_pool(self):
            return pool

        def _on_suspect(self, client, event):
            suspects.append(client)

    mgr = _Mgr()
    client = FakeClient("t/a", params=("b", 1))
    mgr.add_client(client)

    pool.conns[0].fire_suspect()
    assert suspects == [client]


def test_remove_and_stop_close_leases():
    pool = FakePool()
    mgr = _manager(pool)
    a = FakeClient("t/a", params=("b", 1))
    b = FakeClient("t/b", params=("b", 1))
    mgr.add_client(a)
    mgr.add_client(b)
    conn_a, conn_b = pool.conns

    mgr.remove_client(a)
    assert conn_a.closed is True
    assert a not in mgr._leases

    mgr.stop()
    assert conn_b.closed is True
    assert pool.stopped is True


def test_remove_is_not_resurrected_by_reconcile():
    # A removed client must stay gone: the reconcile sweep retries only *wanted*
    # registrations, and remove drops the client from the wanted set.
    pool = FakePool()
    mgr = _manager(pool)
    client = FakeClient("t/a", params=("b", 1))

    mgr.request_registration(client)
    assert client in mgr._leases

    mgr.remove_client(client)
    assert client not in mgr._leases

    mgr.reconcile_registrations()
    assert client not in mgr._leases  # not re-added
    assert len(pool.conns) == 1  # no second lease was leased


@pytest.mark.asyncio
async def test_submit_to_consumer_delegates_to_pool():
    pool = FakePool()
    mgr = _manager(pool)

    async def work():
        return None

    mgr.submit_to_consumer(work, coalesce_key="k")
    assert pool.submitted == [(work, "k")]
