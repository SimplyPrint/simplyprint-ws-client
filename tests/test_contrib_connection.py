"""``contrib.connection`` is the threaded pool/lifecycle promoted from an
integration. Two contracts are pinned here:

* the optional transport libraries (paho-mqtt, websocket-client) and aiohttp stay
  *lazy* -- importing the package must not drag them into the import graph, so a
  base install without the extras still imports cleanly; and
* the pool routing/lifecycle works with a plain fake client -- no brand present.
"""

import subprocess
import sys
import time

from simplyprint_ws_client.contrib.connection import (
    ClientBucket,
    ConnectionManager,
    ConnectionState,
    PooledConnection,
    Watchdog,
    now_ms,
)
from simplyprint_ws_client.contrib.connection.pool import (
    ConnectionAttemptsBoundedInterval,
)


def _import_is_clean(import_line: str, *forbidden: str) -> None:
    """Run ``import_line`` in a fresh interpreter; assert no forbidden module loaded."""
    checks = "\n".join(
        f"assert {mod!r} not in sys.modules, {mod!r}" for mod in forbidden
    )
    code = f"import sys\n{import_line}\n{checks}\nprint('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"import-purity failed for {import_line!r}:\n{result.stderr}"
    )


def test_connection_import_does_not_load_paho():
    _import_is_clean(
        "import simplyprint_ws_client.contrib.connection", "paho.mqtt.client", "paho"
    )


def test_transport_import_does_not_load_optional_libs():
    _import_is_clean(
        "import simplyprint_ws_client.contrib.transport", "websocket", "aiohttp"
    )


def test_contrib_init_is_import_free():
    # Importing the package root must not pull the high-level printer_client (which
    # imports core) nor the leaf submodules -- contrib/__init__.py is empty on purpose.
    _import_is_clean(
        "import simplyprint_ws_client.contrib",
        "simplyprint_ws_client.contrib.printer_client",
        "simplyprint_ws_client.contrib.connection",
        "simplyprint_ws_client.contrib.transport",
    )


def test_connection_state_usable():
    assert ConnectionState.ONLINE.is_usable
    assert not ConnectionState.OFFLINE.is_usable
    assert not ConnectionState.AUTH_FAILED.is_usable
    assert not ConnectionState.CONFIG_INVALID.is_usable


def test_watchdog_expires_after_timeout():
    # The watchdog thread polls on a 1s granularity, so the observation window
    # must clear timeout + one poll tick.
    wd = Watchdog(0.3, name="test-wd")
    wd.start()
    try:
        assert not wd.expired
        time.sleep(1.6)
        assert wd.expired
    finally:
        wd.stop()


def test_watchdog_does_not_expire_before_timeout():
    wd = Watchdog(5.0, name="test-wd-long")
    wd.start()
    try:
        wd.reset_sync()
        time.sleep(1.6)
        assert not wd.expired
    finally:
        wd.stop()




class _FakeWorker:
    def __init__(self):
        self.events = []

    def emit_sync(self, event, *args, **kwargs):
        self.events.append((event, args))


class _FakeClient:
    """A minimal :class:`PoolClient` -- no brand types anywhere."""

    def __init__(self, host: str, topic: str):
        import logging

        self.config = host  # the params_factory just echoes this
        self.logger = logging.getLogger("fake-client")
        self.event_bus_worker = _FakeWorker()
        self.last_message_at = now_ms()
        self.keepalive_attempts = ConnectionAttemptsBoundedInterval.create_variable()
        self._topic = topic
        self._connected = True

    @property
    def report_topic(self) -> str:
        return self._topic

    @property
    def connected(self) -> bool:
        return self._connected


class _FakeConnection(PooledConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_connected = True
        self.stopped = False

    @property
    def connected(self) -> bool:
        return self._is_connected

    def stop(self):
        self.stopped = True
        super().stop()


class _FakeManager(ConnectionManager):
    connected_event = "connected"
    disconnected_event = "disconnected"
    message_event = "message"
    params_factory = staticmethod(lambda config: config)
    wildcard_topics = True

    def __init__(self):
        super().__init__()
        self.keepalives = []
        self.refreshes = []
        self.unsubscribes = []

    def _create_connection(self, params):
        return _FakeConnection(
            self.bucket,
            params,
            connected_event=self.connected_event,
            disconnected_event=self.disconnected_event,
        )

    def _refresh_subscription(self, client, connection):
        self.refreshes.append(client)

    def _send_keepalive(self, client, connection):
        self.keepalives.append(client)

    def _unsubscribe(self, client, connection):
        self.unsubscribes.append(client)


def test_client_bucket_routes_by_params_and_topic():
    bucket = ClientBucket(lambda config: config, wildcard_topics=True)
    a = _FakeClient("10.0.0.1", "devices/a/report")
    b = _FakeClient("10.0.0.1", "devices/b/report")
    c = _FakeClient("10.0.0.2", "devices/c/#")

    for client in (a, b, c):
        bucket.add(client)

    # Two clients share host 10.0.0.1 -> same params group.
    assert set(bucket.get_from_params("10.0.0.1")) == {a, b}
    assert bucket.get_from_params("10.0.0.2") == [c]

    # Exact topic routing.
    assert bucket.get_from_topic("devices/a/report") is a
    # Wildcard prefix routing for c's "devices/c/#" subscription.
    assert bucket.get_from_topic("devices/c/anything") is c

    bucket.remove(a)
    assert bucket.get_from_params("10.0.0.1") == [b]
    assert bucket.get_from_topic("devices/a/report") is None


def test_manager_pools_connections_and_emits_connected():
    manager = _FakeManager()
    try:
        a = _FakeClient("10.0.0.1", "devices/a/report")
        b = _FakeClient("10.0.0.1", "devices/b/report")  # shares a's connection
        c = _FakeClient("10.0.0.2", "devices/c/report")

        manager.add_client(a)
        manager.add_client(b)
        manager.add_client(c)

        # Two distinct params -> two physical connections, shared by params.
        assert len(manager.connections) == 2
        # add_client emits the connected event because the fake connection is up.
        assert ("connected", ()) in a.event_bus_worker.events

        # Removing a leaves b on the shared connection (not torn down).
        manager.remove_client(a)
        assert a in manager.unsubscribes
        assert len(manager.connections) == 2
        # Removing the last client on 10.0.0.2 tears that connection down.
        manager.remove_client(c)
        assert len(manager.connections) == 1
    finally:
        manager.stop()


def test_manager_keepalive_pokes_quiet_clients():
    manager = _FakeManager()
    try:
        a = _FakeClient("10.0.0.1", "devices/a/report")
        manager.add_client(a)
        # Make the client look quiet (last heard from long ago).
        a.last_message_at = now_ms() - manager.keepalive_timeout_ms - 1
        manager.keepalive_check()
        assert a in manager.refreshes
        assert a in manager.keepalives
    finally:
        manager.stop()


def test_pooled_connection_fans_events_to_all_clients():
    manager = _FakeManager()
    try:
        a = _FakeClient("10.0.0.1", "devices/a/report")
        b = _FakeClient("10.0.0.1", "devices/b/report")
        manager.add_client(a)
        manager.add_client(b)
        connection = manager.get_connection_from_client(a)

        a.event_bus_worker.events.clear()
        b.event_bus_worker.events.clear()
        connection.handle_disconnected(reason="bye")

        assert ("disconnected", ("bye",)) in a.event_bus_worker.events
        assert ("disconnected", ("bye",)) in b.event_bus_worker.events
    finally:
        manager.stop()
