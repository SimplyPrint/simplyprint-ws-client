"""One printer's failure never takes the others down.

A process hosts 100-500 printers. A broken driver, a corrupt or incompatible
stored config, or a client that raises in ``tick``/``connect``/``teardown`` is
that printer's problem: it is logged once with its id, recorded where it can be
inspected, its stored config is never deleted, and every other printer runs.
"""

import asyncio
import json
import logging
import threading
from datetime import timedelta
from typing import Dict

import pytest

from simplyprint_ws_client import (
    ClientApp,
    ClientSettings,
    IntegrationId,
    IntegrationSpec,
    PrinterConfig,
)
from simplyprint_ws_client.core.client import Client
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.config import JsonConfigManager, SQLiteConfigManager
from simplyprint_ws_client.core.config.manager import UnreadableConfig
from simplyprint_ws_client.core.manager import ClientList
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.integration.discovery import DiscoveryService
from simplyprint_ws_client.integration.spec import ProductMetadata
from tests._fakes import FakeTransport

_METADATA = ProductMetadata(
    display_name="Test",
    image_url="/test.png",
    supported_transports=(),
    capabilities=(),
)


def _config(unique_id: str, pk: int = 1) -> PrinterConfig:
    return PrinterConfig(id=pk, token="token-%s" % unique_id, unique_id=unique_id)


# ---------------------------------------------------------------------------
# App: building clients at startup and at runtime
# ---------------------------------------------------------------------------


class _OfflineClient(Client[PrinterConfig]):
    """Never asks for a SimplyPrint connection; reports when it has started."""

    started: Dict[str, threading.Event] = {}

    def __init__(self, config, *, context):
        super().__init__(config, context=context)
        # Inactive: the scheduler runs init/tick but never allocates, so these
        # tests need no server.
        self._should_be_allocated = False

    async def init(self):
        _OfflineClient.started.setdefault(
            str(self.unique_id), threading.Event()
        ).set()


def _factory(broken_ids):
    def factory(config, *, context):
        if config.unique_id in broken_ids:
            raise RuntimeError("driver exploded for %s" % config.unique_id)
        return _OfflineClient(config, context=context)

    return factory


def _offline_transport(url, provider, logger):
    return FakeTransport(url, provider, logger)


def _app(factory) -> ClientApp:
    integration = IntegrationSpec(
        id=IntegrationId("test"),
        client_factory=factory,
        config_factory=PrinterConfig,
        metadata=_METADATA,
    )
    return ClientApp(
        ClientSettings(integrations=(integration,), camera_workers=None),
        discovery_service=DiscoveryService(),
        account_providers={},
        transport_factory=_offline_transport,
    )


def _wait_started(unique_id: str) -> None:
    event = _OfflineClient.started.setdefault(unique_id, threading.Event())
    assert event.wait(5.0), "client %s never started" % unique_id


def test_one_raising_factory_does_not_stop_the_others_at_startup(caplog):
    _OfflineClient.started.clear()
    app = _app(_factory({"broken"}))
    configs = [_config("good-a", 1), _config("broken", 2), _config("good-b", 3)]
    for config in configs:
        app.config_manager.persist(config)

    try:
        with caplog.at_level(logging.ERROR):
            app.run_detached()
            _wait_started("good-a")
            _wait_started("good-b")

        assert "good-a" in app.client_list
        assert "good-b" in app.client_list
        assert "broken" not in app.client_list
        assert app.failed_client_ids() == ["broken"]
        failure = app.failed_clients["broken"]
        assert failure.stage == "build"
        assert failure.integration_id == "test"
        assert "driver exploded" in failure.error
        # The config is never deleted; it is retried on the next add/restart.
        assert app.config_manager.contains(configs[1])
        # Logged once, with the printer's id.
        mentions = [r for r in caplog.records if "broken" in r.getMessage()]
        assert len(mentions) == 1
        # No half-registered client: nothing scheduled for it.
        assert "broken" not in app.scheduler._tasks
        assert app.is_running()
    finally:
        app.stop()


def test_runtime_add_failure_is_recorded_and_retried_by_the_next_add():
    attempts = {"n": 0}

    def flaky(config, *, context):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("incompatible config")
        return _OfflineClient(config, context=context)

    app = _app(flaky)
    config = _config("flaky")
    try:
        with pytest.raises(RuntimeError):
            app.add(config)

        assert app.failed_client_ids() == ["flaky"]
        assert app.config_manager.contains(config)
        assert "flaky" not in app.client_list
        assert app.scheduler._tasks == {}

        client = app.add(config)

        assert client.unique_id == "flaky"
        assert app.failed_client_ids() == []
    finally:
        app.stop()


def test_removing_a_printer_that_failed_to_start_deletes_its_config():
    app = _app(_factory({"broken"}))
    config = _config("broken")
    try:
        with pytest.raises(RuntimeError):
            app.add(config)

        app.remove(config)

        assert not app.config_manager.contains(config)
        assert app.failed_client_ids() == []
    finally:
        app.stop()


def test_unreadable_stored_config_is_recorded_and_the_others_start():
    _OfflineClient.started.clear()
    app = _app(_factory(set()))
    app.config_manager.persist(_config("good", 1))
    app.config_manager.unreadable.append(
        UnreadableConfig(
            raw={"id": "nope", "unique_id": "corrupt"},
            error="ValidationError: id",
            unique_id="corrupt",
        )
    )
    try:
        app.run_detached()
        _wait_started("good")

        assert "good" in app.client_list
        assert app.failed_client_ids() == ["corrupt"]
        assert app.failed_clients["corrupt"].stage == "config"
    finally:
        app.stop()


# ---------------------------------------------------------------------------
# Config stores: one bad entry is skipped on load and kept on flush
# ---------------------------------------------------------------------------

_GOOD_A = {"id": 1, "token": "token-a", "unique_id": "a"}
_GOOD_B = {"id": 3, "token": "token-b", "unique_id": "b"}
_BAD = {"id": "not-a-number", "token": "token-bad", "unique_id": "bad"}


def test_json_store_skips_a_corrupt_entry_and_never_deletes_it(tmp_path):
    path = tmp_path / "isolation.json"
    path.write_text(json.dumps([_GOOD_A, _BAD, _GOOD_B]))

    manager = JsonConfigManager(name="isolation", base_directory=str(tmp_path))

    assert sorted(c.unique_id for c in manager.get_all()) == ["a", "b"]
    assert [entry.unique_id for entry in manager.unreadable] == ["bad"]

    manager.by_unique_id("a").name = "renamed"
    manager.flush()

    written = json.loads(path.read_text())
    assert _BAD in written
    assert any(e.get("name") == "renamed" for e in written if isinstance(e, dict))
    assert len(written) == 3


class _FragileConfig(PrinterConfig):
    def as_json(self) -> str:
        if self.name == "explode":
            raise ValueError("cannot serialize")
        return super().as_json()


def test_json_store_keeps_a_config_that_stops_serializing(tmp_path):
    path = tmp_path / "fragile.json"
    path.write_text(json.dumps([_GOOD_A, _GOOD_B]))

    manager = JsonConfigManager(
        name="fragile", base_directory=str(tmp_path), config_t=_FragileConfig
    )
    manager.by_unique_id("a").name = "explode"
    manager.by_unique_id("b").name = "saved"
    manager.flush()

    written = {e["unique_id"]: e for e in json.loads(path.read_text())}
    assert written["a"] == _GOOD_A  # its last good (loaded) form
    assert written["b"]["name"] == "saved"  # the others still save


def test_sqlite_store_skips_unreadable_rows_and_keeps_them(tmp_path):
    manager = SQLiteConfigManager(name="isolation", base_directory=str(tmp_path))
    try:
        rows = [
            (1, "token-a", json.dumps(_GOOD_A)),
            (2, "token-bad", json.dumps(_BAD)),
            (4, "token-garbage", "{{ not json"),
        ]
        manager.db.executemany(
            "INSERT INTO printers (pk, sk, data) VALUES (?, ?, ?)", rows
        )
        manager.db.commit()

        manager.clear()
        manager.load()

        assert [c.unique_id for c in manager.get_all()] == ["a"]
        assert len(manager.unreadable) == 2

        manager.flush()
        manager.flush(manager.by_unique_id("a"))

        count = manager.db.execute("SELECT COUNT(*) FROM printers").fetchone()[0]
        assert count == 3
    finally:
        manager.delete_storage()


# ---------------------------------------------------------------------------
# Scheduler: tick, connect, dispatch and teardown are per client
# ---------------------------------------------------------------------------


class _RecordingClient(Client[PrinterConfig]):
    def __init__(self, config, *, context):
        super().__init__(config, context=context)
        self.tick_calls = 0
        self.teardown_calls = 0

    async def tick(self, delta: timedelta):
        self.tick_calls += 1

    async def teardown(self):
        self.teardown_calls += 1


class _FakeManager:
    #: Not replaying stored printers: the scheduler may place them.
    is_replaying = False

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.allocated = set()
        self.connections = ()

    def is_allocated(self, client):
        return client.unique_id in self.allocated

    async def allocate(self, client):
        if client.unique_id in self.failing:
            raise ConnectionError("connect failed for %s" % client.unique_id)
        self.allocated.add(client.unique_id)

    async def deallocate(self, client):
        self.allocated.discard(client.unique_id)

    def stop(self):
        pass


@pytest.fixture
def scheduler():
    scheduler = Scheduler(
        client_list=ClientList(), settings=ClientSettings(camera_workers=None)
    )
    scheduler.manager = _FakeManager()
    return scheduler


def _client(unique_id: str, pk: int = 1) -> _RecordingClient:
    return _RecordingClient(_config(unique_id, pk), context=ClientContext())


async def _run_scheduled(scheduler: Scheduler) -> None:
    await asyncio.gather(
        *(task.task for task in scheduler._tasks.values() if task.task is not None)
    )


def _other_tasks():
    current = asyncio.current_task()
    return [task for task in asyncio.all_tasks() if task is not current]


@pytest.mark.asyncio
async def test_a_raising_tick_does_not_affect_another_client(scheduler):
    scheduler.use_running_loop()
    bad, good = _client("bad", 1), _client("good", 2)

    async def boom(delta):
        raise RuntimeError("tick exploded")

    bad.tick = boom
    scheduler.client_list.add(bad)
    scheduler.client_list.add(good)

    scheduler._process_clients()
    await _run_scheduled(scheduler)

    assert good.tick_calls == 1
    assert scheduler.manager.is_allocated(good)
    assert _other_tasks() == []


@pytest.mark.asyncio
async def test_a_raising_connect_does_not_affect_another_client(scheduler):
    scheduler.use_running_loop()
    scheduler.manager = _FakeManager(failing={"bad"})
    bad, good = _client("bad", 1), _client("good", 2)
    scheduler.client_list.add(bad)
    scheduler.client_list.add(good)

    scheduler._process_clients()
    await _run_scheduled(scheduler)

    assert scheduler.manager.is_allocated(good)
    assert not scheduler.manager.is_allocated(bad)
    assert bad.tick_calls == 1  # its own lifecycle still ran
    assert _other_tasks() == []


class _BrokenStateClient(_RecordingClient):
    @property
    def has_changes(self) -> bool:
        raise RuntimeError("state exploded")


@pytest.mark.asyncio
async def test_a_client_that_cannot_be_dispatched_does_not_starve_the_rest(
    scheduler, caplog
):
    scheduler.use_running_loop()
    bad = _BrokenStateClient(_config("bad", 1), context=ClientContext())
    bad.initialized = True  # reach the has_changes check
    good = _client("good", 2)
    # Insertion order puts the broken client first: an unguarded loop would
    # abort before ever reaching the good one.
    scheduler.client_list.add(bad)
    scheduler.client_list.add(good)

    with caplog.at_level(logging.ERROR):
        scheduler._process_clients()
        await _run_scheduled(scheduler)
        scheduler._last_ticked.clear()
        scheduler._process_clients()
        await _run_scheduled(scheduler)

    assert good.tick_calls == 2
    errors = [r for r in caplog.records if "dispatching client bad" in r.getMessage()]
    assert len(errors) == 1  # once per failure streak, not once per pass


@pytest.mark.asyncio
async def test_a_raising_teardown_on_stop_does_not_affect_the_others(scheduler):
    scheduler.use_running_loop()
    bad, good = _client("bad", 1), _client("good", 2)

    async def boom():
        raise RuntimeError("teardown exploded")

    bad.teardown = boom
    scheduler.client_list.add(bad)
    scheduler.client_list.add(good)

    await scheduler._teardown()  # must not raise

    assert good.teardown_calls == 1
    assert _other_tasks() == []


@pytest.mark.asyncio
async def test_a_raising_teardown_on_removal_does_not_affect_the_others(scheduler):
    scheduler.use_running_loop()
    bad, good = _client("bad", 1), _client("good", 2)

    async def boom():
        raise RuntimeError("teardown exploded")

    bad.teardown = boom
    for client in (bad, good):
        scheduler.client_list.add(client)
        scheduler.remove(client)

    scheduler._process_to_delete()
    await asyncio.gather(*list(scheduler._teardown_tasks))
    await asyncio.sleep(0)  # let the done-callbacks run

    assert good.teardown_calls == 1
    assert len(scheduler.client_list) == 0
    assert scheduler._teardown_tasks == set()
    assert _other_tasks() == []
