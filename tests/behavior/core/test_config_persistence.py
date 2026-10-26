"""Regression coverage for config persistence safety.

Each guarantee is checked on the JSON manager and on the store manager
(``ConfigManagerType.STORE``); the corrupt-file case differs in mechanism, so
each has its own test.
"""

import json as json_module
import shutil
import threading

import pytest

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.config.json import JsonConfigManager
from simplyprint_ws_client.core.config.store_manager import StoreConfigManager
from simplyprint_ws_client.core.store import Store


@pytest.fixture(params=["json", "store"])
def kind(request):
    return request.param


@pytest.fixture
def stores():
    opened = []
    yield opened
    for store in opened:
        store.close()


def _manager(kind, base, stores, store_dir=None):
    if kind == "json":
        return JsonConfigManager(
            name="printers", config_t=PrinterConfig, base_directory=base
        )
    # The client binds state.db outside the config directory; tests that
    # remove the config dir keep it apart the same way.
    path = (store_dir or base) / "state.db"
    store = next((s for s in stores if s.path == path), None)
    if store is None:
        store = Store.open(path, start=False)
        stores.append(store)
    return StoreConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=base, store=store
    )


def _stored_ids(kind, base, manager):
    if kind == "json":
        return [e["id"] for e in json_module.loads((base / "printers.json").read_text())]
    manager.store.writer.flush_now()
    return [row["pk"] for row in manager.store.rows("printers", store="printers")]


def test_corrupt_json_config_is_preserved_not_reset(tmp_path):
    manager = JsonConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=tmp_path
    )
    corrupt_source = '{"definitely": "not a list"'
    (tmp_path / "printers.json").write_text(corrupt_source)

    manager.load()

    # The unreadable file is preserved for recovery, never silently discarded.
    assert (tmp_path / "printers.json.corrupt").read_text() == corrupt_source
    assert manager.get_all() == []


def test_a_corrupt_store_is_preserved_not_reset(tmp_path):
    corrupt_source = b"definitely not a database " * 100
    (tmp_path / "state.db").write_bytes(corrupt_source)

    manager = StoreConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=tmp_path
    )
    try:
        assert manager.get_all() == []
        report = manager.store.recovery
        # The damaged file is moved aside byte for byte, never deleted.
        assert report is not None and report.moved_to.read_bytes() == corrupt_source
    finally:
        manager.store.close()


def test_config_flush_is_atomic(kind, tmp_path, stores):
    manager = _manager(kind, tmp_path, stores)
    config = PrinterConfig.get_new()
    config.id = 7
    manager.persist(config)
    manager.flush()

    assert _stored_ids(kind, tmp_path, manager) == [7]
    # No temp file left behind.
    assert not (tmp_path / "printers.json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_flushes_do_not_race(kind, tmp_path, stores):
    """Two managers writing the same registry must not collide.

    A fixed ``<name>.json.tmp`` let the first writer's os.replace consume the
    temp out from under the second -> FileNotFoundError. A unique temp per write
    removes that race; the store serializes writers on one connection. This
    fuzzes it across two managers and many rounds.
    """

    def make_manager(printer_id):
        manager = _manager(kind, tmp_path, stores)
        config = PrinterConfig.get_new()
        config.id = printer_id
        manager.persist(config)
        return manager

    managers = [make_manager(1), make_manager(2)]
    barrier = threading.Barrier(len(managers))
    errors = []

    def hammer(manager):
        try:
            barrier.wait()
            for _ in range(50):
                manager.flush()
        except Exception as exc:  # noqa: BLE001 - surface any race as a failure
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(m,)) for m in managers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    ids = _stored_ids(kind, tmp_path, managers[0])
    if kind == "store":
        # Rows are per printer: neither manager's flush drops the other's.
        assert sorted(ids) == [1, 2]
    else:
        # The registry is always valid JSON, one manager's whole list.
        assert ids in ([1], [2])
    assert list(tmp_path.glob("*.tmp")) == []


def test_flush_survives_a_removed_config_dir(kind, tmp_path, stores):
    """A config dir deleted at runtime is healed on the next flush."""
    base = tmp_path / "cfg"
    manager = _manager(kind, base, stores, store_dir=tmp_path / "data")
    config = PrinterConfig.get_new()
    config.id = 11
    manager.persist(config)

    shutil.rmtree(base)
    assert not base.exists()

    manager.flush()  # must not raise

    assert _stored_ids(kind, base, manager) == [11]
