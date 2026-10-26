import pytest

from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.core.config import SQLiteConfigManager, StoreConfigManager
from simplyprint_ws_client.core.store import Store


def test_printer_storage_key_is_its_id_and_token():
    config = PrinterConfig(id=42, token="printer-token")

    assert config.pk == config.id == 42
    assert config.sk == config.token == "printer-token"
    assert config.key == (42, "printer-token")


@pytest.fixture(params=["sqlite", "store"])
def manager(request, tmp_path):
    if request.param == "sqlite":
        yield SQLiteConfigManager(base_directory=str(tmp_path))
        return
    store = Store.open(tmp_path / "state.db", start=False)
    yield StoreConfigManager(base_directory=str(tmp_path), store=store)
    store.close()


def test_persists_and_deletes_by_explicit_config_key(manager):
    config = PrinterConfig(id=42, token="printer-token", name="Workshop")

    manager.persist(config)
    assert manager.by_key(42, "printer-token") is config
    manager.flush()

    manager.clear()
    manager.load()
    restored = manager.by_key(42, "printer-token")
    assert restored is not None
    assert restored.name == "Workshop"

    manager.remove(restored)
    manager.flush()
    manager.clear()
    manager.load()
    assert manager.get_all() == []

    manager.delete_storage()


def test_the_store_keys_a_printer_by_unique_id_across_a_token_change(tmp_path):
    store = Store.open(tmp_path / "state.db", start=False)
    try:
        manager = StoreConfigManager(base_directory=str(tmp_path), store=store)
        config = PrinterConfig(id=0, token="pending", unique_id="slot-1", name="A")
        manager.persist(config)
        manager.flush(config)

        # Keyed: id and token change, the slot (and its row) stays.
        config.id = 42
        config.token = "minted"
        manager.flush()
        store.writer.flush_now()

        rows = store.rows("printers")
        assert [(r["unique_id"], r["pk"], r["sk"]) for r in rows] == [
            ("slot-1", 42, "minted")
        ]
    finally:
        store.close()
