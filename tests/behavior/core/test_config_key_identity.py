from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.core.config import SQLiteConfigManager


def test_printer_storage_key_is_its_id_and_token():
    config = PrinterConfig(id=42, token="printer-token")

    assert config.pk == config.id == 42
    assert config.sk == config.token == "printer-token"
    assert config.key == (42, "printer-token")


def test_sqlite_persists_and_deletes_by_explicit_config_key(tmp_path):
    manager = SQLiteConfigManager(base_directory=str(tmp_path))
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
