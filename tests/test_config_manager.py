import pytest

from simplyprint_ws_client import (
    ConfigManager,
    PrinterConfig,
)
from simplyprint_ws_client.core.config import (
    ConfigManagerType,
    JsonConfigManager,
    MemoryConfigManager,
    SQLiteConfigManager,
    StoreConfigManager,
)
from simplyprint_ws_client.core.store import Store


def test_internal():
    config_manager = MemoryConfigManager()

    assert config_manager.name == "printers"
    assert len(config_manager) == 0

    config1 = PrinterConfig(id=1, token="token1")
    config2 = PrinterConfig(id=2, token="token2")
    config3 = PrinterConfig(id=3, token="token3")

    assert config1.partial_eq(config1)
    assert config2.partial_eq(config2)
    assert config3.partial_eq(config3)

    assert not config1.partial_eq(config2)
    assert not config2.partial_eq(config3)
    assert not config3.partial_eq(config1)

    config_manager.persist(config1)
    config_manager.persist(config2)
    config_manager.persist(config3)

    assert len(config_manager) == 3

    assert config_manager.by_id(1) == config1
    assert config_manager.by_id(2) == config2
    assert config_manager.by_id(3) == config3

    assert config_manager.by_token("token1") == config1
    assert config_manager.by_token("token2") == config2
    assert config_manager.by_token("token3") == config3

    assert config_manager.find(PrinterConfig(id=1, token="token1")) == config1
    assert config_manager.find(PrinterConfig(id=2, token="token2")) == config2
    assert config_manager.find(PrinterConfig(id=3, token="token3")) == config3
    assert config_manager.find(PrinterConfig(id=3, token="token4")) != config3


def _test_manager(config_manager: ConfigManager):
    config_manager.delete_storage()

    config = PrinterConfig.get_blank()
    config_manager.persist(config)

    assert len(config_manager) == 1
    assert config_manager.by_id(config.id) == config

    config_manager.flush()

    config_manager.clear()
    config_manager.load()

    assert len(config_manager) == 0

    config_manager.persist(config)
    config.id = 1337
    config.token = "Super cool token"
    config.unique_id = str(id(config))
    config.public_ip = "127.0.0.1"

    config_manager.flush(config)

    config_manager.clear()
    config_manager.load()

    assert len(config_manager) == 1

    config2 = config_manager.by_id(config.id)

    assert config2.id == config.id
    assert config2.token == config.token
    assert config2.unique_id == config.unique_id

    config_manager.remove(config2)

    assert len(config_manager) == 0
    config_manager.flush()
    config_manager.load()
    assert len(config_manager) == 0

    # Testing pending printer type scenario
    config = PrinterConfig.get_blank()
    config_manager.persist(config)
    config_manager.flush()

    config.id = 1336
    config.token = "Super cool token"

    config_manager.flush(config)
    config_manager.clear()
    config_manager.load()

    assert len(config_manager) == 1

    config_manager.delete_storage()


@pytest.fixture(params=["json", "sqlite", "store"])
def manager_kind(request):
    return request.param


@pytest.fixture
def isolated_manager(manager_kind, tmp_path):
    """Each manager type, in its own directory (the store in its own file)."""
    if manager_kind == "store":
        store = Store.open(tmp_path / "state.db", start=False)
        manager = StoreConfigManager(base_directory=str(tmp_path), store=store)
        yield manager
        store.close()
        return
    manager_t = JsonConfigManager if manager_kind == "json" else SQLiteConfigManager
    yield manager_t(base_directory=str(tmp_path))


def test_every_manager_round_trips(isolated_manager):
    _test_manager(isolated_manager)
    if isinstance(isolated_manager, JsonConfigManager):
        assert not isolated_manager._json_file.exists()
    elif isinstance(isolated_manager, SQLiteConfigManager):
        assert not isolated_manager._database_file.exists()
    else:
        assert isolated_manager.store.rows("printers") == []


def test_the_store_type_builds_a_store_manager(tmp_path):
    store = Store.open(tmp_path / "state.db", start=False)
    StoreConfigManager.bind(store)
    try:
        manager = ConfigManagerType.STORE(name="acme", config_t=PrinterConfig)
        assert isinstance(manager, StoreConfigManager)
        assert manager.store is store
    finally:
        StoreConfigManager.bind(None)
        store.close()


def test_json_manager():
    json_config_manager = JsonConfigManager()
    _test_manager(json_config_manager)
    assert not json_config_manager._json_file.exists()


def test_sqlite3_manager():
    sqlite_config_manager = SQLiteConfigManager()
    _test_manager(sqlite_config_manager)
    assert not sqlite_config_manager._database_file.exists()
