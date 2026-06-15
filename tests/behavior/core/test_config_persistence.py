"""Regression coverage for config persistence safety."""

import json as json_module
import shutil
import threading

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.config.json import JsonConfigManager


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


def test_json_config_flush_is_atomic(tmp_path):
    manager = JsonConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=tmp_path
    )
    config = PrinterConfig.get_new()
    config.id = 7
    manager.persist(config)
    manager.flush()

    data = json_module.loads((tmp_path / "printers.json").read_text())
    assert [entry["id"] for entry in data] == [7]
    # No temp file left behind.
    assert not (tmp_path / "printers.json.tmp").exists()


def test_concurrent_flushes_do_not_race_on_a_shared_temp(tmp_path):
    """Two managers writing the same registry must not collide.

    A fixed ``<name>.json.tmp`` let the first writer's os.replace consume the
    temp out from under the second -> FileNotFoundError. A unique temp per write
    removes that race; this fuzzes it across two managers and many rounds.
    """

    def make_manager(printer_id):
        manager = JsonConfigManager(
            name="printers", config_t=PrinterConfig, base_directory=tmp_path
        )
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
    # The registry is always valid JSON and no temp files are left behind.
    data = json_module.loads((tmp_path / "printers.json").read_text())
    assert isinstance(data, list)
    assert list(tmp_path.glob("*.tmp")) == []


def test_flush_recreates_a_removed_config_dir(tmp_path):
    """A config dir deleted at runtime is healed on the next flush."""
    base = tmp_path / "cfg"
    manager = JsonConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=base
    )
    config = PrinterConfig.get_new()
    config.id = 11
    manager.persist(config)

    shutil.rmtree(base)
    assert not base.exists()

    manager.flush()  # must not raise

    data = json_module.loads((base / "printers.json").read_text())
    assert [entry["id"] for entry in data] == [11]
