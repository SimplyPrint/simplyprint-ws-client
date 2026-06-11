"""Regression coverage for config persistence safety."""

import json as json_module

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
