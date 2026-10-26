"""LegacyImporter: today's config files into the store, once.

The fixtures in ``fixtures/legacy_config`` are shaped like a real install's
config directory: brand registries (one with a field from a newer build, one
entry without a ``unique_id``), a cloud account store, the web settings, an
upload record, and the broken kinds: a ``.json.corrupt`` the JSON manager set
aside, a registry that is not valid JSON, one that is not a list, stray
non-object entries, a plugin's registry nobody names, a ``.db`` that is not
SQLite. The old SQLite registry is made by the test with that manager's exact
DDL.
"""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from simplyprint_ws_client.core.config import PrinterConfig, StoreConfigManager
from simplyprint_ws_client.core.store import Store
from simplyprint_ws_client.core.store.legacy import (
    UPLOADS_DOMAIN,
    AccountFormat,
    LegacyFile,
    LegacyImporter,
    LegacyImportError,
    Rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_config"

PRINTER_STORES = {
    "BambuDevice-bambu",
    "BambuDevice-elegoo",
    "BambuDevice-duet",
    "BambuDevice-prusa",
    "BambuDevice-anycubic",
    "BambuDevice-creality",
}
ACCOUNTS = {"BambuClientCloudAccounts": AccountFormat("uid", "region")}
UPLOAD = "uploads/6f1d2c3b-4a59-4e8f-9a0b-1c2d3e4f5a6b.json"


@pytest.fixture
def config_dir(tmp_path):
    directory = tmp_path / "config"
    shutil.copytree(FIXTURES, directory)
    # An old SQLiteConfigManager registry, made with its DDL.
    conn = sqlite3.connect(str(directory / "BambuDevice-anycubic.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS printers ("
        " pk INTEGER, sk TEXT, data TEXT, PRIMARY KEY (pk, sk))"
    )
    conn.execute(
        "INSERT INTO printers (pk, sk, data) VALUES (?, ?, ?)",
        (
            4242,
            "anycubic-token",
            json.dumps({"id": 4242, "token": "anycubic-token", "name": "Kobra S1",
                        "host": "192.168.1.40"}),
        ),
    )  # fmt: skip
    conn.execute(
        "INSERT INTO printers (pk, sk, data) VALUES (?, ?, ?)",
        (4243, "t2", "{this is not json"),
    )
    conn.commit()
    conn.close()
    return directory


@pytest.fixture
def stores(tmp_path):
    state = Store.open(tmp_path / "data" / "state.db", start=False)
    cache = Store.open(tmp_path / "cache" / "cache.db", durability="cache", start=False)
    yield state, cache
    state.close()
    cache.close()


def _importer(config_dir, printer_stores=PRINTER_STORES):
    return LegacyImporter(config_dir, printer_stores=printer_stores, account_stores=ACCOUNTS)


def _file(config_dir, name):
    return next(f for f in _importer(config_dir).scan() if f.name == name)


# -- scan -------------------------------------------------------------------------


def test_scan_finds_every_legacy_file_and_not_the_store(config_dir):
    Store.open(config_dir / "state.db", start=False).close()
    names = [f.name for f in _importer(config_dir).scan()]
    assert names == [
        "AcmeUnloadedPlugin.json",
        "BambuClientCloudAccounts.json",
        "BambuDevice-anycubic.db",
        "BambuDevice-bambu.json",
        "BambuDevice-creality.json.corrupt",
        "BambuDevice-duet.json",
        "BambuDevice-elegoo.json",
        "BambuDevice-prusa.json",
        "garbage.db",
        "settings.json",
        "uploads/6f1d2c3b-4a59-4e8f-9a0b-1c2d3e4f5a6b.json",
        "uploads/broken.json",
    ]


def test_scan_reads_files_verbatim(config_dir):
    file = _file(config_dir, "BambuDevice-bambu.json")
    assert file.data == (config_dir / "BambuDevice-bambu.json").read_bytes()
    assert file == LegacyFile.of(file.name, file.data)


def test_scan_does_not_touch_the_files(config_dir):
    before = {p: p.read_bytes() for p in config_dir.rglob("*") if p.is_file()}
    _importer(config_dir).scan()
    after = {p: p.read_bytes() for p in config_dir.rglob("*") if p.is_file()}
    assert after == before


# -- convert ----------------------------------------------------------------------


def test_a_registry_converts_keeping_unknown_fields_and_minting_missing_ids(config_dir):
    importer = _importer(config_dir)
    file = _file(config_dir, "BambuDevice-bambu.json")
    rows = importer.convert(file)
    assert rows.classified and rows.errors == ()
    first, second = rows.state
    assert first.key == "6f1d2c3b-4a59-4e8f-9a0b-1c2d3e4f5a6b"
    assert first.row["data"]["field_from_a_newer_build"] == {"kept": True}
    assert (first.row["pk"], first.row["sk"]) == (31337, "f3b0c2a9e8d7465f9a1b2c3d4e5f6071")
    # Minted, written into the data, and the same on every conversion.
    assert second.row["data"]["unique_id"] == second.key
    assert importer.convert(file) == rows


def test_stray_entries_become_unreadable_rows(config_dir):
    rows = _importer(config_dir).convert(_file(config_dir, "BambuDevice-elegoo.json"))
    assert [p.row["state"] for p in rows.state] == ["ok", "unreadable", "unreadable"]
    assert [p.row["data"] for p in rows.state[1:]] == ["a stray string entry", 42]
    assert len({p.key for p in rows.state}) == 3


def test_a_registry_that_is_not_a_list_is_one_entry(config_dir):
    rows = _importer(config_dir).convert(_file(config_dir, "BambuDevice-duet.json"))
    (only,) = rows.state
    assert only.key == "11111111-2222-4333-8444-555555555555"


def test_a_registry_that_is_not_json_converts_to_nothing_and_says_so(config_dir):
    rows = _importer(config_dir).convert(_file(config_dir, "BambuDevice-prusa.json"))
    assert rows.state == () and not rows.classified
    assert "not valid JSON" in rows.errors[0]


def test_a_corrupt_file_is_only_kept(config_dir):
    assert _importer(config_dir).convert(
        _file(config_dir, "BambuDevice-creality.json.corrupt")
    ) == Rows()


def test_an_unknown_file_is_unclassified(config_dir):
    importer = _importer(config_dir)
    assert importer.convert(_file(config_dir, "AcmeUnloadedPlugin.json")) == Rows()
    garbage = _file(config_dir, "garbage.db")
    assert garbage.error and importer.convert(garbage) == Rows()


def test_an_old_sqlite_registry_converts(config_dir):
    rows = _importer(config_dir).convert(_file(config_dir, "BambuDevice-anycubic.db"))
    good, bad = rows.state
    assert good.row["data"]["name"] == "Kobra S1" and good.row["state"] == "ok"
    assert bad.row["state"] == "unreadable" and bad.row["data"] == "{this is not json"


def test_accounts_convert_by_their_kind_key(config_dir):
    rows = _importer(config_dir).convert(_file(config_dir, "BambuClientCloudAccounts.json"))
    (account,) = rows.state
    assert account.table == "accounts"
    assert account.key == ("BambuClientCloudAccounts", "1234567890")
    assert account.row["secondary"] == "eu"
    assert "has no 'uid'" in rows.errors[0]


def test_settings_become_one_row_per_field_with_the_secret_apart(config_dir):
    rows = _importer(config_dir).convert(_file(config_dir, "settings.json"))
    fields = json.loads((config_dir / "settings.json").read_text())
    by_key = {p.key: p.row["value"] for p in rows.state}
    assert len(by_key) == len(fields)
    assert by_key[("web", "port")] == 8484
    assert by_key[("host_secrets", "secret_key")] == fields["secret_key"]
    assert ("web", "secret_key") not in by_key


def test_unreadable_settings_fail_loudly(config_dir):
    (config_dir / "settings.json").write_text('{"use_auth": tru')
    with pytest.raises(LegacyImportError):
        _importer(config_dir).convert(_file(config_dir, "settings.json"))


def test_an_upload_record_goes_to_the_cache(config_dir):
    importer = _importer(config_dir)
    rows = importer.convert(_file(config_dir, UPLOAD))
    (put,) = rows.cache
    assert put.key == (UPLOADS_DOMAIN, "printer:6f1d2c3b-4a59-4e8f-9a0b-1c2d3e4f5a6b", "record")
    assert put.row["value"]["version"] == 1 and rows.state == ()
    broken = importer.convert(_file(config_dir, "uploads/broken.json"))
    assert broken.cache == () and broken.errors


# -- run ----------------------------------------------------------------------------


def test_run_imports_everything_once(config_dir, stores):
    state, cache = stores
    report = _importer(config_dir).run(state, cache)

    assert len(report.recorded) == 12
    assert len(state.legacy_files()) == 12
    stored = {(f["name"]): f["classified"] for f in state.legacy_files()}
    assert stored["BambuDevice-bambu.json"] and not stored["AcmeUnloadedPlugin.json"]
    assert state.legacy_bytes(
        "BambuDevice-creality.json.corrupt",
        _file(config_dir, "BambuDevice-creality.json.corrupt").sha256,
    ) == (config_dir / "BambuDevice-creality.json.corrupt").read_bytes()
    # bambu 2 + elegoo 3 + duet 1 + anycubic 2
    assert len(state.rows("printers")) == 8
    assert len(state.rows("accounts")) == 1
    assert state.get("settings", ("web", "port"))["value"] == 8484
    assert len(cache.rows("kv")) == 1
    assert "AcmeUnloadedPlugin.json" in report.unclassified

    revision = state.revision
    again = _importer(config_dir).run(state, cache)
    assert again.recorded == [] and again.rows == 0
    assert state.revision == revision


def test_unreadable_settings_import_nothing(config_dir, stores):
    state, cache = stores
    (config_dir / "settings.json").write_text("not json")
    with pytest.raises(LegacyImportError):
        _importer(config_dir).run(state, cache)
    assert state.legacy_files() == [] and state.rows("printers") == []


def test_a_file_an_old_build_changed_is_imported_again_without_losing_fields(
    config_dir, stores
):
    state, cache = stores
    _importer(config_dir).run(state, cache)

    # After an OTA rollback the old build rotates a token and, not knowing
    # the newer field, writes the file without it.
    path = config_dir / "BambuDevice-bambu.json"
    entries = json.loads(path.read_text())
    entries[0]["token"] = "rotated-by-the-old-build"
    del entries[0]["field_from_a_newer_build"]
    path.write_text(json.dumps(entries, indent=4))

    report = _importer(config_dir).run(state, cache)
    assert report.recorded == ["BambuDevice-bambu.json"]
    row = state.get("printers", "6f1d2c3b-4a59-4e8f-9a0b-1c2d3e4f5a6b")
    assert row["sk"] == "rotated-by-the-old-build"
    assert row["data"]["field_from_a_newer_build"] == {"kept": True}
    assert len(state.rows("printers")) == 8  # the minted id matched: no duplicate
    assert len([f for f in state.legacy_files() if f["name"] == path.name]) == 2


def test_a_re_import_over_an_upgraded_row_does_not_mix_config_versions(
    config_dir, stores
):
    # The row was upgraded to config version 2 (a migration renamed "name"
    # to "label"); an old build, after a rollback, rewrote the file at
    # version 1. The re-imported row must be version-1 data only, so the
    # upgrade runs once on what it was written for, never over fields it
    # already produced.
    state, cache = stores
    _importer(config_dir).run(state, cache)
    uid = "6f1d2c3b-4a59-4e8f-9a0b-1c2d3e4f5a6b"
    stored = state.get("printers", uid)
    upgraded = dict(stored["data"])
    upgraded["label"] = upgraded.pop("name", "x")
    with state.transaction() as txn:
        txn.put(
            "printers",
            uid,
            {**{c: stored[c] for c in ("store", "pk", "sk", "state", "error")},
             "data": upgraded, "schema": 2},
        )  # fmt: skip

    path = config_dir / "BambuDevice-bambu.json"
    entries = json.loads(path.read_text())
    entries[0]["name"] = "Renamed by the old build"
    path.write_text(json.dumps(entries, indent=4))
    _importer(config_dir).run(state, cache)

    row = state.get("printers", uid)
    assert row["schema"] == 1
    assert row["data"]["name"] == "Renamed by the old build"
    assert "label" not in row["data"]


def test_a_file_is_converted_when_its_plugin_mounts(config_dir, stores):
    state, cache = stores
    _importer(config_dir).run(state, cache)
    assert state.rows("printers", store="AcmeUnloadedPlugin") == []

    later = _importer(config_dir, printer_stores=PRINTER_STORES | {"AcmeUnloadedPlugin"})
    report = later.import_unclassified(state, cache)
    assert report.converted == ["AcmeUnloadedPlugin.json"]
    (row,) = state.rows("printers", store="AcmeUnloadedPlugin")
    assert row["data"]["acme_host"] == "192.168.1.99"
    assert later.import_unclassified(state, cache).converted == []


def test_imported_rows_load_in_a_store_manager(config_dir, stores, tmp_path):
    state, cache = stores
    _importer(config_dir).run(state, cache)
    manager = StoreConfigManager(
        name="BambuDevice-bambu",
        config_t=PrinterConfig,
        base_directory=str(tmp_path / "config"),
        store=state,
    )
    configs = sorted(manager.get_all(), key=lambda c: c.id)
    assert [c.name for c in configs] == ["X1C Workshop", "P1S Garage"]
    configs[0].name = "Renamed"
    manager.flush(configs[0])
    data = state.get("printers", configs[0].unique_id)["data"]
    # Brand fields the base config does not declare survive the rewrite.
    assert data["serial"] == "00M09A350100123"
    assert data["field_from_a_newer_build"] == {"kept": True}
    assert data["name"] == "Renamed"
