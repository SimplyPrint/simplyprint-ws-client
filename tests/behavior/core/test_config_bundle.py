"""Round-trip + fail-safe coverage for the config export/import bundle."""

import io
import zipfile

import pytest

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.config.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleCorrupt,
    BundleError,
    BundleSource,
    BundleVersionError,
    ConfigBundle,
)
from simplyprint_ws_client.core.config.json import JsonConfigManager
from simplyprint_ws_client.core.files.file_backup import FileBackup


def _sources(tmp_path):
    a = tmp_path / "BambuClient.json"
    b = tmp_path / "settings.json"
    a.write_text('[{"id": 1}]')
    b.write_text('{"port": 8000}')
    return [
        BundleSource("BambuClient", "config_store", a),
        BundleSource("settings", "settings", b),
    ]


def test_export_restore_round_trips_byte_identical(tmp_path):
    sources = _sources(tmp_path)
    original = {s.path: s.path.read_bytes() for s in sources}

    bundle = ConfigBundle(sources, app_version="1.0", library_version="2.0")
    buffer = bundle.export_bytes()

    # Corrupt one file and delete another -- restore must put both back.
    sources[0].path.write_text("garbage")
    sources[1].path.unlink()

    report = bundle.restore_from(buffer)

    assert set(report.applied) == {"BambuClient", "settings"}
    assert report.skipped == []
    for path, data in original.items():
        assert path.read_bytes() == data
    # No temp file left behind by the atomic writes.
    assert list(tmp_path.glob("*.tmp")) == []


def test_export_uses_config_manager_storage_path(tmp_path):
    # The integration enumerates managers via the public storage_path property.
    manager = JsonConfigManager(
        name="BambuClient", config_t=PrinterConfig, base_directory=tmp_path
    )
    config = PrinterConfig.get_new()
    config.id = 42
    manager.persist(config)
    manager.flush()

    source = BundleSource(manager.name, "config_store", manager.storage_path)
    manifest = ConfigBundle([source]).export_to(io.BytesIO())

    assert manager.storage_path == tmp_path / "BambuClient.json"
    assert [e.logical_name for e in manifest.entries] == ["BambuClient"]


def test_inspect_rejects_a_tampered_member(tmp_path):
    sources = _sources(tmp_path)
    raw = ConfigBundle(sources).export_bytes().getvalue()

    # Rewrite one member so its bytes no longer match the manifest hash.
    tampered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name.endswith("BambuClient.json"):
                data = b'[{"id": 999}]'
            dst.writestr(name, data)
    tampered.seek(0)

    with pytest.raises(BundleCorrupt):
        ConfigBundle.inspect(tampered)


def test_inspect_rejects_a_newer_format_version(tmp_path):
    sources = _sources(tmp_path)
    raw = ConfigBundle(sources).export_bytes().getvalue()

    bumped = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(bumped, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "manifest.json":
                text = data.decode().replace(
                    f'"format_version": {BUNDLE_FORMAT_VERSION}',
                    f'"format_version": {BUNDLE_FORMAT_VERSION + 1}',
                )
                data = text.encode()
            dst.writestr(name, data)
    bumped.seek(0)

    with pytest.raises(BundleVersionError):
        ConfigBundle.inspect(bumped)


def test_inspect_rejects_a_non_bundle():
    with pytest.raises(BundleError):
        ConfigBundle.inspect(io.BytesIO(b"not a zip at all"))


def test_restore_validates_before_writing_anything(tmp_path):
    sources = _sources(tmp_path)
    raw = ConfigBundle(sources).export_bytes().getvalue()

    # Drop a declared member from the archive -> corrupt; restore must touch nothing.
    broken = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(broken, "w") as dst:
        for name in src.namelist():
            if name.endswith("BambuClient.json"):
                continue
            dst.writestr(name, src.read(name))
    broken.seek(0)

    sources[0].path.write_text("sentinel-A")
    sources[1].path.write_text("sentinel-B")

    with pytest.raises(BundleCorrupt):
        ConfigBundle(sources).restore_from(broken)

    # Nothing was overwritten.
    assert sources[0].path.read_text() == "sentinel-A"
    assert sources[1].path.read_text() == "sentinel-B"


def test_restore_skips_members_with_no_matching_source(tmp_path):
    sources = _sources(tmp_path)
    buffer = ConfigBundle(sources).export_bytes()

    # Restore onto a target set missing the accounts/settings store.
    only_one = [sources[0]]
    report = ConfigBundle(only_one).restore_from(buffer)

    assert report.applied == ["BambuClient"]
    assert report.skipped == ["settings"]


def test_file_backup_list_backups_indexes_newest_first(tmp_path):
    target = tmp_path / "BambuClient.json"
    target.write_text("v0")
    FileBackup.backup_file(target)  # -> .bak.0
    target.write_text("v1")
    FileBackup.backup_file(target, min_age_interval=None)  # rotate -> .bak.0,.bak.1

    backups = FileBackup.list_backups(target)
    assert [b.index for b in backups] == [0, 1]
    assert all(b.size > 0 for b in backups)
