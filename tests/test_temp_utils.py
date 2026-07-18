import os
import tempfile
from pathlib import Path

import simplyprint_ws_client.const as ws_const
from simplyprint_ws_client.common.utils.temp import (
    cache_temporary_directory,
    configure_process_tempdir,
    prune_stale_temporary_entries,
)


def _patch_cache_path(monkeypatch, cache: Path) -> None:
    monkeypatch.setattr(
        type(ws_const.APP_DIRS),
        "user_cache_path",
        property(lambda self: cache),
    )


def test_cache_temporary_directory_defaults_to_app_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    _patch_cache_path(monkeypatch, cache)

    with cache_temporary_directory("unit-") as dirname:
        directory = Path(dirname)
        assert directory.parent == cache / "tmp"
        assert directory.exists()

    assert not directory.exists()


def test_configure_process_tempdir_moves_system_temp_to_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    _patch_cache_path(monkeypatch, cache)
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.delenv("TEMP", raising=False)
    monkeypatch.delenv("TMP", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)

    directory = configure_process_tempdir()

    assert directory == cache / "tmp"
    assert os.environ["TMPDIR"] == str(directory)
    assert os.environ["TEMP"] == str(directory)
    assert os.environ["TMP"] == str(directory)
    assert tempfile.gettempdir() == str(directory)


def test_prune_stale_temporary_entries_deletes_only_old_matching_entries(tmp_path):
    now = 1_000_000.0
    max_age = 60.0
    root = tmp_path / "transfers"
    root.mkdir()
    old_match = root / "sp-transfer-old"
    old_match.mkdir()
    recent_match = root / "sp-transfer-recent"
    recent_match.mkdir()
    old_other = root / "other-old"
    old_other.mkdir()

    os.utime(old_match, (now - 120.0, now - 120.0))
    os.utime(recent_match, (now - 10.0, now - 10.0))
    os.utime(old_other, (now - 120.0, now - 120.0))

    result = prune_stale_temporary_entries(
        root,
        older_than_seconds=max_age,
        prefixes=("sp-transfer-",),
        now=now,
    )

    assert result.scanned == 2
    assert result.deleted == 1
    assert result.errors == 0
    assert not old_match.exists()
    assert recent_match.exists()
    assert old_other.exists()


def test_prune_stale_temporary_entries_can_clean_owned_root_without_prefix(tmp_path):
    now = 1_000_000.0
    root = tmp_path / "tmp"
    root.mkdir()
    stale_file = root / "tmpabc"
    stale_file.write_text("stale")
    os.utime(stale_file, (now - 120.0, now - 120.0))

    result = prune_stale_temporary_entries(
        root,
        older_than_seconds=60.0,
        now=now,
    )

    assert result.deleted == 1
    assert not stale_file.exists()
