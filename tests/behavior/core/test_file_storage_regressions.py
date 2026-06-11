"""Regression coverage for file-manager and backup behavior."""

import datetime
import os
import time

from simplyprint_ws_client.core.files.file_backup import FileBackup
from simplyprint_ws_client.core.files.file_manager import File, FileManager


def test_get_files_to_remove_with_duplicate_entries():
    fm = FileManager(max_age=10)
    now = int(time.time())
    dup = File("dup.gcode", 100, last_modified=now - 100)
    fresh = File("fresh.gcode", 50, last_modified=now)
    files = [dup, fresh, dup]

    removed = list(fm.get_files_to_remove(files, 10_000, 250))

    assert removed.count(dup) == 2
    assert files == [fresh]


def test_backup_file_removes_all_expired_backups(tmp_path):
    target = tmp_path / "config.json"
    target.write_text("current")

    old_mtime = time.time() - 3600
    for i in range(3):
        backup = tmp_path / f"config.json.bak.{i}"
        backup.write_text(f"old-{i}")
        os.utime(backup, (old_mtime, old_mtime))

    FileBackup.backup_file(target, max_age=datetime.timedelta(seconds=60))

    backups = sorted(p.name for p in tmp_path.glob("config.json.bak.*"))
    assert backups == ["config.json.bak.0"]
    assert (tmp_path / "config.json.bak.0").read_text() == "current"
