import datetime
import shutil
from pathlib import Path
from typing import Optional

from simplyprint_ws_client.common.utils.file_tail import strip_log_file


class FileBackup:
    """Small wrapper for count based file backups, used for configs"""

    @staticmethod
    def backup_file(
        file: Path,
        max_count: int = 5,
        min_age_interval: Optional[datetime.timedelta] = None,
        max_age: Optional[datetime.timedelta] = None,
    ):
        """Backup a file with a count based system

        :param file: The file to back up
        :param max_count: The maximum number of backups to keep
        :param min_age_interval: The minimum time between backups
        :param max_age: The maximum age of a backup

        Use the following format

        file.ext.bak.count

        """

        if not file.exists():
            return

        # Remove old backups by first sorting them by age and then removing the oldest ones
        # then adjust the count of the remaining ones
        backups = sorted(file.parent.glob(f"{file.name}.bak.*"), reverse=True)

        latest_backup: Optional[datetime.datetime] = None
        remaining = []

        for backup in backups:
            date_changed = datetime.datetime.fromtimestamp(backup.stat().st_mtime)

            # Keep track of the latest backup
            if not latest_backup or date_changed > latest_backup:
                latest_backup = date_changed

            if max_age and datetime.datetime.now() - date_changed > max_age:
                backup.unlink()
            else:
                remaining.append(backup)

        backups = remaining

        # If the last backup is too recent, don't create a new one and stop this function
        if (
            min_age_interval
            and latest_backup
            and datetime.datetime.now() - latest_backup < min_age_interval
        ):
            return

        for j, backup in enumerate(backups):
            i = len(backups) - j - 1

            if i + 1 >= max_count:
                backup.unlink()
            else:
                backup.rename(file.parent / f"{file.name}.bak.{i + 1}")

        # Now create the new backup by copying the original file
        shutil.copy(file, file.parent / f"{file.name}.bak.0")

    @staticmethod
    def strip_log_file(file: Path, max_size: int = 100 * 1024 * 1024):
        """Strip a log file to a maximum size (the generic primitive lives in
        ``common.utils.file_tail``; this stays the config-backup entry point)."""
        strip_log_file(file, max_size=max_size)
