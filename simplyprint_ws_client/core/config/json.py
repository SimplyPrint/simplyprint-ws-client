__all__ = ["JsonConfigManager"]

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from simplyprint_ws_client.core.config.manager import ConfigManager
from simplyprint_ws_client.core.config import Config
from simplyprint_ws_client.core.files.file_backup import FileBackup


class JsonConfigManager(ConfigManager):
    _file_lock: threading.Lock

    def __init__(self, *args, **kwargs):
        self._file_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def flush(self, config: Optional[Config] = None):
        self._ensure_json_file()

        with self._file_lock:
            data = [
                json.loads(config.as_json())
                for config in self.get_all()
                if not config.is_empty()
            ]
            # Atomic: write to a temp file and replace, so a crash mid-write
            # can never destroy the printer registry.
            tmp = self._json_file.with_suffix(".json.tmp")
            with open(tmp, "w") as file:
                json.dump(data, file, indent=4)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, self._json_file)

    def load(self):
        self._ensure_json_file()

        with self._file_lock:
            try:
                with open(self._json_file, "r") as file:
                    data = json.load(file)
            except json.JSONDecodeError:
                # Never discard registration data: preserve the unreadable
                # file (the next flush writes a fresh one) and start empty.
                corrupt = self._json_file.with_suffix(".json.corrupt")
                logging.error(
                    "%s is not valid JSON; preserving it as %s and starting empty",
                    self._json_file,
                    corrupt,
                )
                try:
                    os.replace(self._json_file, corrupt)
                except OSError:
                    logging.warning(
                        "could not preserve the corrupt config file", exc_info=True
                    )
                data = []

            for config in data:
                self.persist(self.config_t.from_dict(config))

    def delete_storage(self):
        with self._file_lock:
            if not self._json_file.exists():
                return

            self._json_file.unlink()

    def backup_storage(self, *args, **kwargs):
        self._ensure_json_file()

        with self._file_lock:
            FileBackup.backup_file(self._json_file, *args, **kwargs)

    @property
    def _json_file(self) -> Path:
        return self.base_directory / f"{self.name}.json"

    def _ensure_json_file(self):
        with self._file_lock:
            if not self._json_file.exists():
                # Always create a valid JSON file, to prevent issues.
                with open(self._json_file, "w") as file:
                    json.dump([], file, indent=4)
