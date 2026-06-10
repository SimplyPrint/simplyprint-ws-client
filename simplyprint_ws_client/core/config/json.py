__all__ = ["JsonConfigManager"]

import json
import logging
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
            with open(self._json_file, "w") as file:
                data = [
                    json.loads(config.as_json())
                    for config in self.get_all()
                    if not config.is_empty()
                ]
                json.dump(data, file, indent=4)

    def load(self):
        self._ensure_json_file()

        with self._file_lock:
            with open(self._json_file, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    logging.warning(
                        f"Failed to load {self._json_file} configuration file, it's invalid - resetting it!!!"
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
