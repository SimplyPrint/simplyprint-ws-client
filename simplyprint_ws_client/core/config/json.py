__all__ = ["JsonConfigManager"]

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from simplyprint_ws_client.core.config.manager import ConfigManager
from simplyprint_ws_client.core.config import Config
from simplyprint_ws_client.core.files.atomic import atomic_write_text
from simplyprint_ws_client.core.files.file_backup import FileBackup


class JsonConfigManager(ConfigManager):
    _file_lock: threading.Lock
    #: The last dict each config serialized to; written in its place when a
    #: config later fails to serialize (see flush).
    _last_good: Dict[Config, Any]

    def __init__(self, *args, **kwargs):
        self._file_lock = threading.Lock()
        self._last_good = {}
        super().__init__(*args, **kwargs)

    def flush(self, config: Optional[Config] = None):
        self._ensure_json_file()

        with self._file_lock:
            data = []
            last_good: Dict[Config, Any] = {}
            for config in self.get_all():
                # One file holds every printer, so one config that cannot
                # serialize must not stop the rest from being saved. It keeps
                # its last good form: what it was loaded from, or last saved as.
                # (A config added at runtime that never serialized was never on
                # disk, so leaving it out loses nothing stored.)
                try:
                    if config.is_empty():
                        continue
                    entry = json.loads(config.as_json())
                except Exception:
                    logging.exception(
                        "%s: could not serialize printer config %s; "
                        "keeping its last saved form",
                        self.name,
                        getattr(config, "unique_id", None),
                    )
                    entry = self._last_good.get(config)
                    if entry is None:
                        continue
                last_good[config] = entry
                data.append(entry)
            self._last_good = last_good
            # Write unreadable entries back verbatim: rewriting the file from
            # parsed configs alone would silently delete them.
            data.extend(entry.raw for entry in self.unreadable)
            # Atomic + collision-proof: a unique temp file in the same dir is
            # replaced onto the target, so a crash mid-write can never destroy
            # the printer registry and two writers can never race on a shared
            # ``<name>.json.tmp`` (which let the first os.replace consume the
            # temp out from under the second -> FileNotFoundError).
            atomic_write_text(self._json_file, json.dumps(data, indent=4))

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

            if not isinstance(data, list):
                # Valid JSON, but not the registry's list. Treat the whole
                # value as one unreadable entry: it is kept (inside the list
                # flush writes) instead of crashing the load.
                data = [data]

            self.unreadable = []
            for raw in data:
                config = self._load_entry(raw)
                if config is not None:
                    # Its on-disk form is its first "last good" one.
                    self._last_good[config] = raw

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
    def storage_path(self) -> Path:
        return self._json_file

    @property
    def _json_file(self) -> Path:
        return self.base_directory / f"{self.name}.json"

    def _ensure_json_file(self):
        with self._file_lock:
            # Heal a config dir that was removed at runtime, not just at __init__.
            self.base_directory.mkdir(parents=True, exist_ok=True)
            if not self._json_file.exists():
                # Always create a valid JSON file, to prevent issues.
                atomic_write_text(self._json_file, json.dumps([], indent=4))
