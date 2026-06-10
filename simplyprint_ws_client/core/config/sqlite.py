__all__ = ["SQLiteConfigManager"]

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from simplyprint_ws_client.cloud.config import Config
from simplyprint_ws_client.core.config.manager import ConfigManager
from simplyprint_ws_client.cloud.files.file_backup import FileBackup


class SQLiteConfigManager(ConfigManager):
    """
    Handles configuration state and persistence.
    """

    logger: logging.Logger = logging.getLogger("config.sqlite3")

    db: Optional[sqlite3.Connection] = None
    __table_exists: bool = False

    def flush(self, config: Optional[Config] = None):
        self._ensure_database()

        try:
            if config is not None:
                self._flush_single(config)
                self._remove_detached()
                return

            for config in self.get_all():
                # Do not flush blank configs
                if config.is_empty():
                    continue

                self._flush_single(config)

            self._remove_detached()
        finally:
            self.db.commit()

    def load(self):
        self._ensure_database()

        configs = self.db.execute(
            """
            SELECT pk, sk, data FROM printers
            """
        ).fetchall()

        for config in configs:
            kwargs = json.loads(config[2])
            self.persist(self.config_t.from_dict(kwargs))

    def delete_storage(self):
        if not self._database_file.exists():
            return

        self._database_file.unlink()
        self.db = None
        self.__table_exists = False

    def backup_storage(self, *args, **kwargs):
        self._ensure_database()
        FileBackup.backup_file(self._database_file, *args, **kwargs)

    def _get_single(self, config: Config):
        return self.db.execute(
            """
            SELECT pk, sk FROM printers WHERE pk= ? AND sk= ? LIMIT 1
            """,
            (config.pk, config.sk),
        ).fetchone()

    def _flush_single(self, config: Config):
        # Check if unique sk and pk exists
        already_exists = self._get_single(config)

        if not already_exists:
            self.db.execute(
                """
                INSERT INTO printers (pk, sk, data) VALUES (?, ?, ?)
                """,
                (config.pk, config.sk, config.as_json()),
            )

            self.db.commit()
            self.logger.info(f"Inserted config {config}")
            return

        self.db.execute(
            """
            UPDATE printers SET data= ? WHERE pk= ? AND sk= ?
            """,
            (config.as_json(), config.pk, config.sk),
        )

    def _remove_detached(self):
        # Get all configs from the database
        configs = self.db.execute(
            """
            SELECT pk, sk FROM printers
            """
        ).fetchall()

        # Loop over all configs
        for config in configs:
            # If the config is not in the manager
            if not self.find(pk=config[0], sk=config[1]):
                # Remove it from the database
                self.db.execute(
                    """
                    DELETE FROM printers WHERE pk= ? AND sk= ?
                    """,
                    (config[0], config[1]),
                )

    @property
    def _database_file(self) -> Path:
        return self.base_directory / f"{self.name}.db"

    def _ensure_database(self):
        """
        Ensure the config table exists.
        """

        if not self.db or not self._database_file.exists():
            self.db = sqlite3.connect(self._database_file, check_same_thread=False)

        if self.__table_exists:
            return

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS printers (
                pk INTEGER, 
                sk TEXT, 
                data TEXT, 
                PRIMARY KEY (pk, sk)
            );
            """
        )

        self.db.commit()
        self.logger.info("Created printers table")
        self.__table_exists = True
