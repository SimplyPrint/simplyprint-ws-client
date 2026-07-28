import importlib.metadata
from pathlib import Path

from platformdirs import AppDirs as PlatformAppDirs


class ApplicationDirectories:
    def __init__(self, directories: PlatformAppDirs) -> None:
        self._user_config_path = directories.user_config_path
        self._user_data_path = directories.user_data_path
        self._user_cache_path = directories.user_cache_path
        self._user_log_path = directories.user_log_path

    @property
    def user_config_path(self) -> Path:
        return self._user_config_path

    @property
    def user_config_dir(self) -> str:
        return str(self._user_config_path)

    @property
    def user_data_path(self) -> Path:
        return self._user_data_path

    @property
    def user_data_dir(self) -> str:
        return str(self._user_data_path)

    @property
    def user_cache_path(self) -> Path:
        return self._user_cache_path

    @property
    def user_cache_dir(self) -> str:
        return str(self._user_cache_path)

    @property
    def user_log_path(self) -> Path:
        return self._user_log_path

    @property
    def user_log_dir(self) -> str:
        return str(self._user_log_path)

VERSION = importlib.metadata.version("simplyprint_ws_client") or "development"
APP_DIRS = ApplicationDirectories(PlatformAppDirs("SimplyPrint", "SimplyPrint"))
