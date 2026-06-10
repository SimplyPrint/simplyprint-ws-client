__all__ = [
    "ConfigManagerType",
    "ConfigManager",
    "MemoryConfigManager",
    "SQLiteConfigManager",
    "JsonConfigManager",
    "Config",
    "PrinterConfig",
]

from enum import Enum
from typing import Type

from simplyprint_ws_client.core.config.config import Config, PrinterConfig
from simplyprint_ws_client.core.config.json import JsonConfigManager
from simplyprint_ws_client.core.config.manager import ConfigManager
from simplyprint_ws_client.core.config.memory import MemoryConfigManager
from simplyprint_ws_client.core.config.sqlite import SQLiteConfigManager


class ConfigManagerType(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    JSON = "json"

    def __call__(self, *args, **kwargs):
        return self.get_class()(*args, **kwargs)

    def get_class(self) -> Type[ConfigManager]:
        if self == ConfigManagerType.MEMORY:
            return MemoryConfigManager
        elif self == ConfigManagerType.SQLITE:
            return SQLiteConfigManager
        elif self == ConfigManagerType.JSON:
            return JsonConfigManager
        else:
            raise ValueError("Invalid ConfigManagerType")
