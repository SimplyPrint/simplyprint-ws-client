__all__ = ["MemoryConfigManager"]

from typing import Optional

from simplyprint_ws_client.core.config.manager import ConfigManager
from simplyprint_ws_client.cloud.config import Config


class MemoryConfigManager(ConfigManager):
    def flush(self, config: Optional[Config] = None): ...

    def load(self): ...

    def delete_storage(self): ...

    def backup_storage(self, *args, **kwargs): ...
