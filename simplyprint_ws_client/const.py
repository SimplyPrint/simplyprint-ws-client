import importlib.metadata

from platformdirs import AppDirs

VERSION = importlib.metadata.version("simplyprint_ws_client") or "development"
APP_DIRS = AppDirs("SimplyPrint", "SimplyPrint")
