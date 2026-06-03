"""SimplyPrint WebSocket client — build a printer integration on top of this library.

The names below are the supported, author-facing public API. In short:

* Subclass :class:`PrinterClient` (the headline base) -- or the lower-level
  :class:`DefaultClient`/:class:`PhysicalClient` -- to boil your device into a
  :class:`PrinterState`, and mark demand/message handlers with :func:`configure`.
* Declare your client type with a :class:`ClientSpec`, configure the process via
  :class:`ClientSettings`, and run it through :class:`ClientApp`.

Deeper modules (``simplyprint_ws_client.core.*``, ``...shared.*``, ``...contrib.*``)
are importable but not part of the stable surface.

Performance: public names are re-exported **lazily** (PEP 562 ``__getattr__``), so
``import simplyprint_ws_client`` -- or importing a light submodule like
``simplyprint_ws_client.const`` -- does not eagerly build the whole client
(aiohttp/websockets, the protocol models, sentry). A heavy module loads only when
a name it owns is first used. ``_PUBLIC`` is the curated front door
``from simplyprint_ws_client import *`` exposes; any other public name still
resolves lazily for backward compatibility.
"""

from __future__ import annotations

import importlib as _importlib
from typing import TYPE_CHECKING

from . import _polyfill  # noqa: F401  (cheap; installs runtime polyfills)

# Modules whose public names are re-exported, tried light-first so resolving a
# config/settings/transport-base name never drags in core.app/aiohttp/sentry.
_REEXPORT_MODULES = (
    ".contrib.connection",
    ".core.config",
    ".core.settings",
    ".core.state",
    ".core.autowire",
    ".core.ws_protocol.connection",
    ".core.ws_protocol.models",
    ".core.ws_protocol.messages",
    ".contrib.printer_client",
    ".core.client",
    ".core.app",
)

# The curated, documented public surface. Other public names remain importable
# (resolved lazily by __getattr__), just not advertised by ``import *``.
_PUBLIC = (
    "PrinterClient",
    "Client",
    "DefaultClient",
    "PhysicalClient",
    "ClientState",
    "configure",
    "ClientApp",
    "ClientSettings",
    "ClientSpec",
    "ConnectionMode",
    "WebSocketTransport",
    "WebSocketsTransport",
    "AiohttpWebSocketTransport",
    "TransportError",
    "TransportClosed",
    "PrinterConfig",
    "Config",
    "ConfigManager",
    "ConfigManagerType",
    "PrinterState",
    "PrinterStatus",
    "ClientConfigChangedEvent",
    "ClientStateChangeEvent",
    "FileDemandData",
    "PluginInstallDemandData",
    "FileProgressState",
    "FileProgressStateEnum",
    "MaterialEntry",
    "MaterialLayoutEntry",
    "MultiMaterialSolution",
    "NozzleType",
    "NotificationEventSeverity",
    "ObjectsMsg",
)


def __getattr__(name: str):
    # ``import *`` reads __all__; serve the curated list lazily (no static __all__
    # constant, so the re-export hub stays free of "undefined name in __all__").
    if name == "__all__":
        return list(_PUBLIC)
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    for module_path in _REEXPORT_MODULES:
        module = _importlib.import_module(module_path, __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value  # cache: __getattr__ fires once per name
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC))


if TYPE_CHECKING:
    # Eager re-exports for static analysis / IDEs only (no runtime cost).
    from .contrib.printer_client import PrinterClient  # noqa: F401
    from .contrib.connection import (  # noqa: F401
        AiohttpWebSocketTransport,
        TransportClosed,
        TransportError,
        WebSocketsTransport,
        WebSocketTransport,
    )
    from .core.app import *  # noqa: F401,F403
    from .core.client import *  # noqa: F401,F403
    from .core.config import *  # noqa: F401,F403
    from .core.settings import *  # noqa: F401,F403
    from .core.state import *  # noqa: F401,F403
    from .core.ws_protocol.connection import ConnectionMode  # noqa: F401
    from .core.ws_protocol.messages import *  # noqa: F401,F403
    from .core.ws_protocol.models import (  # noqa: F401
        ClientMsgType,
        DemandMsgType,
        DispatchMode,
        ServerMsgType,
    )
