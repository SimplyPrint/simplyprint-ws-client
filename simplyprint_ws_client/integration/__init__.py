"""The authoring kit: what a vendor subclasses to integrate a printer.

Lazy (PEP 562) like every layer ``__init__`` -- importing the package costs
nothing until a name is used.
"""

from __future__ import annotations

import importlib as _importlib

_PUBLIC = {
    "PrinterClient": ".client",
    "AppUpdater": ".client",
    "ConnectionEventBinding": ".client",
    "DeviceDriver": ".driver",
    "DeviceAuthError": ".driver",
    "DeviceLink": ".link",
    "WsDeviceLink": ".link",
    "MqttDeviceLink": ".link",
    "DevicePoller": ".poller",
}


def __getattr__(name: str):
    module_path = _PUBLIC.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC))
