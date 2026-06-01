"""``PrinterClient`` is the authoring front door (the promoted ``BasePrinterClient``).

Two contracts are pinned here: it exposes the documented hook surface an
integration fills, and -- because it is shared library code -- it stays
brand-free (the no-brand-leak rule the integrations enforce in their own
``test_architecture.py``, applied one level up).
"""

import pathlib
import re

from simplyprint_ws_client import DefaultClient, PrinterClient
from simplyprint_ws_client.contrib import printer_client as printer_client_module
from simplyprint_ws_client.shared.camera.mixin import ClientCameraMixin

# Names that must never appear executably (or in docs) in shared library code.
_BRANDS = [
    "bambu",
    "anycubic",
    "creality",
    "prusa",
    "duet",
    "elegoo",
    "centauri",
    "ultimaker",
    "moonraker",
    "klipper",
    "octoprint",
]

_HOOKS = (
    # lifecycle template
    "init",
    "tick",
    "halt",
    "teardown",
    # connection-component lifecycle
    "_start_connection",
    "_stop_connection",
    "_connection_components",
    "_tick_progress",
    # connection event wiring
    "_connection_event_bus",
    "_connection_event_bindings",
    "_wire_connection_events",
    "on_connected_to_printer",
    "on_disconnected_from_printer",
    # status reduction pipeline
    "apply_status",
    "hold_status_on_cancel",
    "hold_status_on_pause",
    "hold_status_while_downloading",
    "is_job_start",
    "is_job_finish",
    "_on_job_start",
    "_on_job_finish",
    "_on_job_progress",
    # camera + telemetry
    "_init_camera",
    "_resolve_camera_uri",
    "update_camera_uri",
    "update_host_telemetry",
)


def test_printer_client_is_the_promoted_authoring_base():
    assert issubclass(PrinterClient, DefaultClient)
    assert issubclass(PrinterClient, ClientCameraMixin)


def test_printer_client_exposes_the_documented_hook_surface():
    missing = [hook for hook in _HOOKS if not hasattr(PrinterClient, hook)]
    assert not missing, f"PrinterClient is missing hooks: {missing}"


def test_contrib_package_is_brand_free():
    contrib_dir = pathlib.Path(printer_client_module.__file__).parent
    offenders = []
    for path in sorted(contrib_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for brand in _BRANDS:
            if re.search(rf"\b{brand}\b", text):
                offenders.append(f"{path.name}: {brand}")
    assert not offenders, f"brand leak in contrib/: {offenders}"
