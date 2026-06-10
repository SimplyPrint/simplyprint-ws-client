"""The process's single live discovery service -- a narrow domain accessor.

There is exactly one passive listener set per process, so the subsystem owns its
active instance here and hands it to spec/onboarding code via
:func:`active_discovery_service` -- a narrow domain accessor, not a general
service-locator. The host (or an app's ``main()``) sets it at startup; tests set
a fake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from simplyprint_ws_client.device.discovery.service import DiscoveryService

_active_service: Optional["DiscoveryService"] = None


def set_active_discovery_service(service: Optional["DiscoveryService"]) -> None:
    """Register (or clear) the process's live discovery service."""
    global _active_service
    _active_service = service


def active_discovery_service() -> "DiscoveryService":
    """The live discovery service, or raise if none has been started."""
    if _active_service is None:
        raise RuntimeError("No discovery service is running")

    return _active_service
