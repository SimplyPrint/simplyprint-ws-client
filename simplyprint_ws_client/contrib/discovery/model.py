"""Neutral discovered-device record.

Active subnet scans have no brand "SSDP device" type to return, so they produce
this neutral record. Brand-specific facts gathered by a probe ride in ``extra``
(e.g. a board unique id + a webcam uri) so later config creation can read them
back without this type ever naming a brand field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DiscoveredRecord:
    host: str
    serial: Optional[str] = None
    name: Optional[str] = None
    device_type: Optional[str] = None
    extra: dict = field(default_factory=dict)
