"""Neutral discovered-device record.

Active subnet scans have no brand "SSDP device" type to return, so they produce
this neutral record. Immutable identity has the first-class ``hardware_id``;
brand-specific presentation/onboarding facts gathered by a probe ride in
``extra`` so later config creation can read them without this type naming a
brand field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DiscoveredRecord:
    host: str
    serial: Optional[str] = None
    hardware_id: Optional[str] = None
    name: Optional[str] = None
    device_type: Optional[str] = None
    extra: dict = field(default_factory=dict)
