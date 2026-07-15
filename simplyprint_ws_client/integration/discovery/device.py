"""Brand-neutral discovery result shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

#: A JSON-shaped value carried in discovery payloads and diagnostic details.
#: Spelled with typing.Union (not the ``X | Y`` operator) because this is a
#: runtime expression and the library floor is Python 3.9.
JsonValue = Union[
    str, int, float, bool, None, Dict[str, "JsonValue"], List["JsonValue"]
]


@dataclass(frozen=True)
class DiscoveredDevice:
    """A device a discovery source turned up on the network.

    ``hardware_id`` is the neutral immutable identity when discovery knows one
    (for example a GUID or MAC); ``serial`` remains a common fallback. ``extra``
    carries presentation/onboarding facts such as model ids and LAN-mode flags,
    never identity data.
    """

    host: str
    name: Optional[str] = None
    serial: Optional[str] = None
    hardware_id: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def hardware_identity(self) -> Optional[str]:
        """The immutable device id used for matching, when discovery knows it."""
        return self.hardware_id or self.serial

    def network_addresses(self) -> tuple[str, ...]:
        """Reachable addresses advertised by this discovery result."""
        return (self.host,) if self.host else ()

    def primary_network_address(self) -> Optional[str]:
        return self.host or None
