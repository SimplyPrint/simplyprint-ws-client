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

    Only the network-neutral facts every integration can supply: a reachable
    ``host`` (IP or hostname) plus an optional human ``name`` and ``serial``.
    Integration-specific discovery payload (model codes, signed tokens, SSDP
    headers, a ``mac``) rides in ``extra`` so callers can read it back without this
    neutral type ever naming vendor fields. The integration decides how to turn
    these facts into a config's hardware-match id (serial, else MAC).
    """

    host: str
    name: Optional[str] = None
    serial: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)
