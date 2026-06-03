"""Brand-neutral discovery specs.

A brand contributes a *spec* describing HOW to discover its printers; the shared
backends in this package execute the spec without knowing which brand it is. All
brand-specific knowledge (multicast group/port, search payloads, SSDP header
shapes, probe endpoints) lives inside the brand's ``mapper``/``probe``/``key``
callables, never in the shared backend. The only string a spec carries verbatim
is its ``brand`` routing label, supplied by the brand module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, TYPE_CHECKING, TypeVar

from simplyprint_ws_client.events import Event

from simplyprint_ws_client.contrib.discovery.ssdp import SSDPRequest

if TYPE_CHECKING:
    from simplyprint_ws_client.contrib.discovery.network import HostProbeContext

DiscoveredRecord = TypeVar("DiscoveredRecord")


@dataclass(frozen=True)
class NetworkServiceSpec:
    """A brand-declared network service the shared scanner may check.

    This is intentionally only transport/port metadata plus neutral purposes. The
    brand still owns any protocol fingerprint request and response interpretation.
    """

    id: str
    transport: str
    port: int
    label: Optional[str] = None
    purposes: tuple[str, ...] = ("discovery",)
    required: bool = True


@dataclass(frozen=True)
class MulticastSpec(Generic[DiscoveredRecord]):
    """How to discover one brand over UDP multicast (SSDP).

    ``search_payload``/``search_interval`` distinguish the two behaviours the
    backend supports: when absent the backend only *listens* (passive
    announcements); when present it additionally sends the payload to the group
    every ``search_interval`` seconds (active M-SEARCH).
    """

    brand: str
    group: str
    port: int
    #: Event emitted for each mapped device. The event class carries its own
    #: sync/async marker, so the backend stays brand-agnostic.
    event_type: type[Event]
    #: Parsed SSDP request + sender address -> a brand device record, or None.
    mapper: Callable[[SSDPRequest, "tuple[str, int]"], Optional[DiscoveredRecord]]
    #: Stable cache key for a mapped record (e.g. its serial).
    key: Callable[[DiscoveredRecord], str]
    multicast_ttl: Optional[int] = None
    search_payload: Optional[bytes] = None
    search_interval: Optional[float] = None

    @property
    def is_active(self) -> bool:
        """True when the backend should periodically send ``search_payload``."""
        return self.search_payload is not None


@dataclass(frozen=True)
class SubnetScanSpec(Generic[DiscoveredRecord]):
    """How to discover one brand by actively probing the local subnet(s).

    The shared backend enumerates hosts and bounds concurrency; the brand only
    supplies a ``probe`` coroutine that confirms (and identifies) one host. When
    ``port`` is set the backend first does a cheap TCP reachability check and only
    runs ``probe`` on hosts that answer, so the (expensive) brand handshake never
    touches the hundreds of dead addresses in a subnet.
    """

    brand: str
    #: Probe one reachable host -> a DiscoveredRecord, or None if not this brand.
    probe: Callable[[str], Awaitable[Optional[DiscoveredRecord]]]
    #: Stable cache key for a probed record (e.g. serial, falling back to host).
    key: Callable[[DiscoveredRecord], str]
    #: Optional cheap TCP port gate run before ``probe``.
    port: Optional[int] = None
    concurrency: int = 64
    gate_timeout: float = 0.3
    #: Declarative services used for discovery/onboarding/debug diagnostics. When
    #: absent, ``port`` is adapted into one required TCP service for compatibility.
    services: tuple[NetworkServiceSpec, ...] = ()
    #: Optional richer probe hook that receives cached port/service facts. Existing
    #: brands can keep the simple ``probe(host)`` hook until they need context.
    context_probe: Optional[
        Callable[["HostProbeContext"], Awaitable[Optional[DiscoveredRecord]]]
    ] = None
