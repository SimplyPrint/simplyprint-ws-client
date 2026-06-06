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
from typing import Awaitable, Callable, Generic, Optional, Tuple, TYPE_CHECKING, TypeVar

from simplyprint_ws_client.events import Event

from simplyprint_ws_client.contrib.discovery.ssdp import SSDPRequest

if TYPE_CHECKING:
    from simplyprint_ws_client.contrib.discovery.network import HostProbeContext
    from simplyprint_ws_client.contrib.discovery.mdns import MDNSResponse

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


@dataclass(frozen=True)
class MDNSSpec(Generic[DiscoveredRecord]):
    """How to discover one brand over mDNS / DNS-SD.

    The backend multicasts each name in ``queries`` as a PTR question every
    ``query_interval`` seconds. For every parsed response it calls ``follow_up``
    (to chain DNS-SD service-type enumeration -- the default issues no follow-up)
    and ``mapper`` (to extract a brand device record). All brand knowledge -- the
    query names, the follow-up rule, the record mapping, the cache key -- lives in
    these callables; the group/port default to the standard mDNS endpoint.
    """

    brand: str
    #: PTR query names to multicast periodically (e.g. a DNS-SD service type).
    queries: Tuple[str, ...]
    #: Event emitted for each mapped device.
    event_type: type[Event]
    #: Parsed mDNS response + sender address -> a brand device record, or None.
    mapper: Callable[["MDNSResponse", "tuple[str, int]"], Optional[DiscoveredRecord]]
    #: Stable cache key for a mapped record (e.g. its serial or host).
    key: Callable[[DiscoveredRecord], str]
    #: Given a response, the additional PTR query names to issue (DNS-SD stage 2).
    #: The default chains nothing -- single-stage brands omit it.
    follow_up: Callable[["MDNSResponse"], "Tuple[str, ...]"] = lambda response: ()
    group: str = "224.0.0.251"
    port: int = 5353
    query_interval: float = 30.0
    multicast_ttl: Optional[int] = None
