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

from simplyprint_ws_client.integration.discovery.ssdp import SSDPRequest

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.discovery.listener import ListenerProtocol
    from simplyprint_ws_client.integration.discovery.network import HostProbeContext
    from simplyprint_ws_client.integration.discovery.mdns import MDNSResponse

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
    supplies a ``probe`` coroutine that confirms (and identifies) one host.
    Required ``services`` are checked before the probe, so an expensive brand
    handshake never touches hosts whose declared endpoints are closed.
    """

    brand: str
    #: Probe one reachable host -> a DiscoveredRecord, or None if not this brand.
    probe: Callable[[str], Awaitable[Optional[DiscoveredRecord]]]
    #: Stable cache key for a probed record (e.g. serial, falling back to host).
    key: Callable[[DiscoveredRecord], str]
    concurrency: int = 64
    gate_timeout: float = 0.3
    #: Declarative services used for discovery/onboarding/debug diagnostics.
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


@dataclass(frozen=True)
class DatagramSpec(Generic[DiscoveredRecord]):
    """How to discover one brand with a raw (non-SSDP) UDP datagram.

    Send ``payload`` to ``address`` (a broadcast address, a multicast group or
    one host) on each of ``ports``, collect the replies, and hand each raw
    reply and its source address to ``parse``. All brand knowledge (payload,
    reply format) lives in ``payload`` and ``parse``.

    On demand, ``DiscoveryService.sweep(brand, timeout)`` sends once and
    collects for ``timeout`` seconds. :attr:`always_on` specs also run as an
    always-on listener keyed ``brand`` (re-sending every ``interval``; a
    ``payload`` of ``None`` only listens, on ``listen_port`` and ``group``),
    whose cache ``snapshot(brand)`` reads.
    """

    brand: str
    #: Destination port(s); the payload goes to each in turn.
    ports: Tuple[int, ...]
    #: Raw reply + sender address -> a brand device record, or None.
    parse: Callable[[bytes, "tuple[str, int]"], Optional[DiscoveredRecord]]
    #: Stable cache key for a parsed record.
    key: Callable[[DiscoveredRecord], str]
    payload: Optional[bytes] = None
    address: str = "255.255.255.255"
    #: Re-send cadence of the always-on listener; ``None``: on demand only.
    interval: Optional[float] = None
    #: Port the replies arrive on; ``0`` binds an ephemeral one.
    listen_port: int = 0
    #: Multicast group to join while listening, if any.
    group: Optional[str] = None
    multicast_ttl: Optional[int] = None
    #: Event emitted for each always-on sighting; ``None`` only caches.
    event_type: Optional[type[Event]] = None

    @property
    def always_on(self) -> bool:
        """True when this spec also runs as an always-on listener."""
        return self.interval is not None or self.payload is None


@dataclass(frozen=True)
class ListenerSpec(Generic[DiscoveredRecord]):
    """A brand-owned always-on datagram listener, hosted sans-I/O.

    The host binds one UDP socket as described here, on the discovery host's
    shared loop, and drives a fresh ``protocol()`` (a
    :class:`~simplyprint_ws_client.integration.discovery.listener.ListenerProtocol`)
    with its datagrams and timers. ``brand`` is the listener's key: unique
    among every always-on listener (multicast, mDNS and these), so a brand
    that also has an SSDP or mDNS listener names this one apart (for example
    ``"<id>.broadcast"``).
    """

    brand: str
    #: A fresh protocol per run (a relaunched listener starts clean).
    protocol: Callable[[], "ListenerProtocol"]
    #: Stable cache key for a sighted record.
    key: Callable[[DiscoveredRecord], str]
    #: Port to bind; ``0`` binds an ephemeral one (hears replies only).
    port: int = 0
    #: Multicast group to join, if any.
    group: Optional[str] = None
    #: Allow sends to a broadcast address.
    broadcast: bool = False
    multicast_ttl: Optional[int] = None
    #: Event emitted for each sighting; ``None`` only caches.
    event_type: Optional[type[Event]] = None
