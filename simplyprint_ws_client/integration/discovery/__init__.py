"""Reusable LAN-discovery subsystem.

Brand-agnostic machinery for finding printers on the local network: parsing SSDP
announcements, always-on multicast listening, on-demand active subnet scanning,
host port/service diagnostics, and a TTL-bounded results cache. Each integration
contributes a neutral *spec* (:class:`MulticastSpec` / :class:`MDNSSpec` /
:class:`SubnetScanSpec` / :class:`NetworkServiceSpec`) describing how its
devices announce or answer; the machinery here never names a vendor.

A brand whose devices speak something the shared listeners do not parse brings
its own always-on listener as a :class:`ListenerSpec`: a sans-I/O
:class:`ListenerProtocol` (events in, :class:`SendTo` / :class:`WakeAfter` /
:class:`Sighted` / :class:`Forget` out) that the host drives from a socket it
owns, under a scope (``DiscoveryService.add_listener``). A plain
"send this payload, parse each raw reply" exchange (a UDP broadcast, a
non-SSDP multicast) needs no protocol: a :class:`DatagramSpec` is swept on
demand (``DiscoveryService.sweep``) and, when ``always_on``, runs as such a
listener (:func:`datagram_listener`).

Imports are ordered by dependency (leaves first) so importing the package binds
each name before a later module needs it.
"""

from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice
from simplyprint_ws_client.integration.discovery.ssdp import (
    SSDPRequest,
    SSDPRequestParser,
)
from simplyprint_ws_client.integration.discovery.model import DiscoveredRecord
from simplyprint_ws_client.integration.discovery.reconcile import DeviceReconciler
from simplyprint_ws_client.integration.discovery.netif import local_subnets, scan_hosts
from simplyprint_ws_client.integration.discovery.spec import (
    DatagramSpec,
    ListenerSpec,
    MDNSSpec,
    MulticastSpec,
    NetworkServiceSpec,
    SubnetScanSpec,
)
from simplyprint_ws_client.integration.discovery.network import (
    DiagnosticCheckResult,
    DiagnosticReason,
    HostDiagnostic,
    HostProbeContext,
    NetworkScanContext,
    PortCheckResult,
    diagnostic_status,
    service_diagnostic_check,
    tcp_port_open,
)
from simplyprint_ws_client.integration.discovery.multicast import (
    MulticastDiscoveryBackend,
    scan_blocking,
)
from simplyprint_ws_client.integration.discovery.mdns import (
    MDNSDiscoveryBackend,
    MDNSRecord,
    MDNSResponse,
    MDNSResponseParser,
)
from simplyprint_ws_client.integration.discovery.subnet import SubnetScanBackend
from simplyprint_ws_client.integration.discovery.listener import (
    DatagramReceived,
    Forget,
    ListenerDiscoveryBackend,
    ListenerProtocol,
    ListenerSession,
    ListenerStarted,
    ListenerTick,
    SendTo,
    Sighted,
    WakeAfter,
)
from simplyprint_ws_client.integration.discovery.datagram import (
    DatagramCollector,
    datagram_listener,
)
from simplyprint_ws_client.integration.discovery.host import DiscoveryServiceHost
from simplyprint_ws_client.integration.discovery.service import DiscoveryService
from simplyprint_ws_client.integration.discovery.results import (
    DiscoveryResult,
    DiscoveryResultsStore,
)

__all__ = [
    "DiscoveredDevice",
    "DiscoveredRecord",
    "DeviceReconciler",
    "SSDPRequest",
    "SSDPRequestParser",
    "local_subnets",
    "scan_hosts",
    "MDNSSpec",
    "MDNSDiscoveryBackend",
    "MDNSRecord",
    "MDNSResponse",
    "MDNSResponseParser",
    "MulticastSpec",
    "NetworkServiceSpec",
    "SubnetScanSpec",
    "DiagnosticCheckResult",
    "DiagnosticReason",
    "HostDiagnostic",
    "HostProbeContext",
    "NetworkScanContext",
    "PortCheckResult",
    "diagnostic_status",
    "service_diagnostic_check",
    "tcp_port_open",
    "MulticastDiscoveryBackend",
    "scan_blocking",
    "SubnetScanBackend",
    "ListenerSpec",
    "ListenerProtocol",
    "ListenerSession",
    "ListenerDiscoveryBackend",
    "ListenerStarted",
    "ListenerTick",
    "DatagramReceived",
    "SendTo",
    "WakeAfter",
    "Sighted",
    "Forget",
    "DatagramSpec",
    "DatagramCollector",
    "datagram_listener",
    "DiscoveryServiceHost",
    "DiscoveryService",
    "DiscoveryResult",
    "DiscoveryResultsStore",
]
