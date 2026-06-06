"""Reusable LAN-discovery subsystem.

Brand-agnostic machinery for finding printers on the local network: parsing SSDP
announcements, always-on multicast listening, on-demand active subnet scanning,
host port/service diagnostics, and a TTL-bounded results cache. Each integration
contributes a neutral *spec* (:class:`MulticastSpec` / :class:`SubnetScanSpec` /
:class:`NetworkServiceSpec`) describing how its devices announce or answer; the
machinery here never names a vendor.

Imports are ordered by dependency (leaves first) so importing the package binds
each name before a later module needs it.
"""

from simplyprint_ws_client.contrib.discovery.device import DiscoveredDevice
from simplyprint_ws_client.contrib.discovery.ssdp import SSDPRequest, SSDPRequestParser
from simplyprint_ws_client.contrib.discovery.model import DiscoveredRecord
from simplyprint_ws_client.contrib.discovery.netif import local_subnets, scan_hosts
from simplyprint_ws_client.contrib.discovery.spec import (
    MDNSSpec,
    MulticastSpec,
    NetworkServiceSpec,
    SubnetScanSpec,
)
from simplyprint_ws_client.contrib.discovery.network import (
    DiagnosticCheckResult,
    HostDiagnostic,
    HostProbeContext,
    NetworkScanContext,
    PortCheckResult,
    diagnostic_status,
    service_diagnostic_check,
    tcp_port_open,
)
from simplyprint_ws_client.contrib.discovery.multicast import (
    MulticastDiscoveryBackend,
    scan_blocking,
)
from simplyprint_ws_client.contrib.discovery.mdns import (
    MDNSDiscoveryBackend,
    MDNSRecord,
    MDNSResponse,
    MDNSResponseParser,
)
from simplyprint_ws_client.contrib.discovery.subnet import SubnetScanBackend
from simplyprint_ws_client.contrib.discovery.host import DiscoveryServiceHost
from simplyprint_ws_client.contrib.discovery.service import DiscoveryService
from simplyprint_ws_client.contrib.discovery.results import (
    DiscoveryResult,
    DiscoveryResultsStore,
)

__all__ = [
    "DiscoveredDevice",
    "DiscoveredRecord",
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
    "DiscoveryServiceHost",
    "DiscoveryService",
    "DiscoveryResult",
    "DiscoveryResultsStore",
]
