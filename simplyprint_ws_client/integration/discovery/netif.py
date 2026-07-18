"""Local network helpers for active subnet discovery.

Neutral utilities lifted from the per-brand subnet scanners so the shared
scanner has one home for "what hosts should I probe". No brand knowledge here.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import List

import psutil

#: Hard ceiling on how many hosts a single scan will probe. A /24 is 254 hosts;
#: a misconfigured /16 would be 65k. We cap (and log) rather than flood the LAN.
_MAX_SCAN_HOSTS = 1024


def local_subnets() -> List[ipaddress.IPv4Network]:
    """Every non-loopback IPv4 subnet this machine is attached to."""
    subnets: List[ipaddress.IPv4Network] = []

    for addrs in psutil.net_if_addrs().values():
        for addr in addrs:
            if addr.family != socket.AF_INET or addr.address == "127.0.0.1":
                continue
            try:
                subnets.append(
                    ipaddress.IPv4Network(
                        f"{addr.address}/{addr.netmask}", strict=False
                    )
                )
            except ValueError:
                continue

    return subnets


def scan_hosts(max_hosts: int = _MAX_SCAN_HOSTS) -> List[str]:
    """Deduplicated host addresses to probe, smallest subnets first, capped.

    Smallest-subnet-first means the likely /24 the printer is on gets covered
    before any larger range, and the cap keeps a stray large subnet from turning
    discovery into a full-network sweep.
    """
    seen: set[str] = set()
    hosts: List[str] = []

    for subnet in sorted(local_subnets(), key=lambda net: net.num_addresses):
        for ip in subnet.hosts():
            host = str(ip)
            if host in seen:
                continue
            seen.add(host)
            hosts.append(host)
            if len(hosts) >= max_hosts:
                return hosts

    return hosts
