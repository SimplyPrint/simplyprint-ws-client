"""Resolve the MAC address of a host on the local network, brand-neutrally.

A discovered device without a serial still has a stable hardware identity: its
network adapter's MAC. We read it from the OS neighbour/ARP table (no root, no
extra dependency), optionally warming that table first with a best-effort TCP
connect so a freshly-seen host has an entry. This is the generic fallback behind
:meth:`DiscoveredDevice.stable_id` -- it names no brand and works for any device
the host has talked to on the LAN.

The MAC only resolves for hosts on the same layer-2 segment (the ARP/neighbour
table never holds a routed host's hardware address); off-segment hosts return
``None`` and the caller falls back to a random id, which is correct.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from typing import Optional

from simplyprint_ws_client.common.process import run as run_command
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict

#: host -> resolved MAC / unresolved marker. Passive re-announcements can arrive
#: every few seconds per printer, so cache both hits and misses to avoid repeatedly
#: shelling out to neighbour-table tools on the caller's loop.
_MAC_CACHE_TTL = 15 * 60
_MAC_NEGATIVE_CACHE_TTL = 5 * 60
_mac_cache: ExpiringDict = ExpiringDict(ttl=_MAC_CACHE_TTL)
_mac_negative_cache: ExpiringDict = ExpiringDict(ttl=_MAC_NEGATIVE_CACHE_TTL)

# A normalised, fully-specified unicast MAC (six octets, not the all-zero /
# broadcast placeholders the neighbour table parks against unresolved entries).
_MAC_RE = re.compile(
    r"(?<![0-9a-f])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f])"
    r"|(?<![0-9a-f])(?:[0-9a-f]{2}-){5}[0-9a-f]{2}(?![0-9a-f])"
)
_EMPTY_MACS = {"00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff"}

# Ports worth a quick knock to populate the neighbour table when it is cold. The
# connect attempt itself triggers ARP regardless of whether the port answers, so
# the list is just "things a LAN device might have open" -- a refused connection
# warms the table just as well as an accepted one.
_WARM_PORTS = (80, 443, 9100)


def _resolve_host_ip(host: str) -> Optional[str]:
    """Best-effort host -> IPv4 string (the neighbour table is keyed by IP)."""
    try:
        socket.inet_aton(host)
        return host
    except OSError:
        pass
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def _warm_neighbour_table(ip: str, timeout: float) -> None:
    """Best-effort knock so the OS resolves the host's MAC into its table.

    A TCP connect makes the kernel ARP for the destination before it can send a
    SYN, so even a refused/timed-out connect leaves a neighbour entry behind.
    """
    for port in _WARM_PORTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect((ip, port))
            return
        except OSError:
            # Refused/unreachable/timeout all still warmed (or could not warm)
            # the table; try the next port only on a clean failure to connect.
            continue


def _mac_from_proc(ip: str) -> Optional[str]:
    """Read the Linux ARP table (``/proc/net/arp``) for ``ip``'s MAC."""
    if sys.platform != "linux":
        return None
    try:
        with open("/proc/net/arp", "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    for line in lines[1:]:  # skip the header row
        fields = line.split()
        if len(fields) >= 4 and fields[0] == ip:
            return _normalise_mac(fields[3])
    return None


def _mac_from_command(ip: str, timeout: float) -> Optional[str]:
    """Fall back to one native neighbour-table command for this platform."""
    argv = _neighbour_command(ip)
    if argv is None:
        return None
    try:
        completed = run_command(
            argv,
            action="resolve MAC from OS neighbour table",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _first_mac(completed.stdout)


def _neighbour_command(ip: str) -> Optional[list[str]]:
    if sys.platform == "win32":
        return ["arp", "-a", ip]
    if sys.platform == "darwin":
        return ["arp", "-n", ip]
    if sys.platform == "linux":
        return ["ip", "neigh", "show", ip]
    return None


def _normalise_mac(value: str) -> Optional[str]:
    mac = value.strip().lower().replace("-", ":")
    if _MAC_RE.fullmatch(mac) and mac not in _EMPTY_MACS:
        return mac
    return None


def _first_mac(text: str) -> Optional[str]:
    for match in _MAC_RE.finditer(text.lower()):
        mac = _normalise_mac(match.group(0))
        if mac is not None:
            return mac
    return None


def resolve_mac(host: str, *, warm: bool = True, timeout: float = 0.3) -> Optional[str]:
    """Return ``host``'s MAC from the OS neighbour table, or ``None``.

    Reads the table (warming it first with a best-effort TCP knock when ``warm``);
    returns a normalised lowercase ``aa:bb:cc:dd:ee:ff`` for a same-segment host
    the OS has resolved, else ``None`` (off-segment, unreachable, or unknown OS).
    Never raises and never needs elevated privileges.
    """
    cached = _mac_cache.get(host)
    if cached is not None:
        return cached
    if host in _mac_negative_cache:
        return None
    ip = _resolve_host_ip(host)
    if ip is None:
        _mac_negative_cache[host] = True
        return None
    mac = _mac_from_proc(ip)
    if mac is None and warm:
        _warm_neighbour_table(ip, timeout)
        mac = _mac_from_proc(ip)
    if mac is None:
        mac = _mac_from_command(ip, timeout)
    if mac is not None:
        _mac_cache[host] = mac
    else:
        _mac_negative_cache[host] = True
    return mac
