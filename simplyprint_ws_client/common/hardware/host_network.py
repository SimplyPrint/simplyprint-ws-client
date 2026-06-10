"""Host network telemetry: the connector machine's own local IP and MAC.

This is *host* networking — the IP/MAC of the machine running the connector,
reported as connector telemetry — not printer identity. Any integration that
needs to tell the backend "which interface am I on" can use it unchanged.
"""

import socket
from typing import NamedTuple, Optional

import psutil

#: Any public address works here: the socket is never written to, it only makes
#: the kernel pick the outbound interface so we can read this host's local ip.
_OUTBOUND_PROBE_ADDR = ("168.119.98.102", 80)


class NetworkInfo(NamedTuple):
    """Network information tuple."""

    ip: str
    mac: Optional[str]


def get_local_ip_and_mac() -> NetworkInfo:
    """Get the local IP and MAC address of the machine.

    The MAC is only reported when an interface actually carries ``ip``;
    callers correlate hardware identity on it, so a guess is worse than None.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(_OUTBOUND_PROBE_ADDR)
        local_ip = s.getsockname()[0]
    except socket.error:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    mac: Optional[str] = None

    for iface, addrs in psutil.net_if_addrs().items():
        if iface == "lo":
            continue

        iface_mac = None
        found = False

        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == local_ip:
                found = True
            if addr.family == psutil.AF_LINK:
                iface_mac = addr.address

        if found:
            mac = iface_mac
            break

    return NetworkInfo(ip=local_ip, mac=mac)
