"""Regression coverage for host-network IP/MAC resolution."""

import socket
import types

import psutil


def test_no_mac_reported_when_no_interface_matches(monkeypatch):
    from simplyprint_ws_client.common.hardware import host_network

    class _FakeSocket:
        def __init__(self, *args, **kwargs): ...

        def settimeout(self, value): ...

        def connect(self, addr): ...

        def getsockname(self):
            return ("10.99.99.99", 0)

        def close(self): ...

    monkeypatch.setattr(host_network.socket, "socket", _FakeSocket)
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "eth0": [
                types.SimpleNamespace(
                    family=psutil.AF_LINK, address="aa:bb:cc:dd:ee:ff"
                ),
                types.SimpleNamespace(family=socket.AF_INET, address="192.168.1.2"),
            ]
        },
    )

    info = host_network.get_local_ip_and_mac()

    assert info.ip == "10.99.99.99"
    # No interface carries that ip, so no MAC may be guessed.
    assert info.mac is None


def test_mac_reported_for_matching_interface(monkeypatch):
    from simplyprint_ws_client.common.hardware import host_network

    class _FakeSocket:
        def __init__(self, *args, **kwargs): ...

        def settimeout(self, value): ...

        def connect(self, addr): ...

        def getsockname(self):
            return ("192.168.1.2", 0)

        def close(self): ...

    monkeypatch.setattr(host_network.socket, "socket", _FakeSocket)
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "wlan0": [
                types.SimpleNamespace(
                    family=psutil.AF_LINK, address="11:22:33:44:55:66"
                ),
                types.SimpleNamespace(family=socket.AF_INET, address="192.168.1.2"),
            ]
        },
    )

    info = host_network.get_local_ip_and_mac()

    assert info == ("192.168.1.2", "11:22:33:44:55:66")
