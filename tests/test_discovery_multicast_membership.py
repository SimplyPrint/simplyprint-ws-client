import asyncio
import errno
import logging
import socket
from types import SimpleNamespace

import pytest

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.integration.discovery import netif
from simplyprint_ws_client.integration.discovery.multicast_base import (
    GroupMembership,
    send_from_interfaces,
)

GROUP = "239.255.255.250"
LOGGER = logging.getLogger("test")


def _mreq(interface: str) -> bytes:
    return socket.inet_aton(GROUP) + socket.inet_aton(interface)


class _Socket:
    """Records the multicast options and sends a real socket would get."""

    def __init__(self, fail_join=None, fail_send=None):
        self.joins: list[bytes] = []
        self.sent: list[tuple[str | None, bytes]] = []
        self._fail_join = fail_join or {}
        self._fail_send = fail_send or set()
        self._out = None

    def setsockopt(self, level, option, value):
        assert level == socket.IPPROTO_IP
        if option == socket.IP_ADD_MEMBERSHIP:
            error = self._fail_join.get(value)
            if error is not None:
                raise OSError(error, "join failed")
            self.joins.append(value)
        elif option == socket.IP_MULTICAST_IF:
            self._out = socket.inet_ntoa(value)

    def sendto(self, payload, address):
        if self._out in self._fail_send:
            raise BlockingIOError(errno.EAGAIN, "would block")
        self.sent.append((self._out, payload))


def test_joins_the_group_on_every_interface():
    sock = _Socket()
    membership = GroupMembership(GROUP, LOGGER, lambda: ["192.168.1.20", "172.20.0.1"])

    membership.refresh(sock)

    assert sorted(sock.joins) == sorted([_mreq("192.168.1.20"), _mreq("172.20.0.1")])
    assert membership.joined == {"192.168.1.20", "172.20.0.1"}
    assert membership.default_joined is False


def test_an_interface_already_joined_counts_and_one_that_fails_is_skipped():
    sock = _Socket(
        fail_join={
            _mreq("10.8.0.2"): errno.EADDRINUSE,
            _mreq("169.254.3.3"): errno.ENODEV,
        }
    )
    membership = GroupMembership(
        GROUP, LOGGER, lambda: ["10.8.0.2", "169.254.3.3", "192.168.1.20"]
    )

    membership.refresh(sock)

    assert membership.joined == {"10.8.0.2", "192.168.1.20"}
    assert membership.default_joined is False


def test_falls_back_to_the_default_interface_when_none_can_be_joined():
    sock = _Socket(fail_join={_mreq("192.168.1.20"): errno.ENODEV})
    membership = GroupMembership(GROUP, LOGGER, lambda: ["192.168.1.20"])

    membership.refresh(sock)

    assert membership.joined == set()
    assert membership.default_joined is True
    assert sock.joins == [_mreq("0.0.0.0")]


def test_no_interfaces_listed_is_the_old_default_join():
    sock = _Socket()

    GroupMembership(GROUP, LOGGER, lambda: []).refresh(sock)

    assert sock.joins == [_mreq("0.0.0.0")]


def test_raises_only_when_not_even_the_default_interface_joins():
    sock = _Socket(fail_join={_mreq("0.0.0.0"): errno.ENODEV})

    with pytest.raises(OSError):
        GroupMembership(GROUP, LOGGER, lambda: []).refresh(sock)


def test_refresh_joins_a_new_interface_and_forgets_a_vanished_one():
    interfaces = ["192.168.1.20"]
    sock = _Socket()
    membership = GroupMembership(GROUP, LOGGER, lambda: list(interfaces))
    membership.refresh(sock)

    # A VPN comes up, and later the Wi-Fi changes network.
    interfaces.append("10.8.0.2")
    assert membership.refresh(sock) is True
    interfaces.remove("192.168.1.20")
    assert membership.refresh(sock) is True
    assert membership.refresh(sock) is False

    assert membership.joined == {"10.8.0.2"}
    assert sock.joins == [_mreq("192.168.1.20"), _mreq("10.8.0.2")]

    # Back on the first network: joined again (the OS dropped the old membership).
    interfaces.append("192.168.1.20")
    membership.refresh(sock)
    assert sock.joins[-1] == _mreq("192.168.1.20")


def test_sends_one_copy_out_of_each_joined_interface():
    sock = _Socket()
    membership = GroupMembership(GROUP, LOGGER, lambda: ["192.168.1.20", "10.8.0.2"])
    membership.refresh(sock)

    membership.send(sock, b"M-SEARCH", (GROUP, 1900))

    assert sorted(sock.sent) == [
        ("10.8.0.2", b"M-SEARCH"),
        ("192.168.1.20", b"M-SEARCH"),
    ]


def test_a_failing_interface_does_not_stop_the_others():
    sock = _Socket(fail_send={"10.8.0.2"})

    send_from_interfaces(
        sock, b"q", (GROUP, 5353), ["10.8.0.2", "192.168.1.20"], LOGGER
    )

    assert sock.sent == [("192.168.1.20", b"q")]


def test_without_interfaces_it_sends_once_through_the_default():
    sock = _Socket()

    send_from_interfaces(sock, b"q", (GROUP, 5353), [], LOGGER)

    assert sock.sent == [(None, b"q")]


def test_multicast_interfaces_lists_up_ipv4_without_loopback(monkeypatch):
    def addr(family, address):
        return SimpleNamespace(family=family, address=address, netmask="255.255.255.0")

    monkeypatch.setattr(
        netif.psutil,
        "net_if_addrs",
        lambda: {
            "lo0": [addr(socket.AF_INET, "127.0.0.1")],
            "en0": [
                addr(socket.AF_INET, "192.168.1.20"),
                addr(socket.AF_INET6, "fe80::1"),
            ],
            "utun3": [addr(socket.AF_INET, "10.8.0.2")],
            "bridge0": [addr(socket.AF_INET, "172.20.0.1")],
        },
    )
    monkeypatch.setattr(
        netif.psutil,
        "net_if_stats",
        lambda: {
            "lo0": SimpleNamespace(isup=True),
            "en0": SimpleNamespace(isup=True),
            "utun3": SimpleNamespace(isup=True),
            "bridge0": SimpleNamespace(isup=False),
        },
    )

    assert netif.multicast_interfaces() == ["10.8.0.2", "192.168.1.20"]


def test_multicast_interfaces_is_empty_when_they_cannot_be_listed(monkeypatch):
    def boom():
        raise OSError("no permission")

    monkeypatch.setattr(netif.psutil, "net_if_addrs", boom)

    assert netif.multicast_interfaces() == []


class _Transport:
    def __init__(self):
        self.sent = []

    def sendto(self, payload, address):
        self.sent.append((payload, address))


def _active_spec():
    from simplyprint_ws_client.integration.discovery.spec import MulticastSpec

    return MulticastSpec(
        brand="probe",
        group=GROUP,
        port=1900,
        event_type=object,
        mapper=lambda request, addr: None,
        key=lambda record: record,
        search_payload=b"M-SEARCH",
        search_interval=60,
    )


def test_a_listener_sends_through_its_membership_once_running():
    from simplyprint_ws_client.integration.discovery.multicast import (
        MulticastDiscoveryBackend,
    )

    backend = MulticastDiscoveryBackend(_active_spec(), EventBus())
    transport = _Transport()

    # Not running yet: straight through the transport, as before.
    backend.send_to_group(transport, b"M-SEARCH")
    assert transport.sent == [(b"M-SEARCH", (GROUP, 1900))]

    sock = _Socket()
    backend._membership = GroupMembership(
        GROUP, LOGGER, lambda: ["192.168.1.20", "10.8.0.2"]
    )
    backend._membership.refresh(sock)
    backend._sock = sock
    backend.send_to_group(transport, b"M-SEARCH")

    assert len(transport.sent) == 1
    assert sorted(out for out, _ in sock.sent) == ["10.8.0.2", "192.168.1.20"]


@pytest.mark.asyncio
async def test_the_listener_keeps_its_membership_current(monkeypatch):
    from simplyprint_ws_client.integration.discovery import multicast_base
    from simplyprint_ws_client.integration.discovery.multicast import (
        MulticastDiscoveryBackend,
    )

    monkeypatch.setattr(multicast_base, "MEMBERSHIP_REFRESH", 0.01)
    interfaces = ["192.168.1.20"]
    backend = MulticastDiscoveryBackend(_active_spec(), EventBus())
    backend._membership = GroupMembership(GROUP, LOGGER, lambda: list(interfaces))
    sock = _Socket()
    backend._membership.refresh(sock)

    keeper = asyncio.ensure_future(backend._keep_membership_current(sock))
    interfaces.append("10.8.0.2")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "10.8.0.2" in backend._membership.joined:
            break
    backend.stop()
    await asyncio.wait_for(keeper, 1)

    assert backend._membership.joined == {"192.168.1.20", "10.8.0.2"}
