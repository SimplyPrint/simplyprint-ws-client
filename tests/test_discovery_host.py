"""DiscoveryServiceHost runs every multicast listener on ONE thread + loop.

Brand-neutral (no brand imports): two passive listener specs are bound on free
loopback ports and fed REAL UDP datagrams. These pin (1) both listeners receive
on the single shared host loop, (2) exactly one ``discovery-host`` thread exists
and no per-brand ``discovery-<brand>`` thread, (3) a stopped listener is restarted
on the same loop -- without spawning a new thread, and (4) shutdown joins cleanly.

NOTE: true cross-interface multicast and a real device answering an active
M-SEARCH cannot be exercised without hardware/a LAN; this loopback unicast-to-
bound-port test (with the real IP_ADD_MEMBERSHIP group-join) is the closest
faithful check. Green here is not on-wire proof.
"""

import socket
import threading
import time

import pytest

from simplyprint_ws_client.common.events import Event

from simplyprint_ws_client.contrib.discovery.service import DiscoveryService
from simplyprint_ws_client.contrib.discovery.spec import MulticastSpec


class _FoundEvent(Event): ...


def _mapper(request, addr):
    usn = request.headers.get("usn")
    if not usn:
        return None
    return {"id": usn, "host": addr[0]}


def _passive_spec(brand: str, port: int) -> MulticastSpec:
    # No search_payload -> passive: binds the fixed port and joins the group.
    return MulticastSpec(
        brand=brand,
        group="239.255.255.250",
        port=port,
        event_type=_FoundEvent,
        mapper=_mapper,
        key=lambda record: record["id"],
    )


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _send_ssdp(port: int, usn: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(
            f"NOTIFY * HTTP/1.1\r\nUSN: {usn}\r\n\r\n".encode(), ("127.0.0.1", port)
        )
    finally:
        sock.close()


def _ids(backend) -> set:
    return {device["id"] for device in backend.get_devices()}


def _receives(port: int, usn: str, backend, timeout: float = 8.0) -> bool:
    """Poll-with-resend (UDP is lossy and the bind is async): keep sending until
    the backend's cache shows the device, or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _send_ssdp(port, usn)
        if usn in _ids(backend):
            return True
        time.sleep(0.05)
    return usn in _ids(backend)


def _discovery_thread_names() -> list:
    return [t.name for t in threading.enumerate() if t.name.startswith("discovery-")]


def test_host_runs_two_listeners_on_one_thread_and_both_receive():
    port_a, port_b = _free_udp_port(), _free_udp_port()
    service = DiscoveryService(
        multicast_specs=[_passive_spec("alpha", port_a), _passive_spec("beta", port_b)],
        restart_interval=0.2,
    )
    service.start()
    try:
        backend_a = service._multicast["alpha"]
        backend_b = service._multicast["beta"]

        assert _receives(port_a, "dev-a", backend_a)
        assert _receives(port_b, "dev-b", backend_b)

        # One shared host thread; no per-brand discovery-<brand> thread survives.
        names = _discovery_thread_names()
        assert names.count("discovery-host") == 1
        assert [n for n in names if n != "discovery-host"] == []
    finally:
        service.stop()

    # Shutdown joined the host thread.
    assert "discovery-host" not in _discovery_thread_names()


def test_discovery_service_start_is_idempotent():
    port = _free_udp_port()
    service = DiscoveryService(
        multicast_specs=[_passive_spec("alpha", port)],
        restart_interval=0.2,
    )
    service.start()
    try:
        first_host = service._host
        service.start()

        assert service._host is first_host
        assert _discovery_thread_names().count("discovery-host") == 1
    finally:
        service.stop()


def test_discovery_service_rejects_duplicate_multicast_specs():
    port_a, port_b = _free_udp_port(), _free_udp_port()

    with pytest.raises(ValueError, match="duplicate multicast discovery spec: alpha"):
        DiscoveryService(
            multicast_specs=[
                _passive_spec("alpha", port_a),
                _passive_spec("alpha", port_b),
            ],
        )


def test_host_restarts_a_stopped_listener_without_a_new_thread():
    port = _free_udp_port()
    service = DiscoveryService(
        multicast_specs=[_passive_spec("alpha", port)],
        restart_interval=0.2,
    )
    service.start()
    try:
        backend = service._multicast["alpha"]
        assert _receives(port, "dev-1", backend)

        # Kill the listener coroutine; the host's supervisor relaunches it on the
        # SAME loop. A fresh datagram landing proves the relaunched listener rebound.
        backend.stop()
        assert _receives(port, "dev-2", backend)

        # Restart did not spawn a per-brand thread; still just the one host thread.
        names = _discovery_thread_names()
        assert names.count("discovery-host") == 1
        assert [n for n in names if n != "discovery-host"] == []
    finally:
        service.stop()
