"""The raw datagram discovery family: one payload out, raw replies in.

Drives a :class:`~simplyprint_ws_client.integration.discovery.spec.DatagramSpec`
two ways:

* on demand, :func:`sweep`: send the payload once (to every port), collect
  replies for ``timeout`` seconds, return the parsed records deduplicated by
  key;
* always on, :func:`datagram_listener`: the same exchange as a
  :class:`~simplyprint_ws_client.integration.discovery.spec.ListenerSpec`
  whose protocol re-sends every ``interval`` (or only listens), so it runs on
  the discovery host like any brand-owned listener.

:class:`DatagramCollector` is the parsing half both share, free of I/O.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
from typing import Any, Dict, List, Optional, Tuple

from simplyprint_ws_client.integration.discovery.listener import (
    DatagramReceived,
    ListenerAction,
    ListenerEvent,
    SendTo,
    Sighted,
    WakeAfter,
)
from simplyprint_ws_client.integration.discovery.spec import (
    DatagramSpec,
    ListenerSpec,
)

__all__ = ["DatagramCollector", "datagram_listener", "sweep"]

_logger = logging.getLogger("discovery")


class DatagramCollector:
    """Parses raw replies with a spec and keeps the records by key."""

    def __init__(self, spec: DatagramSpec) -> None:
        self.spec = spec
        self._records: Dict[str, Any] = {}

    def parse(self, data: bytes, source: Tuple[Any, ...]) -> Optional[Any]:
        """The spec's record for one reply, or ``None`` (a parser that raises
        counts as no record)."""
        try:
            return self.spec.parse(bytes(data), (source[0], source[1]))
        except Exception:
            _logger.debug("datagram parse failed for %s", self.spec.brand, exc_info=True)
            return None

    def feed(self, data: bytes, source: Tuple[Any, ...]) -> Optional[Any]:
        """Parse one reply and keep its record (the latest per key wins)."""
        record = self.parse(data, source)
        if record is None:
            return None
        try:
            key = self.spec.key(record)
        except Exception:
            _logger.debug("datagram key failed for %s", self.spec.brand, exc_info=True)
            return None
        self._records[key] = record
        return record

    def records(self) -> List[Any]:
        return list(self._records.values())

    def sends(self) -> List[SendTo]:
        """The datagrams one round sends (none for a listen-only spec)."""
        if self.spec.payload is None:
            return []
        return [
            SendTo(self.spec.payload, self.spec.address, port)
            for port in self.spec.ports
        ]


def _sweep_socket(spec: DatagramSpec) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if spec.listen_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if spec.multicast_ttl is not None:
            sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, spec.multicast_ttl
            )
        sock.bind(("", spec.listen_port))
        sock.setblocking(False)
    except BaseException:
        sock.close()
        raise
    return sock


async def sweep(spec: DatagramSpec, timeout: float) -> List[Any]:
    """Send ``spec``'s payload once and return what answered in ``timeout``.

    A socket the OS refuses (no broadcast route, no permission) yields an
    empty list, as an unreachable subnet does for a scan.
    """
    collector = DatagramCollector(spec)
    loop = asyncio.get_running_loop()

    class _Collect(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, addr) -> None:
            collector.feed(data, addr)

    try:
        sock = _sweep_socket(spec)
    except OSError:
        _logger.warning("datagram sweep for %s could not open a socket", spec.brand)
        _logger.debug("datagram sweep socket failed", exc_info=True)
        return []
    transport, _ = await loop.create_datagram_endpoint(_Collect, sock=sock)
    try:
        for send in collector.sends():
            try:
                transport.sendto(send.data, (send.host, send.port))
            except OSError:
                _logger.debug(
                    "datagram sweep send failed for %s", spec.brand, exc_info=True
                )
        await asyncio.sleep(max(0.0, timeout))
    finally:
        transport.close()
    return collector.records()


class _DatagramFamilyProtocol:
    """The always-on half: send each round, sight every parsed reply."""

    def __init__(self, spec: DatagramSpec) -> None:
        self._collector = DatagramCollector(spec)
        self._interval = spec.interval

    def handle(self, event: ListenerEvent) -> List[ListenerAction]:
        if isinstance(event, DatagramReceived):
            record = self._collector.parse(event.data, event.source)
            return [Sighted(record)] if record is not None else []
        # ListenerStarted or ListenerTick: one round.
        actions: List[ListenerAction] = list(self._collector.sends())
        if actions and self._interval is not None:
            actions.append(WakeAfter(self._interval))
        return actions


def datagram_listener(spec: DatagramSpec) -> ListenerSpec:
    """``spec``'s always-on listener, keyed ``spec.brand``."""
    return ListenerSpec(
        brand=spec.brand,
        protocol=lambda: _DatagramFamilyProtocol(spec),
        key=spec.key,
        port=spec.listen_port,
        group=spec.group,
        broadcast=True,
        multicast_ttl=spec.multicast_ttl,
        event_type=spec.event_type,
    )
