"""Allocation, refcounting, and fan-out -- one shared transport per endpoint.

This is where "many printers share one socket" lives. A :class:`Pool` keeps ONE
:class:`~simplyprint_ws_client.wire.transport.Transport` per endpoint key,
ref-counted, and hands every caller a :class:`~simplyprint_ws_client.wire.lease.Lease`
lease. The first lease on a key builds and starts the transport and subscribes
once to its event bus; the last lease to close stops and drops the transport.

The pool subscribes to a transport's events exactly once and fans each event to
that transport's leases: lifecycle events (``Connecting`` / ``Connected`` /
``Disconnected``) go to every lease, while a ``MessageReceived`` goes only to the
leases whose subscription set matches the pool route function (no route function
means a 1:1 wire and reaches every lease). The per-lease bounded, QoS-governed
delivery of those events to async handlers lives in :mod:`connection`.

The pool is pure sync bookkeeping: ``connect`` and lease release only touch a
refcounted dict under one lock and never do I/O. On a sync wire (paho) the
transport already couriers its events onto the loop before they reach the pool, so
the pool stays thread-free either way.
"""

from __future__ import annotations

import logging
import threading
from typing import (
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

import yarl

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

from simplyprint_ws_client.wire.lease import Lease
from simplyprint_ws_client.wire.events import (
    Connected,
    Connecting,
    WireEvent,
    Disconnected,
    MessageReceived,
)
from simplyprint_ws_client.wire.transport import Transport, is_wildcard_filter

__all__ = ["Pool", "Endpoint"]

T = TypeVar("T", bound=Transport)

#: Builds (but does not start) a transport for a URL and the caller's params.
#: Generic over the pool's transport type; params stay ``object`` because each
#: front door defines its own params shape and narrows with ``isinstance``.
TransportBuilder = Callable[[yarl.URL, object], T]
#: Maps a URL and params to the hashable endpoint key the pool shares by.
EndpointKey = Callable[[yarl.URL, object], Hashable]
#: Maps an inbound message to the route leases match against. ``None`` broadcasts.
MessageRoute = Callable[[object], Optional[Hashable]]


class Endpoint(Generic[T]):
    """One shared transport, its leases, and its refcount.

    The pool keeps one of these per endpoint key. It subscribes to the
    transport's event bus once (on creation) and detaches on teardown, fanning
    each event into each lease's owned delivery drain.
    """

    def __init__(
        self,
        key: Hashable,
        transport: T,
        route: Optional[MessageRoute],
        logger: logging.Logger,
    ) -> None:
        self.key = key
        self.transport = transport
        self.route = route
        self.logger = logger
        self.leases: Set["Lease[T]"] = set()
        self.unfiltered_leases: Set["Lease[T]"] = set()
        self.route_leases: Dict[Hashable, Set["Lease[T]"]] = {}
        self.wildcard_leases: Set["Lease[T]"] = set()

    @property
    def refs(self) -> int:
        """Live lease count -- the endpoint's refcount, derived from the lease set
        (every ``Pool.connect`` adds one unique lease; ``release`` removes it)."""
        return len(self.leases)

    def add_lease(self, lease: "Lease[T]") -> None:
        self.leases.add(lease)
        if not lease.topics:
            self.unfiltered_leases.add(lease)

    def remove_lease(self, lease: "Lease[T]") -> None:
        self.leases.discard(lease)
        self.unfiltered_leases.discard(lease)
        self.wildcard_leases.discard(lease)
        for route, leases in list(self.route_leases.items()):
            leases.discard(lease)
            if not leases:
                self.route_leases.pop(route, None)

    def add_route(self, lease: "Lease[T]", route: Hashable) -> None:
        self.unfiltered_leases.discard(lease)
        if is_wildcard_filter(route):
            self.wildcard_leases.add(lease)
            return
        self.route_leases.setdefault(route, set()).add(lease)

    def remove_route(self, lease: "Lease[T]", route: Hashable) -> None:
        if is_wildcard_filter(route):
            self.wildcard_leases.discard(lease)
        else:
            leases = self.route_leases.get(route)
            if leases is not None:
                leases.discard(lease)
                if not leases:
                    self.route_leases.pop(route, None)
        if lease in self.leases and not lease.topics:
            self.unfiltered_leases.add(lease)

    def attach(self) -> None:
        """Begin fanning the transport's events out to the leases."""
        self.transport.events.on(Connecting, self.fan_lifecycle)
        self.transport.events.on(Connected, self.fan_lifecycle)
        self.transport.events.on(Disconnected, self.fan_lifecycle)
        self.transport.events.on(MessageReceived, self.fan_message)

    def detach(self) -> None:
        """Stop fanning the transport's events out."""
        self.transport.events.off(Connecting, self.fan_lifecycle)
        self.transport.events.off(Connected, self.fan_lifecycle)
        self.transport.events.off(Disconnected, self.fan_lifecycle)
        self.transport.events.off(MessageReceived, self.fan_message)

    async def fan_lifecycle(self, event: WireEvent) -> None:
        """A lifecycle event reaches every lease on this transport."""
        for lease in tuple(self.leases):
            await self.emit_to_lease(lease, event)

    async def fan_message(self, event: MessageReceived) -> None:
        """A message reaches only the leases whose route matches it.

        A ``None`` routing key (a 1:1 wire) reaches every lease.
        """
        route = self.route(event.message) if self.route is not None else None
        if route is None:
            leases = tuple(self.leases)
        else:
            targeted = set(self.unfiltered_leases)
            targeted.update(self.route_leases.get(route, ()))
            if self.wildcard_leases:
                targeted.update(
                    lease for lease in self.wildcard_leases if lease.wants(route)
                )
            leases = tuple(targeted)
        for lease in leases:
            await self.emit_to_lease(lease, event)

    async def emit_to_lease(self, lease: "Lease[T]", event: WireEvent) -> None:
        if lease.closed:
            return
        try:
            lease.deliver(event)
        except Exception:  # noqa: BLE001 -- one bad lease must not drop the transport
            self.logger.warning("connection listener failed", exc_info=True)


class Pool(Generic[T]):
    """Shares one transport per endpoint across many connection leases.

    Construct it with ``build`` (make a transport for a URL + params) and ``key``
    (the hashable endpoint identity to share by). :meth:`connect` hands back a
    lease; the first lease on a key starts the transport, the last to close stops
    it.
    """

    def __init__(
        self,
        *,
        build: TransportBuilder[T],
        key: EndpointKey,
        route: Optional[MessageRoute] = None,
        lease_class: Optional[Type[Lease[T]]] = None,
        provider: Optional[EventLoopProvider] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.build = build
        self.key = key
        self.route = route
        #: The lease flavour to hand out -- :class:`Lease` by default, or a
        #: protocol-specific subclass (``MqttLease`` / ``WsLease``) a
        #: front door passes so callers get ``subscribe`` / framed ``send``.
        self.lease_class: Type[Lease[T]] = lease_class or Lease
        self.provider = provider or EventLoopProvider.default()
        self.logger = logger or logging.getLogger("wire.pool")
        self.endpoints: Dict[Hashable, Endpoint[T]] = {}
        #: Held only for refcount/dict bookkeeping -- never across I/O.
        self.lock = threading.Lock()

    def connect(self, url: Union[str, yarl.URL], params: object = None) -> "Lease[T]":
        """Lease the shared transport for ``url`` + ``params``.

        Synchronous and fire-and-forget: it refcounts and returns at once. Building
        the transport (the ``build`` callback) may raise on a bad URL or params.
        The first lease on an endpoint starts the transport; readiness is reported
        through events on the returned lease's bus.
        """
        parsed = yarl.URL(url) if isinstance(url, str) else url
        if not isinstance(parsed, yarl.URL):
            raise TypeError("Pool.connect url must be str or yarl.URL")
        endpoint_key = self.key(parsed, params)
        started: Optional[T] = None

        with self.lock:
            endpoint = self.endpoints.get(endpoint_key)
            if endpoint is None:
                transport = self.build(parsed, params)
                endpoint = Endpoint(endpoint_key, transport, self.route, self.logger)
                endpoint.attach()
                self.endpoints[endpoint_key] = endpoint
                started = transport
            shared = endpoint.transport
            lease = self.lease_class(
                self, shared, parsed, endpoint_key, provider=self.provider
            )
            endpoint.add_lease(lease)

        if started is not None:
            started.start()
        elif not shared.supervising():
            # A fresh lease on an endpoint whose transport permanently gave up
            # (bounded retry policy exhausted) re-arms supervision -- start()
            # resets the give-up and is a no-op on a live wire.
            shared.start()
        return lease

    def release(self, lease: "Lease[T]") -> Optional[T]:
        """Drop one lease's hold; return the transport to stop iff it was the last.

        The transport is returned (rather than stopped here) so the caller can
        ``await`` its async ``stop`` off the lock.
        """
        with self.lock:
            endpoint = self.endpoints.get(lease.endpoint_key)
            if endpoint is None or lease not in endpoint.leases:
                return None
            endpoint.remove_lease(lease)
            if endpoint.leases:
                return None
            endpoint.detach()
            self.endpoints.pop(endpoint.key, None)
            return endpoint.transport

    def stop(self) -> List[T]:
        """Detach every endpoint, clear the pool, and return the live transports.

        Releasing a lease after this finds no endpoint, so nobody is left to
        stop the sockets - the caller must await (or schedule) each returned
        transport's async ``stop``.
        """
        with self.lock:
            endpoints = list(self.endpoints.values())
            for endpoint in endpoints:
                endpoint.detach()
            self.endpoints.clear()
        return [endpoint.transport for endpoint in endpoints]

    def add_route(self, lease: "Lease[T]", route: Hashable) -> None:
        with self.lock:
            endpoint = self.endpoints.get(lease.endpoint_key)
            if endpoint is not None:
                endpoint.add_route(lease, route)

    def remove_route(self, lease: "Lease[T]", route: Hashable) -> None:
        with self.lock:
            endpoint = self.endpoints.get(lease.endpoint_key)
            if endpoint is not None:
                endpoint.remove_route(lease, route)
