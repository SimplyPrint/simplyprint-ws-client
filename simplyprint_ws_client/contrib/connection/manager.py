"""Brand-agnostic connection management on the transport/lease primitives.

The successor to the ``ClientBucket`` + ``Connection`` + ``ConnectionManager``
trio: :class:`PooledConnectionManager` owns the register / remove / refresh /
keepalive lifecycle for many clients, but delegates transport *sharing* and
message *routing* to a :class:`~.transport.Pool`'s ``connect() -> Connection``
lease. Each client gets a lease scoped to its route (an MQTT topic, or ``None``
for a 1:1 socket); the lease delivers that client's messages and connect/
disconnect events **on the consumer loop** (its courier does the hop), so the
manager wires them straight to the client's own event bus -- there is no separate
dispatch worker thread.

A brand subclass supplies the hooks: which :class:`~.transport.Pool` to build
(:meth:`_make_pool`), how a config becomes params (:attr:`params_factory`), the
brand event types, the keepalive command (:meth:`_send_keepalive`), and -- if not
a topic -- the route (:meth:`_route_for`).
"""

from __future__ import annotations

import logging
import threading
import time
from abc import abstractmethod
from typing import (
    Callable,
    Dict,
    Generic,
    Hashable,
    Optional,
    Protocol,
    Set,
    TypeVar,
    runtime_checkable,
)

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.shared.utils.bounded_variable import (
    BoundedInterval,
    BoundedVariable,
)
from simplyprint_ws_client.shared.utils.stoppable import SyncStoppable

from simplyprint_ws_client.contrib.connection.transport import (
    Connection,
    Pool,
)

__all__ = [
    "KEEPALIVE_TIMEOUT_MS",
    "ConnectionAttemptsBoundedInterval",
    "PoolClient",
    "PooledConnectionManager",
    "now_ms",
]

#: Time without an inbound message before the keepalive loop pokes a client.
KEEPALIVE_TIMEOUT_MS = 20_000

#: How many failed connect/keepalive attempts before we declare a disconnect.
ConnectionAttemptsBoundedInterval: BoundedInterval[int] = BoundedInterval(3, 1, 0)

TParams = TypeVar("TParams", bound=Hashable)


def now_ms() -> int:
    """Wall-clock milliseconds; the manager's clock for keepalive bookkeeping."""
    return int(time.time() * 1000)


@runtime_checkable
class PoolClient(Protocol):
    """The surface a pooled manager needs from a client.

    Unchanged from the old pool except the dispatch seam: a client exposes its
    :class:`EventBus` directly (the lease delivers on the loop, so events are
    emitted to it in place) rather than a separate threaded ``event_bus_worker``.
    """

    config: object
    logger: logging.Logger
    #: The client's event bus; the manager emits brand events onto it (on the loop).
    event_bus: EventBus
    #: Wall-clock ms of the last inbound message (the client keeps this fresh).
    last_message_at: int
    keepalive_attempts: BoundedVariable[int]

    @property
    def report_topic(self) -> str: ...

    @property
    def connected(self) -> bool: ...


TClient = TypeVar("TClient", bound=PoolClient)


class PooledConnectionManager(SyncStoppable, Generic[TClient, TParams]):
    """Owns the client lifecycle + keepalive over a shared :class:`Pool`."""

    logger: logging.Logger = logging.getLogger("connection_manager")

    #: Brand event types emitted to clients. Set by the concrete subclass.
    connected_event: Hashable
    disconnected_event: Hashable
    message_event: Hashable
    #: How a client's config becomes hashable connection params.
    params_factory: Callable[[object], TParams]
    keepalive_timeout_ms: int = KEEPALIVE_TIMEOUT_MS

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._pool: Pool = self._make_pool()
        self._leases: Dict[TClient, Connection] = {}
        self._params: Dict[TClient, TParams] = {}
        self._registering: Dict[TClient, object] = {}
        self._cancelled_registrations: Set[object] = set()
        #: Clients that asked to register; retried by :meth:`reconcile_registrations`
        #: until their config can produce params -- no per-client retry thread.
        self._wanted: Set[TClient] = set()
        self._lock = threading.Lock()

    @abstractmethod
    def _make_pool(self) -> Pool:
        """Build the wire pool (e.g. an MQTT or threaded-WS pool)."""
        raise NotImplementedError

    def _route_for(self, client: TClient) -> Optional[Hashable]:
        """The lease route for ``client`` -- its report topic by default; a 1:1
        socket brand returns ``None`` (every message on the link is the client's)."""
        return client.report_topic

    def _send_keepalive(self, client: TClient) -> None:
        """Poke an idle client (brand keepalive command)."""

    def _on_suspect(self, client: TClient, event: object) -> None:
        """A suspect connect for ``client`` (repeated failure / CONNACK reject).
        Default: no-op. A brand overrides to re-handshake / refresh credentials
        (typically via :meth:`submit_to_consumer`)."""

    def _emit_to_client(self, client: TClient, event: Hashable, *args: object) -> None:
        client.event_bus.emit_sync(event, *args)

    def _deliver(self, client: TClient, event: Hashable, *args: object) -> None:
        def deliver() -> None:
            self._emit_to_client(client, event, *args)

        call_to_consumer = getattr(self._pool, "call_to_consumer", None)
        if call_to_consumer is not None:
            call_to_consumer(deliver)
        else:
            deliver()

    def add_client(self, client: TClient) -> None:
        # Idempotent: a reconcile sweep may race a successful registration.
        token = object()
        with self._lock:
            if client in self._leases or client in self._registering:
                return
            self._registering[client] = token

        try:
            # Transactional: validate params BEFORE leasing, so a client whose config
            # can't yet produce params stays un-leased and cleanly retryable.
            params = self.params_factory(client.config)
            if params is None:
                raise ValueError("Failed to create connection params")

            lease = self._pool.connect(params, route=self._route_for(client))
            lease.on_message(
                lambda msg, c=client: self._deliver(c, self.message_event, msg)
            )
            lease.on_connected(lambda c=client: self._deliver(c, self.connected_event))
            lease.on_disconnected(
                lambda e, c=client: self._deliver(c, self.disconnected_event, e.reason)
            )
            lease.on_suspect(lambda e, c=client: self._on_suspect(c, e))

            close_lease = False
            with self._lock:
                self._registering.pop(client, None)
                if token in self._cancelled_registrations:
                    self._cancelled_registrations.discard(token)
                    close_lease = True
                else:
                    self._leases[client] = lease
                    self._params[client] = params
        except Exception:
            with self._lock:
                self._registering.pop(client, None)
                self._cancelled_registrations.discard(token)
            raise

        if close_lease:
            lease.close()
            return

        # If the shared link is already up, let this client know now.
        if lease.connected:
            self._deliver(client, self.connected_event)

    def request_registration(self, client: TClient) -> None:
        """Register ``client`` now, retried by the keepalive sweep until it lands."""
        with self._lock:
            self._wanted.add(client)
        self._try_register(client)

    def _try_register(self, client: TClient) -> bool:
        try:
            self.add_client(client)
            return True
        except Exception as e:  # noqa: BLE001 -- deferred, retried on next sweep
            client.logger.warning("Deferring registration: %s", e)
            return False

    def reconcile_registrations(self) -> None:
        """Re-attempt registration for any wanted client not yet leased."""
        with self._lock:
            wanted = list(self._wanted)
        for client in wanted:
            with self._lock:
                should_try = (
                    client not in self._leases and client not in self._registering
                )
            if should_try:
                self._try_register(client)

    def refresh_client(self, client: TClient) -> None:
        """Recompute a client's params (config changed) and reconnect it."""
        with self._lock:
            leased = client in self._leases
        if not leased:
            return
        self.remove_client(client)
        self.request_registration(client)

    def remove_client(self, client: TClient) -> None:
        with self._lock:
            self._wanted.discard(client)
            token = self._registering.pop(client, None)
            if token is not None:
                self._cancelled_registrations.add(token)
            lease = self._leases.pop(client, None)
            self._params.pop(client, None)
        if lease is not None:
            lease.close()

    def get_connection_from_client(self, client: TClient) -> Optional[Connection]:
        """This client's lease, or ``None`` if it isn't registered/leased yet.

        The handle a client drives directly (e.g. ``send``); the manager owns its
        lifecycle, the client just borrows it to publish."""
        with self._lock:
            return self._leases.get(client)

    def keepalive_check(self) -> None:
        # Re-attempt deferred registrations first (offline-at-boot clients land here).
        self.reconcile_registrations()

        with self._lock:
            clients = list(self._leases.keys())

        for client in clients:
            if not client.connected:
                continue  # transport reconnects on its own
            if now_ms() - client.last_message_at < self.keepalive_timeout_ms:
                continue  # heard from it recently
            if client.keepalive_attempts.guard_until_bound():
                self._deliver(
                    client, self.disconnected_event, "Keepalive failed enough times"
                )
                continue
            self._send_keepalive(client)
            # Give the printer a chance to answer before we poke again.
            client.last_message_at = now_ms() - (self.keepalive_timeout_ms // 2)

    def submit_to_consumer(
        self,
        coro_factory: Callable[[], object],
        *,
        coalesce_key: Optional[Hashable] = None,
    ) -> None:
        """Run async work on the consumer loop from a transport thread (the
        sanctioned cross-thread hop, e.g. an auth re-handshake). Delegates to the
        pool's :class:`~.transport.ConsumerLoop`."""
        self._pool.submit_to_consumer(coro_factory, coalesce_key=coalesce_key)

    def stop(self) -> None:
        self.logger.info("Stopping connection manager")
        super().stop()
        with self._lock:
            leases = list(self._leases.values())
            self._leases.clear()
            self._params.clear()
            self._wanted.clear()
            self._cancelled_registrations.update(self._registering.values())
            self._registering.clear()
        for lease in leases:
            lease.close()
        pool_stop = getattr(self._pool, "stop", None)
        if pool_stop is not None:
            pool_stop()
