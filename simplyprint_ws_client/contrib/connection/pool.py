"""Brand-agnostic connection pooling.

Generalised from the most mature brand implementation and proven to cover
others too. The pattern: many *clients* (one per printer) share a smaller
set of physical *connections*, keyed by a hashable ``params`` value derived from
each client's config. Clients with identical params share one connection.

Three collaborators, all transport-independent:

* :class:`ClientBucket` -- thread-safe registry of clients + topic/param routing.
* :class:`PooledConnection` -- one physical connection; transports subclass it.
* :class:`ConnectionManager` -- owns the bucket + connection pool and the
  register / remove / refresh / keepalive lifecycle.

Transport specifics (paho-mqtt, websockets, ...) live in sibling modules that
subclass :class:`PooledConnection` / :class:`ConnectionManager`. Brand specifics
(keepalive command, topic shape, auth refresh) are subclass hooks.
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
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    runtime_checkable,
)

from simplyprint_ws_client.shared.utils.bounded_variable import (
    BoundedInterval,
    BoundedVariable,
)
from simplyprint_ws_client.shared.utils.stoppable import SyncStoppable
from simplyprint_ws_client.shared.utils.synchronized import Synchronized

#: Time without an inbound message before the keepalive loop pokes a client.
KEEPALIVE_TIMEOUT_MS = 20_000

#: How many failed connect/keepalive attempts before we declare a disconnect.
ConnectionAttemptsBoundedInterval: BoundedInterval[int] = BoundedInterval(3, 1, 0)

TParams = TypeVar("TParams", bound=Hashable)


def now_ms() -> int:
    """Wall-clock milliseconds; the pool's clock for keepalive bookkeeping."""
    return int(time.time() * 1000)


@runtime_checkable
class EventBusWorker(Protocol):
    """The slice of a threaded event-bus worker the pool drives.

    The pool only ever fans a (brand-defined, but here opaque) event out to the
    clients on a connection; it does not care what the event is or who handles
    it, only that it can be emitted synchronously.
    """

    def emit_sync(self, event: Hashable, *args: object, **kwargs: object) -> object: ...


@runtime_checkable
class PoolClient(Protocol):
    """The surface a pooled connection / manager needs from a client.

    Every brand's low-level client already satisfies this; it is documented here
    so a *new* client only has to provide these members to join the pool.
    """

    #: Brand config; opaque to the pool, only handed back to ``params_factory``.
    config: object
    logger: logging.Logger
    #: Threaded event-bus worker; the pool emits brand events onto it.
    event_bus_worker: EventBusWorker
    #: Wall-clock ms of the last inbound message (the client keeps this fresh).
    last_message_at: int
    keepalive_attempts: BoundedVariable[int]

    @property
    def report_topic(self) -> str: ...

    @property
    def connected(self) -> bool: ...


TClient = TypeVar("TClient", bound=PoolClient)


class ClientBucket(Synchronized, Generic[TClient, TParams]):
    """Thread-safe registry of clients with topic- and param-based routing.

    ``params_factory`` turns a client's config into its hashable connection
    params (raising ``ValueError`` when the config can't yet produce them).
    ``wildcard_topics`` enables prefix matching for brands that subscribe to
    wildcard topics (e.g. ``.../#``).
    """

    def __init__(
        self,
        params_factory: Callable[[object], TParams],
        *,
        wildcard_topics: bool = False,
    ) -> None:
        Synchronized.__init__(self)
        self._params_factory = params_factory
        self._wildcard_topics = wildcard_topics
        self.clients: Set[TClient] = set()
        self.topic_to_client: Dict[str, TClient] = {}
        self.params_cache: Dict[TClient, TParams] = {}
        self.params_to_clients: Dict[TParams, Set[TClient]] = {}

    # -- params -------------------------------------------------------------

    def _get_or_create_params(self, client: TClient) -> Optional[TParams]:
        if client not in self.params_cache:
            try:
                self.params_cache[client] = self._params_factory(client.config)
            except ValueError as e:
                client.logger.warning("Failed to create connection params", exc_info=e)
                return None

        return self.params_cache.get(client)

    def _rebuild_params_cache(self, *clients: TClient) -> None:
        for client in list(clients or self.clients):
            if client not in self.clients:
                continue

            # Drop any stale cache entry, then recompute (config may have changed).
            self.params_cache.pop(client, None)
            params = self._get_or_create_params(client)

            # Invalidated params (deleted account, bad config) -> leave uncached.
            if not params:
                continue

            self.params_cache[client] = params

        # Forget clients that are no longer registered.
        for client in list(self.params_cache.keys()):
            if client not in self.clients:
                self.params_cache.pop(client, None)

    def _rebuild_client_params(self, *clients: TClient) -> None:
        for client in list(clients or self.clients):
            if client not in self.clients:
                continue

            param = self.params_cache.get(client)
            self.params_to_clients.setdefault(param, set()).add(client)

        # Prune empty param groups and drop deregistered clients.
        for param, param_clients in list(self.params_to_clients.items()):
            self.params_to_clients[param] = param_clients & self.clients
            if not self.params_to_clients[param]:
                self.params_to_clients.pop(param, None)

    def _rebuild_topic_cache(self, *clients: TClient) -> None:
        for client in list(clients or self.clients):
            if client not in self.clients:
                continue

            if client.report_topic not in self.topic_to_client:
                # Drop any stale topic that used to point at this client.
                self.topic_to_client = {
                    k: v for k, v in self.topic_to_client.items() if v != client
                }

            self.topic_to_client[client.report_topic] = client

    def rebuild(self, *clients: TClient) -> None:
        with self:
            self._rebuild_params_cache(*clients)
            self._rebuild_client_params(*clients)
            self._rebuild_topic_cache(*clients)

    # -- lookups ------------------------------------------------------------

    def get_from_topic(self, topic: str) -> Optional[TClient]:
        with self:
            client = self.topic_to_client.get(topic)

            # Rebuild once before giving up -- routing may be stale.
            if not client:
                self._rebuild_topic_cache()
                client = self.topic_to_client.get(topic)

            # Prefix match for wildcard subscribers (e.g. ``.../#``).
            if not client and self._wildcard_topics:
                for registered_topic, c in self.topic_to_client.items():
                    base = registered_topic.rstrip("/#")
                    if topic.startswith(base):
                        return c

            return client

    def get_from_params(self, params: TParams) -> List[TClient]:
        with self:
            return list(self.params_to_clients.get(params, []))

    def get_params(self, client: TClient) -> Optional[TParams]:
        with self:
            return self._get_or_create_params(client)

    # -- membership ---------------------------------------------------------

    def add(self, client: TClient) -> None:
        with self:
            self.clients.add(client)
            self._rebuild_params_cache(client)
            self._rebuild_client_params(client)
            self._rebuild_topic_cache(client)

    def remove(self, client: TClient) -> None:
        with self:
            self.clients.discard(client)
            self.params_cache.pop(client, None)
            self.topic_to_client = {
                k: v for k, v in self.topic_to_client.items() if v != client
            }
            self._rebuild_client_params()

    def __iter__(self) -> Iterator[TClient]:
        with self:
            return iter(list(self.clients))

    def __contains__(self, item: object) -> bool:
        with self:
            return item in self.clients


class PooledConnection(SyncStoppable, Generic[TParams]):
    """One physical connection shared by all clients with matching params.

    Transports (MQTT, WebSocket, ...) subclass this and translate their native
    connect / message / disconnect callbacks into the ``handle_*`` transitions,
    which fan the brand's connected/disconnected events out to every client on
    this connection. Subclasses must implement :attr:`connected`.
    """

    logger: logging.Logger = logging.getLogger("connection")

    def __init__(
        self,
        bucket: ClientBucket,
        params: TParams,
        *,
        connected_event: Hashable,
        disconnected_event: Hashable,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)

        self.bucket = bucket
        self.params = params
        self.connected_event = connected_event
        self.disconnected_event = disconnected_event
        self.failed_attempts = ConnectionAttemptsBoundedInterval.create_variable()

    @property
    @abstractmethod
    def connected(self) -> bool:
        """Whether the underlying transport currently holds a live connection."""
        raise NotImplementedError

    def emit_sync_all(self, event: Hashable, *args: object, **kwargs: object) -> None:
        """Emit ``event`` to every client sharing this connection."""
        for client in self.bucket.get_from_params(self.params):
            client.event_bus_worker.emit_sync(event, *args, **kwargs)

    # -- transitions (called by transport callbacks) ------------------------

    def handle_connected(self) -> None:
        self.logger.info("Connected to %s", self.params)
        self.failed_attempts.reset()
        self.emit_sync_all(self.connected_event)

    def handle_connect_failed(
        self, reason: str = "Connection failed enough times"
    ) -> None:
        self.failed_attempts.increment()
        self.logger.info(
            "Connect failed to %s (%s attempts)",
            self.params,
            self.failed_attempts.value,
        )
        if self.failed_attempts.is_at_bound():
            self.emit_sync_all(self.disconnected_event, reason)

    def handle_disconnected(
        self, transient: bool = False, reason: str = "Disconnected"
    ) -> None:
        # Transient drops (keepalive/conn-lost) are tolerated until they pile up;
        # the transport reconnects on its own in the meantime.
        if transient:
            self.failed_attempts.increment()
            self.logger.warning(
                "Transient disconnect from %s (%s attempts)",
                self.params,
                self.failed_attempts.value,
            )
            if not self.failed_attempts.is_at_bound():
                return

        self.logger.info("Disconnected from %s", self.params)
        self.emit_sync_all(self.disconnected_event, reason)


class ConnectionManager(SyncStoppable, Generic[TClient, TParams]):
    """Owns the client bucket and the pool of physical connections.

    Generic over transport: subclasses implement :meth:`_create_connection`
    (build a :class:`PooledConnection` for some params) and the keepalive hooks
    :meth:`_refresh_subscription` / :meth:`_send_keepalive`.
    """

    logger: logging.Logger = logging.getLogger("connection_manager")

    #: Brand event types emitted to clients. Set by the concrete subclass.
    connected_event: Hashable
    disconnected_event: Hashable
    message_event: Hashable
    #: How a client's config becomes hashable connection params.
    params_factory: Callable[[object], TParams]
    wildcard_topics: bool = False
    keepalive_timeout_ms: int = KEEPALIVE_TIMEOUT_MS

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.bucket: ClientBucket[TClient, TParams] = ClientBucket(
            self.params_factory, wildcard_topics=self.wildcard_topics
        )
        self.connections: Dict[TParams, PooledConnection] = {}
        self._lock = threading.Lock()

    # -- transport hooks (implemented by transport/brand subclasses) --------

    @abstractmethod
    def _create_connection(self, params: TParams) -> PooledConnection:
        """Build a new physical connection for ``params``."""
        raise NotImplementedError

    def _refresh_subscription(
        self, client: TClient, connection: PooledConnection
    ) -> None:
        """Re-assert the client's subscription during keepalive (transport hook)."""

    def _send_keepalive(self, client: TClient, connection: PooledConnection) -> None:
        """Ask the printer to send us a message (brand keepalive command)."""

    def _unsubscribe(self, client: TClient, connection: PooledConnection) -> None:
        """Drop the client's subscription on removal (transport hook)."""

    # -- pool ---------------------------------------------------------------

    def create_or_get_connection(self, params: TParams) -> PooledConnection:
        """Atomically reuse or create the single connection for ``params``."""
        with self._lock:
            if params in self.connections:
                return self.connections[params]

            connection = self._create_connection(params)
            self.connections[params] = connection
            return connection

    def rebuild_connections(self, *clients: TClient) -> None:
        """Reconcile the connection pool with the current set of clients."""
        with self._lock:
            connection_items = list(self.connections.items())

        # Tear down connections no client needs anymore.
        for params, connection in connection_items:
            if not self.bucket.get_from_params(params):
                with self._lock:
                    connection = self.connections.pop(params, None)
                if connection is not None:
                    connection.stop()

        # Make sure every (specified) client has its connection.
        for client in list(clients or self.bucket):
            params = self.bucket.get_params(client)
            if params:
                self.create_or_get_connection(params)

    # -- client lifecycle ---------------------------------------------------

    def add_client(self, client: TClient) -> None:
        if client in self.bucket:
            raise ValueError("Client already registered")

        self.bucket.add(client)

        params = self.bucket.get_params(client)
        if not params:
            raise ValueError("Failed to create connection params")

        connection = self.create_or_get_connection(params)

        # If the shared connection is already up, let this client know now.
        if connection.connected:
            client.event_bus_worker.emit_sync(self.connected_event)

    def refresh_client(self, client: TClient) -> None:
        """Recompute a client's params (config changed) and reconcile the pool."""
        if client not in self.bucket:
            return

        self.bucket.rebuild(client)
        self.rebuild_connections(client)

    def remove_client(self, client: TClient) -> None:
        if client not in self.bucket:
            return

        connection = self.get_connection_from_client(client)
        if connection is not None:
            self._unsubscribe(client, connection)

        self.bucket.remove(client)
        self.rebuild_connections()

    def get_connection_from_client(self, client: TClient) -> Optional[PooledConnection]:
        params = self.bucket.get_params(client)
        with self._lock:
            return self.connections.get(params)

    # -- keepalive ----------------------------------------------------------

    def keepalive_check(self) -> None:
        for client in list(self.bucket):
            # Not connected -> nothing to keep alive; the transport reconnects.
            if not client.connected:
                continue

            # Heard from it recently -> no poke needed.
            if now_ms() - client.last_message_at < self.keepalive_timeout_ms:
                continue

            # Increments and returns True once the bound is reached.
            if client.keepalive_attempts.guard_until_bound():
                client.event_bus_worker.emit_sync(
                    self.disconnected_event, "Keepalive failed enough times"
                )
                continue

            connection = self.get_connection_from_client(client)
            if connection is None:
                client.logger.debug("No active connection during keepalive check.")
                continue

            self._refresh_subscription(client, connection)
            self._send_keepalive(client, connection)

            # Give the printer a chance to answer before we poke again.
            client.last_message_at = now_ms() - (self.keepalive_timeout_ms // 2)

    # -- shutdown -----------------------------------------------------------

    def stop(self) -> None:
        self.logger.info("Stopping connection manager")
        super().stop()
        with self._lock:
            connections = list(self.connections.values())
            self.connections.clear()
        for connection in connections:
            connection.stop()
