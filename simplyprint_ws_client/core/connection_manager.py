"""CloudConnection strategy: allocate / deallocate a client onto a backend socket.

This is the SimplyPrint **protocol** multiplexing layer, and it lives in ``core``
on purpose -- it is *not* the brand-free connection pool
(:mod:`...device.connection.pool`) and must not be folded into it. The two share
a vague silhouette ("fan one socket to N consumers") but differ in the essential
logic, which is the test for whether things are the same (they are not):

* The brand-free pool routes by one rule -- ``topic_of(payload) == lease.route`` --
  and is forbidden from knowing any message type. The backend's MULTI routing is
  protocol-aware: a global ``ConnectedMsg`` is *broadcast* to every client as
  :class:`CloudConnectionEstablishedEvent`; ``MultiPrinterAdded/RemovedMsg`` route by a
  *different* field (``msg.data.unique_id``); ordinary messages route by
  ``for_client``; the raw transport ``CloudConnectionEstablishedEvent`` is *suppressed*
  in favour of ``ConnectedMsg``; ``CloudConnectionLostEvent`` is broadcast. Broadcast,
  multi-field routing, and event translation have no expression in the pool's
  single-key model -- and teaching it ``ConnectedMsg``/``for_client`` would breach
  the no-brand-leak rule. So this stays here.

* **Reconnection is not duplicated here.** Each backend :class:`CloudConnection` rides
  the shared :class:`~...common.wire.reconnect.Reconnecting` engine (connect /
  backoff / liveness / single version bump). In MULTI mode the one shared
  ``CloudConnection`` gets that for free; :class:`ClientView` only fans its events to the
  right client(s). There is no PAUSE state: a client's "pause" is releasing its hold
  (``deallocate``), and when the last client leaves a view its ``CloudConnection`` is
  disconnected (``engine.stop()``) -- the refcount/lease lifecycle, protocol-side.
"""

__all__ = ["ClientConnectionManager", "ClientList", "ClientView"]

import asyncio
import functools
import logging
from datetime import timedelta
from typing import Union, Dict, Mapping
from typing import final, Optional, Set, cast, Iterable, MutableSet, Hashable

from simplyprint_ws_client.cloud.client import Client
from simplyprint_ws_client.cloud.client import ClientState
from simplyprint_ws_client.cloud.config import PrinterConfig
from simplyprint_ws_client.cloud.protocol.connection import ConnectionHint
from simplyprint_ws_client.cloud.protocol.connection import (
    ConnectionMode,
    CloudConnection,
)
from simplyprint_ws_client.cloud.protocol.events import (
    CloudConnectionIncomingEvent,
    CloudConnectionEstablishedEvent,
)
from simplyprint_ws_client.cloud.protocol.events import (
    CloudConnectionOutgoingEvent,
    CloudConnectionLostEvent,
    CloudConnectionSuspectEvent,
)
from simplyprint_ws_client.cloud.protocol.messages import ClientMsg
from simplyprint_ws_client.cloud.protocol.messages import (
    MultiPrinterAddedMsg,
    MultiPrinterRemovedMsg,
    Msg,
    ConnectedMsg,
)
from simplyprint_ws_client.const import APP_DIRS
from simplyprint_ws_client.common.events.emitter import Emitter, TEvent
from simplyprint_ws_client.common.events.event_bus_listeners import ListenerUniqueness
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.shared.debug.connectivity import ConnectivityReport
from simplyprint_ws_client.common.utils.stoppable import AsyncStoppable

TUniqueId = Union[str, int]


class ClientList(Mapping[Union[TUniqueId, Client, PrinterConfig], Client]):
    clients: Dict[TUniqueId, Client]

    def __init__(self):
        self.clients = {}

    def add(self, client: Client):
        self.clients[client.unique_id] = client

    def remove(self, client: Client):
        del self.clients[client.unique_id]

    def __contains__(self, key):
        if isinstance(key, (Client, PrinterConfig)):
            key = key.unique_id

        return key in self.clients

    def __getitem__(self, key, /):
        if isinstance(key, (Client, PrinterConfig)):
            key = key.unique_id

        return self.clients[key]

    def __len__(self):
        return len(self.clients)

    def __iter__(self):
        return iter(self.clients)


class ClientView(Emitter, MutableSet[Client], Hashable):
    """The protocol-aware fan-out over the clients sharing one backend
    :class:`CloudConnection` -- the multiplexing the brand-free pool deliberately does
    not do (see the module docstring). :meth:`emit` is the routing brain: it
    broadcasts the global handshake, routes per-printer messages by ``for_client``
    (or ``data.unique_id`` for add/remove), suppresses the raw transport
    ``established``, and broadcasts ``lost`` -- delivering each onto the target
    client's own event bus. Reconnection is the ``CloudConnection``'s engine's job, not
    this object's; a view only decides *who hears what*."""

    mode: ConnectionMode
    connection: CloudConnection
    client_list: ClientList
    clients: Set[TUniqueId]
    logger: logging.Logger

    def __init__(
        self,
        mode: ConnectionMode,
        connection: CloudConnection,
        client_list: ClientList,
        logger=logging.getLogger(__name__),
    ):
        self.mode = mode
        self.connection = connection
        self.client_list = client_list
        self.clients = set()
        self.logger = logger

    async def _emit_all(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        # Iterate over a stable snapshot so mutations to view/client maps during awaits
        # do not abort fanout partway through.
        client_ids = tuple(self.clients)

        for client_id in client_ids:
            client = self.client_list.clients.get(client_id)

            if client is None:
                continue

            try:
                await client.event_bus.emit(event, *args, **kwargs)
            except Exception as e:
                self.logger.error(
                    "Error when handling event %s for client %s:",
                    event,
                    client.unique_id,
                    exc_info=e,
                )

    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        # Only handle connection events when we have at least one client.
        if len(self) == 0:
            return

        is_multi_mode = self.mode == ConnectionMode.MULTI

        # Custom handling for incoming messages
        # when we have multiple clients.
        if is_multi_mode and event == CloudConnectionIncomingEvent:
            if len(args) != 2:
                return

            msg, v = args

            if not isinstance(msg, Msg):
                return

            # For multi-printer connections the `connected` message is the `established` message.
            if isinstance(msg, ConnectedMsg) and msg.for_client is None:
                self.logger.debug(
                    "Converted base ConnectedMsg to CloudConnectionEstablishedEvent with v: %d.",
                    v,
                )
                await self._emit_all(CloudConnectionEstablishedEvent(v))
                return

            # We can get a routing hint from the message directly.
            client_id = msg.for_client

            # Some messages are targeting the top level connection,
            # but we can further route them to the correct client.
            if client_id is None and isinstance(
                msg, (MultiPrinterAddedMsg, MultiPrinterRemovedMsg)
            ):
                client_id = msg.data.unique_id

            if client_id not in self.clients:
                return

            try:
                await self.client_list[client_id].event_bus.emit(event, *args, **kwargs)
            except Exception as e:
                self.logger.error(
                    "Error when handling event %s for client %s:",
                    event,
                    client_id,
                    exc_info=e,
                )

            return

        if is_multi_mode and isinstance(event, CloudConnectionEstablishedEvent):
            self.logger.debug(
                "Dropped CloudConnectionEstablishedEvent for multi-mode connection with v: %d in favor of ConnectedMsg",
                event.v,
            )
            return

        # Default handling for all other events.
        await self._emit_all(event, *args, **kwargs)

    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None:
        raise NotImplementedError("Use emit() instead.")

    def add(self, value: Client):
        self.clients.add(value.unique_id)

    def discard(self, value: Client):
        self.clients.discard(value.unique_id)

    def __contains__(self, x: Client):
        return x.unique_id in self.clients

    def __len__(self):
        return len(self.clients)

    def __iter__(self):
        for client in self.clients:
            yield self.client_list[client]

    def __hash__(self):
        return hash(id(self))


@final
class ClientConnectionManager(
    AsyncStoppable, EventLoopProvider[asyncio.AbstractEventLoop]
):
    """
    Attributes:
        mode: ConnectionMode
        client_list: Global list of clients (immutable from this context)
        client_views: Mapping of client unique_id to client view
        views: Set of all client views
        logger: Logger instance
    """

    mode: ConnectionMode
    client_list: ClientList
    max_clients_per_connection: Optional[int]
    client_views: Dict[TUniqueId, ClientView]
    views: Set[ClientView]
    logger: logging.Logger
    _next_connection_id: int

    def __init__(
        self,
        mode: ConnectionMode,
        client_list: ClientList,
        max_clients_per_connection: Optional[int] = None,
        logger: logging.Logger = logging.getLogger("ws_manager"),
        **kwargs,
    ):
        AsyncStoppable.__init__(self, **kwargs)
        EventLoopProvider.__init__(self, **kwargs)

        if max_clients_per_connection is not None and max_clients_per_connection < 1:
            raise ValueError("max_clients_per_connection must be a positive integer")

        self.mode = mode
        self.client_list = client_list
        self.max_clients_per_connection = max_clients_per_connection
        self.client_views = {}
        self.views = set()
        self.logger = logger
        self._next_connection_id = 0

    def _suspect_connection(self, connection: CloudConnection, _):
        """Called when a connection suspects its ability to connect is compromised."""
        self.logger.info(
            "CloudConnection %s suspects it is unable to connect. Running connectivity test suite and generating log file",
            connection.url,
        )

        path = APP_DIRS.user_log_path / "connectivity_reports"

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        previous_reports = ConnectivityReport.read_previous_reports(path)

        # TODO Make this configurable, and be able to hook into real-time data (network events etc.)
        if previous_reports and previous_reports[
            0
        ].timestamp - ConnectivityReport.utc_now() < timedelta(minutes=20):
            self.logger.info(
                "Last connectivity test was run less than 20 minute ago, skipping."
            )
            return

        # Perform various checks, do we have internet, is the server down, etc.
        report = ConnectivityReport.generate_default()

        path = report.store_in_path(path)

        self.logger.info("Connectivity test suite complete. Report saved to %s", path)

    def _allocate_new_connection(self) -> ClientView:
        if self.is_stopped():
            raise RuntimeError("Cannot allocate connections when stopped.")

        existing = self._select_existing_view()
        if existing is not None:
            return existing

        # Creating a new connection / client view pair.
        loggerName = "ws"
        connection_id = self._next_connection_id
        self._next_connection_id += 1

        # Indicate what connection instance this is.
        # In multimode all messages are annotated with an id.
        if self.mode != ConnectionMode.MULTI:
            loggerName = f"{loggerName}[{connection_id}]"

        connection = CloudConnection(
            provider=self, logger=logging.getLogger(loggerName)
        )
        client_view = ClientView(self.mode, connection, self.client_list)
        self.views.add(client_view)

        # Registering the connection with the client view.
        connection.event_bus.on(
            CloudConnectionIncomingEvent,
            functools.partial(client_view.emit, CloudConnectionIncomingEvent),
            unique=ListenerUniqueness.EXCLUSIVE_WITH_ERROR,
        )

        connection.event_bus.on(
            CloudConnectionEstablishedEvent,
            client_view.emit,
            unique=ListenerUniqueness.EXCLUSIVE_WITH_ERROR,
        )

        connection.event_bus.on(
            CloudConnectionLostEvent,
            client_view.emit,
            unique=ListenerUniqueness.EXCLUSIVE_WITH_ERROR,
        )

        connection.event_bus.on(
            CloudConnectionSuspectEvent,
            functools.partial(self._suspect_connection, connection),
            unique=ListenerUniqueness.EXCLUSIVE_WITH_ERROR,
        )

        return client_view

    def _select_existing_view(self) -> Optional[ClientView]:
        if self.mode != ConnectionMode.MULTI or not self.views:
            return None

        if self.max_clients_per_connection is None:
            candidates = tuple(self.views)
        else:
            candidates = tuple(
                view
                for view in self.views
                if len(view) < self.max_clients_per_connection
            )

        if not candidates:
            return None

        return min(candidates, key=lambda view: (len(view), id(view)))

    def _derive_connection_hint(self, client: Client) -> ConnectionHint:
        # Use up-to-date credentials to connect.
        # p/id/token else mp/0/0
        is_single = self.mode == ConnectionMode.SINGLE

        return ConnectionHint(
            mode=self.mode,
            config=client.config if is_single else PrinterConfig.get_blank(),
        )

    @property
    def connections(self) -> Iterable[CloudConnection]:
        return map(lambda x: cast(ClientView, x).connection, list(self.views))

    def get_connection_for_client(self, client: Client) -> Optional[CloudConnection]:
        view = self.client_views.get(client.unique_id)

        if not view:
            return None

        return view.connection

    def is_allocated(self, client: Client) -> bool:
        return client.unique_id in self.client_views

    async def allocate(self, client: Client) -> None:
        """Allocate a connection and instrument the client with it."""
        if self.is_allocated(client):
            return

        view = self._allocate_new_connection()
        connection = view.connection

        if connection.connected:
            client.v = connection.v
            client.state = ClientState.NOT_CONNECTED

        self.client_views[client.unique_id] = view
        view.add(client)

        if self.mode == ConnectionMode.MULTI:
            # Bind unique_id to the handler.
            def transform_message_with_unique_id(
                message: ClientMsg, *args, unique_id=client.unique_id
            ):
                message.for_client = unique_id
                return (message,) + args

            client.event_bus.on(
                CloudConnectionOutgoingEvent,
                transform_message_with_unique_id,
                priority=10,
            )

        client.event_bus.on(
            CloudConnectionOutgoingEvent,
            functools.partial(connection.event_bus.emit, CloudConnectionOutgoingEvent),
        )

        await connection.connect(hint=self._derive_connection_hint(client))

    async def deallocate(self, client: Client):
        """Deallocate a connection and remove it from the client."""
        if client.unique_id not in self.client_views:
            return

        client_view = self.client_views.pop(client.unique_id)
        client_view.discard(client)
        client.event_bus.clear(CloudConnectionOutgoingEvent)

        # Tell removed the client it has lost its connection, since it no longer receives messages.
        # This must be awaited (not emit_task) to prevent a race condition where the client
        # is re-allocated before the CloudConnectionLostEvent handler runs, causing the handler
        # to overwrite the state set by allocate() and permanently sticking the client in
        # CONNECTING state.
        await client.event_bus.emit(CloudConnectionLostEvent(client.v))

        # Disconnect the connection if no clients are left.
        if len(client_view) == 0:
            await client_view.connection.disconnect()
            self.views.discard(client_view)

    def stop(self):
        super().stop()

        for connection in self.connections:
            connection.stop()
