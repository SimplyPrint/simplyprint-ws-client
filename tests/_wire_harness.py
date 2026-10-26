"""A hand-driven SimplyPrint wire for scheduler-level tests.

The real write path is ``Scheduler._schedule_client`` -> ``Client.send`` ->
the manager's view -> ``SimplyPrintConnection`` -> ``SimplyPrintProtocol.send``
-> the transport. These helpers keep every piece of that path real and replace
only the socket, which the test opens, drops and fails by hand. There is no
supervise task, so every step is deterministic: nothing happens between two
``await``s that the test did not ask for.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.manager import ClientList
from simplyprint_ws_client.core.protocol.connection import (
    ConnectionMode,
    SimplyPrintConnection,
)
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionEstablishedEvent,
    SimplyPrintConnectionIncomingEvent,
    SimplyPrintConnectionLostEvent,
)
from simplyprint_ws_client.core.protocol.messages import (
    ConnectedMsg,
    MultiPrinterAddedMsg,
)
from simplyprint_ws_client.core.protocol.models import ServerMsgType
from simplyprint_ws_client.core.scheduler import Scheduler
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.wire.state import ConnectionState

from tests._fakes import FakeTransport


class ManualSocket(FakeTransport):
    """A socket that opens when started and fails only when told to."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Raised by the next writes while set (a socket dying under a send).
        self.fail_writes: Optional[BaseException] = None

    def start(self) -> None:
        self.open_for_test()

    def drop(self) -> None:
        self._open = False
        self.live = False
        self.state = ConnectionState.DISCONNECTED

    async def write(self, data: object) -> None:
        if self.fail_writes is not None:
            raise self.fail_writes
        await super().write(data)


def new_client(unique_id: Optional[str] = None) -> Client:
    config = PrinterConfig.get_new()
    if unique_id is not None:
        config.unique_id = unique_id
    config.id = 1
    config.in_setup = False
    return Client(config, context=ClientContext())


class Wire:
    """A scheduler with real connections over :class:`ManualSocket`\\ s."""

    def __init__(self, mode: ConnectionMode, **settings) -> None:
        self.mode = mode
        self.sockets: List[ManualSocket] = []
        self._socket_of: Dict[int, ManualSocket] = {}
        self.scheduler = Scheduler(
            ClientList(),
            ClientSettings(mode=mode, **settings),
            transport_factory=self._factory,
        )
        self.scheduler.use_running_loop()

    def _factory(self, url, provider, logger) -> ManualSocket:
        socket = ManualSocket(url, provider, logger)
        self.sockets.append(socket)
        self._socket_of[id(provider)] = socket
        return socket

    # -- topology -----------------------------------------------------------

    def connection(self, client: Client) -> SimplyPrintConnection:
        connection = self.scheduler.manager.get_connection_for_client(client)
        assert connection is not None, "client is not allocated"
        return connection

    def socket(self, client: Client) -> ManualSocket:
        return self._socket_of[id(self.connection(client))]

    def frames(self, client: Client) -> List[str]:
        """Every frame written to the client's socket, in order."""
        return list(self.socket(client).sent)

    # -- driving ------------------------------------------------------------

    async def schedule(self, client: Client) -> None:
        """One scheduler pass over ``client`` (allocate, add, flush)."""
        if client.unique_id not in self.scheduler.client_list:
            self.scheduler.client_list.add(client)
        await self.scheduler._schedule_client(client)

    async def server(self, client: Client, msg) -> None:
        """Deliver one inbound server message on the client's connection."""
        connection = self.connection(client)
        await connection.event_bus.emit(
            SimplyPrintConnectionIncomingEvent, msg, connection.v
        )

    async def open(self, client: Client) -> None:
        """The server side of a (re)connect, up to the printer being added.

        SINGLE: the socket opens and the server sends ``connected``. MULTI: the
        hub's bare ``connected``, the scheduler's ``add_connection``, then the
        child's ``connected`` (the backend's ``PrinterSession::onOpen``) and the
        add response, in the backend's order.
        """
        await self.schedule(client)  # allocates on the first pass
        connection = self.connection(client)
        socket = self.socket(client)
        if not socket.connected:
            socket.open_for_test()

        if self.mode is ConnectionMode.SINGLE:
            await connection.event_bus.emit(
                SimplyPrintConnectionEstablishedEvent(connection.v)
            )
            await self.server(client, ConnectedMsg(type=ServerMsgType.CONNECTED))
            return

        await self.server(client, ConnectedMsg(type=ServerMsgType.CONNECTED))
        await self.schedule(client)  # writes add_connection
        await self.server(
            client,
            ConnectedMsg(type=ServerMsgType.CONNECTED, for_client=client.unique_id),
        )
        await self.server(
            client,
            MultiPrinterAddedMsg(
                type=ServerMsgType.ADD_CONNECTION,
                data={"status": True, "unique_id": client.unique_id, "pid": 1},
            ),
        )

    async def drop(self, client: Client) -> None:
        """The socket dies; the protocol reports it and bumps its version."""
        connection = self.connection(client)
        self.socket(client).drop()
        await connection.event_bus.emit(SimplyPrintConnectionLostEvent(connection.v))
        connection.v += 1


def with_for(frame: str, unique_id: str) -> str:
    """``frame`` as a hub writes it for ``unique_id``.

    ``Msg``'s serializer emits ``for`` last, so this is a pure string splice:
    nothing is re-encoded, and a byte difference anywhere else still shows.
    """
    assert frame.endswith("}")
    return frame[:-1] + ',"for":' + json.dumps(unique_id) + "}"
