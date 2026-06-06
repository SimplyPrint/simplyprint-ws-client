"""Pool many printer clients onto a smaller set of shared connections.

This is the one home for "give me a live link to a printer, fan its messages to
every client that shares it, and keep it alive." It is brand-agnostic throughout:
the N printers on one host share a handful of real sockets rather than one apiece.

Layering (bottom -> top), all leaves brand-free:

* :mod:`.transport` -- the wire *contract*: the :class:`TransportEvent` surface and
  the :class:`Transport` / :class:`AsyncTransport` / :class:`Pool` / :class:`AsyncPool`
  abstractions, plus :class:`Connection` (the per-client lease) and the
  :class:`TransportRouter` that fans a shared transport's events to the right leases.
* :mod:`.manager` -- :class:`PooledConnectionManager`, the brand-agnostic lifecycle:
  it owns a :class:`Pool`, leases one :class:`Connection` per client, wires the
  lease's events onto the client's bus, and runs keepalive / reconnect / the
  registration-reconcile sweep. Brands subclass it (a few event types + a
  ``params_factory``), they do not reimplement it.
* the **wire families** that back a manager: :mod:`.sync_mqtt` (paho-mqtt, many
  printers sharing one broker), :mod:`.async_mqtt` (the asyncio MQTT family), and
  :mod:`.sync_ws` (websocket-client, one supervised socket per host), the last
  wrapping the blocking :class:`.threaded_ws.ThreadedWebSocketTransport` wire.
* :class:`ConnectionState` -- the shared "is this printer reachable" vocabulary.
* :mod:`.wire` -- the raw async WebSocket wire the SimplyPrint backend socket rides.

Off the wire, every cross-thread hop comes home on the consumer loop through one
:class:`~simplyprint_ws_client.shared.asyncio.courier.Courier` -- no per-event
future, no dedicated dispatch thread.

The blocking wire libraries (paho-mqtt, websocket-client) are imported lazily
(PEP 562 ``__getattr__`` here, plus lazy wire factories in the families) so a plain
``import simplyprint_ws_client.contrib.connection`` drags neither; only the
contract / manager / state leaves load eagerly, and they carry no third-party
imports.
"""

from simplyprint_ws_client.contrib.connection.manager import (
    KEEPALIVE_TIMEOUT_MS,
    ConnectionAttemptsBoundedInterval,
    PoolClient,
    PooledConnectionManager,
    now_ms,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState

__all__ = [
    "ConnectionAttemptsBoundedInterval",
    "ConnectionState",
    "KEEPALIVE_TIMEOUT_MS",
    "PoolClient",
    "PooledConnectionManager",
    "ThreadedWebSocketTransport",
    "now_ms",
]


def __getattr__(name: str):
    # The blocking websocket-client wire stays lazy: importing the package must
    # not drag websocket-client. Reached as
    # ``contrib.connection.ThreadedWebSocketTransport``.
    if name == "ThreadedWebSocketTransport":
        from simplyprint_ws_client.contrib.connection import threaded_ws

        return threaded_ws.ThreadedWebSocketTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
