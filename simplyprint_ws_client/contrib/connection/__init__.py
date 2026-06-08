"""Pool many printer clients onto a smaller set of shared connections.

This is the one home for "give me a live link to a printer, fan its messages to
every client that shares it, and keep it alive." It is brand-agnostic throughout:
the N printers on one host share a handful of real sockets rather than one apiece.

The pieces, all brand-free:

* :mod:`.events` -- the :class:`TransportEvent` vocabulary every wire speaks.
* :mod:`.transport` -- the *contracts*: :class:`Transport` / :class:`AsyncTransport`
  (a supervised link), :class:`Lease` / :class:`AsyncLease` (a per-client
  lease), and :class:`Pool` / :class:`AsyncPool` (sharing by endpoint).
* :mod:`.pool` -- the machinery behind those contracts: ref-counted pooling, the
  event fan-out (one bus listener per transport; each lease self-filters by route),
  the per-client leases, and :class:`DeliveryConfig`.
* :mod:`.loop` -- :class:`LoopBridge`, the one sanctioned cross-thread hop onto the
  pool loop.
* :mod:`.manager` -- :class:`PooledConnectionManager`, the lifecycle a brand subclasses
  (event types + a ``params_factory``): register / keepalive / reconnect / reconcile.
* the wire families -- :mod:`.mqtt` (paho + aiomqtt) and :mod:`.websocket`
  (websocket-client + asyncio ``websockets`` / aiohttp), each a small package with
  ``common`` / ``sync`` / ``aio`` and, for WebSocket, the raw socket impls.
* :class:`ConnectionState` -- the shared "is this printer reachable" vocabulary.

Off the wire, every cross-thread hop comes home on the pool loop through one
:class:`~simplyprint_ws_client.shared.asyncio.courier.Courier` -- no per-event
future, no dedicated dispatch thread. The blocking wire libraries are imported
lazily by the families, so a plain ``import simplyprint_ws_client.contrib.connection``
drags none of them; only the contract / manager / state leaves load eagerly.
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
    "now_ms",
]
