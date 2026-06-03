"""Connect to a printer, fan its events, and keep the link alive.

This is the one home for "give me a connection to X and tell me what it says".
It owns three layers, brand-agnostic throughout:

* the **pool** -- :class:`ClientBucket`, :class:`Connection`,
  :class:`ConnectionManager` -- lets many clients (one per printer) share a
  smaller set of physical connections keyed by hashable params. Wire bindings
  subclass the pool: :mod:`.mqtt` ships the paho-mqtt binding,
  :mod:`.threaded_ws` the websocket-client binding.
* the **transport** -- the async :class:`WebSocketTransport` ABC
  (:mod:`.transport`) the SimplyPrint backend ``Connection`` drives, with the
  two shipped wire leaves under :mod:`.transports` (``websockets`` / aiohttp).
* the shared **vocabulary** -- :class:`ConnectionState`, the
  :class:`TransportError` / :class:`TransportClosed` exceptions, the WS close
  codes, and the :class:`Watchdog`.

Every wire library (paho, websocket-client, websockets, aiohttp) is imported
lazily (PEP 562 ``__getattr__``) so plain ``import
simplyprint_ws_client.contrib.connection`` drags none of them; only the pool /
state / transport-ABC / watchdog leaves load eagerly, and they carry no
third-party imports.
"""

from typing import TYPE_CHECKING

from simplyprint_ws_client.contrib.connection.pool import (
    KEEPALIVE_TIMEOUT_MS,
    ClientBucket,
    Connection,
    ConnectionManager,
    EventBusWorker,
    PoolClient,
    now_ms,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.transport import (
    WS_CLOSE_OK,
    WS_CLOSE_PROTOCOL_ERROR,
    TransportClosed,
    TransportError,
    TransportFactory,
    WebSocketTransport,
)
from simplyprint_ws_client.contrib.connection.watchdog import Watchdog

if TYPE_CHECKING:  # eager names for IDEs / type checkers
    from simplyprint_ws_client.contrib.connection.mqtt import (
        MqttConnection,
        MqttConnectionManager,
        MqttConnectionParams,
    )
    from simplyprint_ws_client.contrib.connection.threaded_ws import (
        ThreadedWebSocketTransport,
    )
    from simplyprint_ws_client.contrib.connection.transports import (
        AiohttpWebSocketTransport,
        WebSocketsTransport,
    )

__all__ = [
    "AiohttpWebSocketTransport",
    "ClientBucket",
    "Connection",
    "ConnectionManager",
    "ConnectionState",
    "EventBusWorker",
    "KEEPALIVE_TIMEOUT_MS",
    "MqttConnection",
    "MqttConnectionManager",
    "MqttConnectionParams",
    "PoolClient",
    "ThreadedWebSocketTransport",
    "TransportClosed",
    "TransportError",
    "TransportFactory",
    "Watchdog",
    "WebSocketsTransport",
    "WebSocketTransport",
    "WS_CLOSE_OK",
    "WS_CLOSE_PROTOCOL_ERROR",
    "now_ms",
]


def __getattr__(name: str):
    if name in ("MqttConnection", "MqttConnectionManager", "MqttConnectionParams"):
        from simplyprint_ws_client.contrib.connection import mqtt

        return getattr(mqtt, name)
    if name == "ThreadedWebSocketTransport":
        from simplyprint_ws_client.contrib.connection.threaded_ws import (
            ThreadedWebSocketTransport,
        )

        return ThreadedWebSocketTransport
    if name in ("WebSocketsTransport", "AiohttpWebSocketTransport"):
        from simplyprint_ws_client.contrib.connection import transports

        return getattr(transports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
