"""Threaded connection pooling + transports for printer integrations.

The brand-agnostic pool lifecycle -- :class:`ClientBucket`,
:class:`PooledConnection`, :class:`ConnectionManager` -- lets many clients (one
per printer) share a smaller set of physical connections keyed by hashable
params. Transports subclass the pool: :mod:`.mqtt` ships the paho-mqtt binding.
The shared :class:`ConnectionState` vocabulary and the :class:`Watchdog` round
it out.

The MQTT binding is imported lazily (PEP 562 ``__getattr__``) so plain
``import simplyprint_ws_client.contrib.connection`` never drags ``paho`` into the
import graph; the pool / state / watchdog leaves carry no third-party imports.
"""

from typing import TYPE_CHECKING

from simplyprint_ws_client.contrib.connection.pool import (
    KEEPALIVE_TIMEOUT_MS,
    ClientBucket,
    ConnectionManager,
    EventBusWorker,
    PoolClient,
    PooledConnection,
    now_ms,
)
from simplyprint_ws_client.contrib.connection.state import ConnectionState
from simplyprint_ws_client.contrib.connection.watchdog import Watchdog

if TYPE_CHECKING:  # eager names for IDEs / type checkers
    from simplyprint_ws_client.contrib.connection.mqtt import (
        MqttConnection,
        MqttConnectionManager,
        MqttConnectionParams,
    )

__all__ = [
    "ClientBucket",
    "ConnectionManager",
    "ConnectionState",
    "EventBusWorker",
    "KEEPALIVE_TIMEOUT_MS",
    "MqttConnection",
    "MqttConnectionManager",
    "MqttConnectionParams",
    "PoolClient",
    "PooledConnection",
    "Watchdog",
    "now_ms",
]


def __getattr__(name: str):
    if name in ("MqttConnection", "MqttConnectionManager", "MqttConnectionParams"):
        from simplyprint_ws_client.contrib.connection import mqtt

        return getattr(mqtt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
