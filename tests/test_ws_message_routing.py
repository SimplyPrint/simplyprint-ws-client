"""Connection routes each WS message's log line to the printer it belongs to.

The headline requirement: backend WebSocket messages about a printer land in that
printer's log file (``<uid>/ws.log``), not the global one -- while genuinely
global messages (MULTI-mode handshake etc.) stay on the system ``ws`` logger.
"""

from types import SimpleNamespace

from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.contrib.logging.naming import printer_logger_name
from simplyprint_ws_client.core.ws_protocol.connection import (
    Connection,
    ConnectionHint,
    ConnectionMode,
)


def test_multi_mode_routes_message_to_its_printer():
    conn = Connection(hint=ConnectionHint(mode=ConnectionMode.MULTI))
    msg = SimpleNamespace(for_client="printer-9")
    assert conn.protocol._message_logger(msg).name == printer_logger_name(
        "printer-9", "ws"
    )
    conn.stop()


def test_multi_mode_global_message_stays_global():
    conn = Connection(hint=ConnectionHint(mode=ConnectionMode.MULTI))
    msg = SimpleNamespace(for_client=None)
    # No printer association -> the connection's (system-scope) ws logger.
    assert conn.protocol._message_logger(msg) is conn.logger
    conn.stop()


def test_single_mode_routes_to_the_connection_printer():
    config = PrinterConfig.get_new()
    config.id = 7
    conn = Connection(hint=ConnectionHint(mode=ConnectionMode.SINGLE, config=config))
    # SINGLE: one printer; every message belongs to it, regardless of for_client.
    msg = SimpleNamespace(for_client=None)
    assert conn.protocol._message_logger(msg).name == printer_logger_name(
        config.unique_id, "ws"
    )
    conn.stop()
