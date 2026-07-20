"""SimplyPrintConnection routes each server WS log line to its printer.

The headline requirement: backend WebSocket messages about a printer land in that
printer's established ``<uid>/ws.log``, not a device driver's
``client-ws.log`` -- while genuinely global messages (MULTI-mode handshake etc.)
stay on the system ``ws`` logger.
"""

from types import SimpleNamespace

from yarl import URL

from simplyprint_ws_client import PrinterConfig
from simplyprint_ws_client.common.logging.naming import printer_logger_name
from simplyprint_ws_client.core.protocol.connection import (
    SimplyPrintConnection,
    ConnectionHint,
    ConnectionMode,
)

_WS = URL("wss://ws.example")


def test_default_connection_hints_do_not_share_config_state():
    first = ConnectionHint(_WS)
    second = ConnectionHint(_WS)

    first.config.id = 42

    assert second.config.id == 0


def test_multi_mode_routes_message_to_its_printer():
    conn = SimplyPrintConnection(
        _WS, hint=ConnectionHint(_WS, mode=ConnectionMode.MULTI)
    )
    msg = SimpleNamespace(for_client="printer-9")
    assert conn.protocol._message_logger(msg).name == printer_logger_name(
        "printer-9", "ws"
    )
    conn.stop()


def test_multi_mode_global_message_stays_global():
    conn = SimplyPrintConnection(
        _WS, hint=ConnectionHint(_WS, mode=ConnectionMode.MULTI)
    )
    msg = SimpleNamespace(for_client=None)
    # No printer association -> the connection's (system-scope) ws logger.
    assert conn.protocol._message_logger(msg) is conn.logger
    conn.stop()


def test_single_mode_routes_to_the_connection_printer():
    config = PrinterConfig.get_new()
    config.id = 7
    conn = SimplyPrintConnection(
        _WS,
        hint=ConnectionHint(_WS, mode=ConnectionMode.SINGLE, config=config),
    )
    # SINGLE: one printer; every message belongs to it, regardless of for_client.
    msg = SimpleNamespace(for_client=None)
    assert conn.protocol._message_logger(msg).name == printer_logger_name(
        config.unique_id, "ws"
    )
    conn.stop()
