"""Reusable library components an integration composes.

``contrib`` is the home for the library's building blocks -- the headline
:class:`~simplyprint_ws_client.contrib.printer_client.PrinterClient`, the
:mod:`~simplyprint_ws_client.contrib.connection` subsystem (the transport
contract, the pooled printer connections, and the per-protocol ``mqtt`` /
``websocket`` families), and the
:mod:`~simplyprint_ws_client.contrib.logging` facility. Import the submodule you
need (e.g. ``from simplyprint_ws_client.contrib.connection.websocket import
WebsocketsImpl``); the common names are also re-exported from the package
root ``simplyprint_ws_client``.

This package ``__init__`` is intentionally import-free so that low-level
submodules (``connection``) can be imported by ``core`` without dragging in the
high-level ones (``printer_client``, which imports ``core``).
"""
