"""Reusable library components an integration composes.

``contrib`` is the home for the library's building blocks -- the headline
:class:`~simplyprint_ws_client.contrib.printer_client.PrinterClient`, the
swappable WebSocket :mod:`~simplyprint_ws_client.contrib.transport`, and the
:mod:`~simplyprint_ws_client.contrib.logging` facility. Import the submodule you
need (e.g. ``from simplyprint_ws_client.contrib.transport import WebSocketsTransport``);
the common names are also re-exported from the package root
``simplyprint_ws_client``.

This package ``__init__`` is intentionally import-free so that low-level
submodules (``transport``) can be imported by ``core`` without dragging in the
high-level ones (``printer_client``, which imports ``core``).
"""
