"""Explicit installation of a CPython SSL transport correctness fix.

CPython 3.12.8, 3.13.1, and 3.14 include the upstream fix for
``python/cpython#118950``.  Older Python versions supported by this package do
not.  Installing it when a WebSocket is opened keeps package import free of
stdlib mutation and places the exceptional private-API use at the wire boundary
that needs it.
"""

import sys

__all__ = ["install_ssl_transport_workaround"]


def install_ssl_transport_workaround() -> None:
    """Install CPython's SSL closing-state fix on affected Python versions."""
    if sys.version_info >= (3, 12, 8) and not (
        (3, 13) <= sys.version_info < (3, 13, 1)
    ):
        return

    from asyncio.sslproto import _SSLProtocolTransport, SSLProtocol

    # Exact backport of CPython PR 118960. Reassignment is intentionally
    # idempotent, so each independently composed transport can activate it.
    SSLProtocol._is_transport_closing = _ssl_protocol_is_transport_closing
    _SSLProtocolTransport.is_closing = _ssl_transport_is_closing


def _ssl_protocol_is_transport_closing(self) -> bool:
    return self._transport is not None and self._transport.is_closing()


def _ssl_transport_is_closing(self) -> bool:
    return self._closed or self._ssl_protocol._is_transport_closing()
