"""TLS contexts shared by the wire families."""

from __future__ import annotations

import os
import ssl
import tempfile

from simplyprint_ws_client.wire.options import TlsClientAuth

__all__ = ["client_cert_ssl_context"]


def client_cert_ssl_context(auth: TlsClientAuth) -> ssl.SSLContext:
    """Build the mutual-TLS context used by certificate-authenticated printers."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=auth.ca_pem)
    directory = tempfile.mkdtemp()
    cert_path = os.path.join(directory, "cert.pem")
    key_path = os.path.join(directory, "key.pem")
    try:
        for path, data in ((cert_path, auth.cert_pem), (key_path, auth.key_pem)):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(descriptor, data.encode())
            finally:
                os.close(descriptor)
        context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    finally:
        for path in (cert_path, key_path):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(directory)
        except OSError:
            pass
    return context
