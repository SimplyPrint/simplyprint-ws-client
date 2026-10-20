"""Mutual-TLS client-certificate support: the context builder, and a real
handshake through the MQTT front door against a loopback broker that
requires a client certificate (as printers that issue one at pairing do)."""

from __future__ import annotations

import asyncio
import ssl
import subprocess
import tempfile
from pathlib import Path
from typing import List, NamedTuple

import pytest
import pytest_asyncio

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire import mqtt
from simplyprint_ws_client.wire.events import Disconnected
from simplyprint_ws_client.wire.options import ConnectionOptions, TlsClientAuth
from simplyprint_ws_client.wire.pools import PoolRegistry
from simplyprint_ws_client.wire.tls import client_cert_ssl_context
from tests import mqtt_broker
from tests.mqtt_broker import Broker

# Self-signed cert+key generated once (test-only material, not a secret).
# openssl req -x509 -newkey rsa:2048 -keyout k.pem -out c.pem -days 1 -nodes \
#             -subj "/CN=test-printer-serial"
_TEST_CERT_PEM = """\
-----BEGIN CERTIFICATE-----
MIIDHTCCAgWgAwIBAgIUMBgYsvAoE67EWOof+GvI0f/+LkIwDQYJKoZIhvcNAQEL
BQAwHjEcMBoGA1UEAwwTdGVzdC1wcmludGVyLXNlcmlhbDAeFw0yNjA3MDkxMDM4
MzNaFw0yNjA3MTAxMDM4MzNaMB4xHDAaBgNVBAMME3Rlc3QtcHJpbnRlci1zZXJp
YWwwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCe2j4JL4kYZnxcqaBG
5ndUgGWT8/Gzyh0w3xyokp7EqCD9zyTxh/Uculxipx21g4uJjH05mJYJtJwcrlm/
KhKmzJraP/tunHNrFKmqAE0tCfqTYfb0rZbhMcWWgYpETaCKXgSsq8fdigZm6AmY
HiaottnotmuKnKc8DKRMzh8QCPkn+HVy9RHzBi5oC81aRaezkSrfFDLetJlM5OlF
TJ2TaVK8IRrfkwDntlfPJElNe9qbNXI/bDJTr4oBWPnWae36+V1OQx2LK1bn2aI1
pEdK5ipCwEnIrwGY5QZx0gWVf4IsLjRujthaHUztUC+cOokd55ouHJNjKNhVwsnh
l3iNAgMBAAGjUzBRMB0GA1UdDgQWBBQxkxlr+8OMJep4UQrMZs5KoFLLqzAfBgNV
HSMEGDAWgBQxkxlr+8OMJep4UQrMZs5KoFLLqzAPBgNVHRMBAf8EBTADAQH/MA0G
CSqGSIb3DQEBCwUAA4IBAQAlKd/iZMsUK+BAO1H+R50dmN0K+925M+zGPlgUsB27
5BSnv7vTSE3IX8dXXLqdrGuNazBBs0w9+dW+Po2Ez5AJD2TIk0pxGt3rfhvd60Pp
9cC4dlOMsODbolkAzZg6W1noYBQ1yDw9FNWrjUtLKx0puB8NBC4Ga7y5W92VHAUN
5fJVvCPFun++wEskJyW9vYpUC3/zY+kHMqPORfW3rVXP0LhxZ8J6PAhF7H6cfNDz
+AgPYEuKbH5bbsclTZ6OKbgY2slPNyz/hgAeh4/bKAshEOOkV6Z6bpbNLp1pVfbm
KDhogH0FES04SCfKai3szPYDJJUJzJcl/sYlIoSnpMYx
-----END CERTIFICATE-----
"""

_TEST_KEY_PEM = """\
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCe2j4JL4kYZnxc
qaBG5ndUgGWT8/Gzyh0w3xyokp7EqCD9zyTxh/Uculxipx21g4uJjH05mJYJtJwc
rlm/KhKmzJraP/tunHNrFKmqAE0tCfqTYfb0rZbhMcWWgYpETaCKXgSsq8fdigZm
6AmYHiaottnotmuKnKc8DKRMzh8QCPkn+HVy9RHzBi5oC81aRaezkSrfFDLetJlM
5OlFTJ2TaVK8IRrfkwDntlfPJElNe9qbNXI/bDJTr4oBWPnWae36+V1OQx2LK1bn
2aI1pEdK5ipCwEnIrwGY5QZx0gWVf4IsLjRujthaHUztUC+cOokd55ouHJNjKNhV
wsnhl3iNAgMBAAECggEAEFdCUxmTuxyCey8WhhSb0+fStdyTUZEKCFv7NmVsIFjB
7roKAoVKM/wBGEOOrScXXz1CKH7yIuofC7LtF9rRHl16fyxecKt2vV4xHIQkFuz+
3NMStMAiqY8Vf1JbZy6rxBrAf666xFG4L4p7v4KNLC/bIghEmcwG8V4aN1hrc5pU
TzkR1RC/cqttP7lT9flHF1izCOQm9aaZMkguK97j7FGfzBeyFmW8XHzPndvJYhdg
nH550McDMDHdot9Wa5EuMzi7ZsH1v7U4TSD9/wwEVNJa0DhBdB4gZsxZ8922azR0
SByGvZpsVrtNWuCZwUHVCUVLExaO2rU2oebuR5zndQKBgQDghqcGFqCnETwsB3jM
gOAUp6xy983XXqwFiPC06OzamuJZVxsQlV/sYjYuCfhf6xyA5HNUU0igxKjsEJqp
fBfPDU05GyedrCf7+EyxwRQ3rrdiSL49y4EXCwueKfaJQndaWiPRlpXVbXVSkIx7
HAWhm/jrBXAvE1vTPGFzDoOBrwKBgQC1HtRDCpCfdV5sOh+0XErqPwz+OCMcljA9
QVnkIpCATtTQZQ41zZrUvndJRl1xv86sNVf4nGCUXF46BMGORwU8FsL299SvsQd7
MwraM6dwF6tou0yzfEpQFWVvaUx+mxiuOQR3zLn9zg0SKhxFiXmcfqQ7U8yVEkUJ
/kWGBEikgwKBgCXfsOb/BCSKbLsEm6TrjGEPk7BlCqzsxFm+qtPpgqKxg8MMRX17
pQ2r13XWqrYLY+h+INI1pkewRMplVqGGVEHR/ZfHc9xOAQSo8s79QdrOtxJ2MNkd
re3kKBaK/5JRyu5LzET7gNTavPKrfXb62BxVwhxq82yNeGzef5W3+gjzAoGANnJH
EsQ5R4Yr1VL/tuNLrfE4Qa+0dmJ7q95aXGc7kyafeOn4BJqDIdMD3uYlLw2e3kvG
3zh7G/5MYRqO6OtWmoKpJz7HE+2etx1X9NI0UlD7OSec3hPN7xcPgBdiZGjRWYZQ
XocPnklzynMYPpseELpNOnxtcp6kXGWwlqHxCGsCgYEAtXLqcQNrulwCp0GgMyOm
fO18QkK2hsUP2x6NjDQKfBb1M7+MlLpQf7a8zWFo3ZYTdCc+kjEM8cPJIamDeCs9
vXaonjswsr8PWyD0brmcV4sjowdHWxSOIBjfcGBfgeFZGtXHj1A5NU2dh/sOQ8fN
VM2qXZURYYeZmS0RsoqaaYM=
-----END PRIVATE KEY-----
"""


@pytest.fixture
def auth() -> TlsClientAuth:
    return TlsClientAuth(
        ca_pem=_TEST_CERT_PEM,
        cert_pem=_TEST_CERT_PEM,
        key_pem=_TEST_KEY_PEM,
    )


def test_client_cert_ssl_context_properties(auth: TlsClientAuth) -> None:
    ctx = client_cert_ssl_context(auth)
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_REQUIRED


def test_client_cert_ssl_context_cleans_up_temp_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    auth: TlsClientAuth,
) -> None:
    known_dir = tmp_path / "tls_tmp"
    known_dir.mkdir()

    def patched_mkdtemp(**kwargs):
        return str(known_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", patched_mkdtemp)

    client_cert_ssl_context(auth)

    assert not known_dir.exists()


# --------------------------------------------------------------------------- #
# A real mutual-TLS handshake. The printer is its own CA: it signs both its
# broker certificate and the client certificate it hands out at pairing.
# --------------------------------------------------------------------------- #


class Pki(NamedTuple):
    ca_pem: str
    server_cert: Path
    server_key: Path
    client_cert_pem: str
    client_key_pem: str


def _openssl(*args: str) -> None:
    subprocess.run(["openssl", *args], check=True, capture_output=True)


def _issue_ca(directory: Path, name: str) -> tuple:
    key, cert = directory / f"{name}-ca.key", directory / f"{name}-ca.pem"
    _openssl(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
        "-out", str(cert), "-days", "1", "-subj", f"/CN={name} CA",
    )
    return key, cert


def _issue_leaf(directory: Path, ca: tuple, common_name: str) -> tuple:
    ca_key, ca_cert = ca
    key = directory / f"{common_name}.key"
    request = directory / f"{common_name}.csr"
    cert = directory / f"{common_name}.pem"
    _openssl(
        "req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key),
        "-out", str(request), "-subj", f"/CN={common_name}",
    )
    _openssl(
        "x509", "-req", "-in", str(request), "-CA", str(ca_cert),
        "-CAkey", str(ca_key), "-CAcreateserial", "-days", "1", "-out", str(cert),
    )
    return key, cert


def _make_pki(directory: Path, name: str) -> Pki:
    ca = _issue_ca(directory, name)
    server_key, server_cert = _issue_leaf(directory, ca, f"{name}-printer")
    client_key, client_cert = _issue_leaf(directory, ca, f"{name}-client")
    return Pki(
        ca_pem=ca[1].read_text(),
        server_cert=server_cert,
        server_key=server_key,
        client_cert_pem=client_cert.read_text(),
        client_key_pem=client_key.read_text(),
    )


@pytest.fixture(scope="module")
def printer_pki(tmp_path_factory) -> Pki:
    return _make_pki(tmp_path_factory.mktemp("printer-pki"), "printer")


@pytest.fixture(scope="module")
def stranger_pki(tmp_path_factory) -> Pki:
    return _make_pki(tmp_path_factory.mktemp("stranger-pki"), "stranger")


def _paired(pki: Pki) -> TlsClientAuth:
    return TlsClientAuth(
        ca_pem=pki.ca_pem, cert_pem=pki.client_cert_pem, key_pem=pki.client_key_pem
    )


@pytest_asyncio.fixture
async def mtls_broker(printer_pki: Pki, monkeypatch):
    """The test broker with a server context that REQUIRES a client
    certificate signed by the printer's CA."""

    def require_client_cert() -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(printer_pki.server_cert, printer_pki.server_key)
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cadata=printer_pki.ca_pem)
        return context

    monkeypatch.setattr(mqtt_broker, "self_signed_context", require_client_cert)
    broker = await Broker(tls=True).start()
    yield broker
    await broker.stop()


def _connect(broker: Broker, options: ConnectionOptions):
    pool = mqtt.pool_for(
        PoolRegistry(), EventLoopProvider(asyncio.get_running_loop())
    )
    lease = mqtt.connect(f"mqtts://127.0.0.1:{broker.port}", pool=pool, options=options)
    drops: List[Disconnected] = []
    lease.event_bus.on(Disconnected, drops.append)
    return lease, drops


async def _first_drop(drops: List[Disconnected]) -> Disconnected:
    for _ in range(500):
        if drops:
            return drops[0]
        await asyncio.sleep(0.01)
    raise AssertionError("the connection was never refused")


@pytest.mark.asyncio
async def test_mutual_tls_handshake_presents_the_paired_client_certificate(
    mtls_broker: Broker, printer_pki: Pki
) -> None:
    lease, _drops = _connect(
        mtls_broker, ConnectionOptions(tls_client_auth=_paired(printer_pki))
    )
    try:
        assert await lease.ready(timeout=5)
        await mtls_broker.wait_for(lambda: mtls_broker.sessions)
        peer = mtls_broker.sessions[0].writer.get_extra_info("peercert")
        assert dict(field[0] for field in peer["subject"]) == {
            "commonName": "printer-client"
        }
    finally:
        await lease.close()


@pytest.mark.asyncio
async def test_mutual_tls_without_the_paired_certificate_is_refused(
    mtls_broker: Broker, printer_pki: Pki, stranger_pki: Pki
) -> None:
    """Plain TLS (no client certificate) and a certificate from another CA are
    both turned away by the broker's handshake."""
    anonymous, anonymous_drops = _connect(mtls_broker, ConnectionOptions())
    stranger_auth = TlsClientAuth(
        # Trusts the printer, but presents a certificate the printer never issued.
        ca_pem=printer_pki.ca_pem,
        cert_pem=stranger_pki.client_cert_pem,
        key_pem=stranger_pki.client_key_pem,
    )
    stranger, stranger_drops = _connect(
        mtls_broker, ConnectionOptions(tls_client_auth=stranger_auth)
    )
    try:
        await _first_drop(anonymous_drops)
        await _first_drop(stranger_drops)
        assert not anonymous.connected and not stranger.connected
        assert mtls_broker.connects == []  # no CONNECT ever got past the handshake
    finally:
        await anonymous.close()
        await stranger.close()


@pytest.mark.asyncio
async def test_mutual_tls_verifies_the_broker_against_the_paired_ca(
    mtls_broker: Broker, printer_pki: Pki, stranger_pki: Pki
) -> None:
    """The paired CA pins the broker: a client trusting a different CA refuses
    the printer's certificate before sending anything."""
    impostor_trust = TlsClientAuth(
        ca_pem=stranger_pki.ca_pem,
        cert_pem=printer_pki.client_cert_pem,
        key_pem=printer_pki.client_key_pem,
    )
    lease, drops = _connect(
        mtls_broker, ConnectionOptions(tls_client_auth=impostor_trust)
    )
    try:
        drop = await _first_drop(drops)
        assert isinstance(drop.code.transport_error, ssl.SSLCertVerificationError)
        assert mtls_broker.connects == []
    finally:
        await lease.close()
