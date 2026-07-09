"""Tests for mutual-TLS client-certificate support.

Covers :func:`~simplyprint_ws_client.wire.paho.client_cert_ssl_context` and
the aiomqtt guard in :func:`~simplyprint_ws_client.wire.mqtt.connect`.
"""

from __future__ import annotations

import ssl
import tempfile

import pytest
import yarl

from simplyprint_ws_client.wire.options import TlsClientAuth
from simplyprint_ws_client.wire.paho import client_cert_ssl_context

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


def test_aiomqtt_tls_client_auth_raises() -> None:
    from simplyprint_ws_client.wire import mqtt

    auth = TlsClientAuth(
        ca_pem=_TEST_CERT_PEM,
        cert_pem=_TEST_CERT_PEM,
        key_pem=_TEST_KEY_PEM,
    )

    with pytest.raises(ValueError, match="tls_client_auth"):
        pool = mqtt.build_pool("aiomqtt", None)
        pool.connect(
            yarl.URL("mqtts://printer.local:8883"),
            mqtt.MqttConnectParams(
                broker=mqtt.MqttBroker("printer.local", 8883),
                retry=mqtt.RetryPolicy(),
                tls_client_auth=auth,
            ),
        )
