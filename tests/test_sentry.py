"""Tests for Sentry PII scrubbing and the prod-only / quiet-by-default gating."""

import json

import sentry_sdk

from simplyprint_ws_client.core.api.sentry import Sentry
from simplyprint_ws_client.core.settings import ClientSettings


def test_before_send_redacts_pii():
    event = {
        "message": "open /home/javad/.config/SimplyPrint/x failed at 192.168.1.50",
        "exception": {
            "values": [
                {
                    "type": "ValueError",
                    "value": (
                        "mqtts://bblp:secretcode@10.0.0.2:8883 down; "
                        "access_code=abcd1234"
                    ),
                }
            ]
        },
        "extra": {
            "mac": "AA:BB:CC:DD:EE:FF",
            "note": "token=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd",
        },
    }

    scrubbed = Sentry._before_send(event, {})
    blob = json.dumps(scrubbed)

    # No raw PII survives anywhere in the event.
    assert "javad" not in blob
    assert "192.168.1.50" not in blob
    assert "10.0.0.2" not in blob
    assert "AA:BB:CC" not in blob
    assert "secretcode" not in blob
    assert "abcd1234" not in blob
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in blob

    # ...but the error stays legible: placeholders + non-PII context remain.
    assert "<user>" in blob
    assert "<ip>" in blob
    assert "<mac>" in blob
    assert "<redacted>" in blob
    assert "ValueError" in blob  # exception type is not PII
    assert "failed" in scrubbed["message"]


def test_register_scrubber_composes_after_builtins():
    saved = list(Sentry.extra_scrubbers)
    try:
        Sentry.register_scrubber(lambda text: text.replace("SECRET-SERIAL", "<serial>"))
        out = Sentry._scrub_text("printer SECRET-SERIAL at 192.168.0.9")
        assert "SECRET-SERIAL" not in out
        assert "<serial>" in out
        assert "<ip>" in out  # built-in patterns still apply
    finally:
        Sentry.extra_scrubbers[:] = saved


def test_initialize_sentry_skips_in_development(monkeypatch):
    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda *a, **k: calls.append((a, k)))

    Sentry.initialize_sentry(
        ClientSettings(sentry_dsn="https://example@o0.ingest.sentry.io/1", development=True)
    )

    assert calls == []  # dev/source runs never report


def test_initialize_sentry_skips_without_dsn(monkeypatch):
    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda *a, **k: calls.append((a, k)))

    Sentry.initialize_sentry(ClientSettings(sentry_dsn=None, development=False))

    assert calls == []
