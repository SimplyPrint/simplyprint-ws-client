import asyncio
import logging
from dataclasses import fields

import pytest

from simplyprint_ws_client.integration.discovery.network import DiagnosticReason
from simplyprint_ws_client.integration.discovery.spec import SubnetScanSpec
from simplyprint_ws_client.integration.discovery.subnet import SubnetScanBackend


def test_subnet_gate_has_one_explicit_service_contract() -> None:
    assert "port" not in {field.name for field in fields(SubnetScanSpec)}


async def _slow_probe(host: str):
    await asyncio.sleep(1)
    return None


def _timeout_backend() -> SubnetScanBackend:
    return SubnetScanBackend(
        SubnetScanSpec(
            brand="test",
            probe=_slow_probe,
            key=lambda record: record.host,
            concurrency=1,
        )
    )


@pytest.mark.asyncio
async def test_scan_timeout_is_a_quiet_miss(caplog):
    caplog.set_level(logging.DEBUG, logger="discovery")

    records = await _timeout_backend().scan(timeout=0.01, hosts=["192.0.2.10"])

    assert records == []
    assert "discovery probe timed out" in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
async def test_diagnostic_timeout_reports_probe_error_without_traceback(caplog):
    caplog.set_level(logging.DEBUG, logger="discovery")

    diagnostic = await _timeout_backend().diagnose("192.0.2.10", timeout=0.01)

    assert diagnostic.reason == DiagnosticReason.PROBE_ERROR
    assert diagnostic.matched is False
    assert diagnostic.message.endswith("the probe timed out")
    assert diagnostic.checks[-1].message == "Fingerprint probe timed out"
    assert "Traceback" not in caplog.text
