"""Shared test fixtures with proper type hints."""

from pathlib import Path

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig


@pytest.fixture(autouse=True)
def _no_connection_jitter(monkeypatch):
    """Hub opens and adds are jittered by design (random). The suite pins the
    jitter to zero so every open and add happens when the test drives it;
    the jitter tests patch it back to something they control."""
    import simplyprint_ws_client.core.client as client_module

    monkeypatch.setattr(client_module, "_jitter", lambda upper: 0.0)


@pytest.fixture
def client() -> Client:
    """Create a configured Client instance for testing."""
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    client.config.id = 1
    client.config.in_setup = False
    return client


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by suite lane so commands can target the taxonomy."""
    for item in items:
        explicit = {marker.name for marker in item.iter_markers()}
        if explicit & {"behavior", "contract", "guardrail", "performance"}:
            continue

        path = Path(getattr(item, "path", item.fspath))
        parts = set(path.parts)
        if "performance" in parts:
            item.add_marker(pytest.mark.performance)
        elif "guardrails" in parts:
            item.add_marker(pytest.mark.guardrail)
        elif "contracts" in parts:
            item.add_marker(pytest.mark.contract)
        else:
            item.add_marker(pytest.mark.behavior)
