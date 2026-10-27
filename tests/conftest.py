"""Shared test fixtures with proper type hints."""

import time
from pathlib import Path

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig


@pytest.fixture(autouse=True)
def _no_connection_jitter(monkeypatch):
    """Hub opens and adds are jittered by design (random). The suite pins the
    jitter to zero so every open and add happens when the test drives it;
    the jitter tests patch it back to something they control."""
    import simplyprint_ws_client.core.client as client_module
    import simplyprint_ws_client.core.manager as manager_module

    monkeypatch.setattr(client_module, "_jitter", lambda upper: 0.0)
    monkeypatch.setattr(manager_module, "_jitter", lambda upper: 0.0)


_REAL_CLOCKS = (time.time, time.monotonic)
_last_test = ["(none)"]


@pytest.fixture(autouse=True)
def _clock_is_not_leaked(request):
    """A test that replaces the process clock without restoring it breaks
    every test after it, far from the cause. Checked as each test starts
    (after the previous one's monkeypatch undid its own patches), naming the
    test that leaked it."""
    leaked = [
        name
        for name, f, real in zip(("time", "monotonic"), (time.time, time.monotonic), _REAL_CLOCKS)
        if f is not real
    ]
    previous, _last_test[0] = _last_test[0], request.node.nodeid
    if leaked:
        time.time, time.monotonic = _REAL_CLOCKS
        pytest.fail(
            f"{previous} left time.{', time.'.join(leaked)} replaced; patch it with monkeypatch"
        )


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
