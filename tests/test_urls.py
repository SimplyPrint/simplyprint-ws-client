from simplyprint_ws_client.core.api.url_builder import (
    SimplyPrintBackend,
    resolve_backend_endpoints,
)
from simplyprint_ws_client.core.settings import ClientSettings


def test_backends_expose_immutable_endpoint_values():
    production = SimplyPrintBackend.PRODUCTION.endpoints()
    testing = SimplyPrintBackend.TESTING.endpoints()
    staging = SimplyPrintBackend.STAGING.endpoints()

    assert str(production.main_url) == "https://simplyprint.io"
    assert str(production.api_url) == "https://api.simplyprint.io"
    assert str(production.websocket_url) == "wss://ws.simplyprint.io"
    assert str(testing.main_url) == "https://test.simplyprint.io"
    assert str(testing.api_url) == "https://testapi.simplyprint.io"
    assert str(testing.websocket_url) == "wss://testws3.simplyprint.io"
    assert str(staging.main_url) == "https://staging.simplyprint.io"
    assert str(staging.api_url) == "https://apistaging.simplyprint.io"
    assert str(staging.websocket_url) == "wss://wsstaging.simplyprint.io"


def test_client_settings_capture_custom_environment_once(monkeypatch):
    monkeypatch.delenv("SIMPLYPRINT_BACKEND", raising=False)
    monkeypatch.setenv("SIMPLYPRINT_MAIN_URL", "https://one.example")
    monkeypatch.setenv("SIMPLYPRINT_API_URL", "https://api-one.example")
    monkeypatch.setenv("SIMPLYPRINT_WS_URL", "wss://ws-one.example")
    first = ClientSettings()

    monkeypatch.setenv("SIMPLYPRINT_MAIN_URL", "https://two.example")
    monkeypatch.setenv("SIMPLYPRINT_API_URL", "https://api-two.example")
    monkeypatch.setenv("SIMPLYPRINT_WS_URL", "wss://ws-two.example")
    second = ClientSettings()

    assert first.backend is SimplyPrintBackend.CUSTOM
    assert str(first.endpoints.api_url) == "https://api-one.example"
    assert str(first.endpoints.websocket_url) == "wss://ws-one.example"
    assert str(second.endpoints.api_url) == "https://api-two.example"
    assert first.endpoints is not second.endpoints


def test_explicit_backends_are_isolated_between_settings():
    production = ClientSettings(backend=SimplyPrintBackend.PRODUCTION)
    staging = ClientSettings(backend=SimplyPrintBackend.STAGING)

    assert production.endpoints is SimplyPrintBackend.PRODUCTION.endpoints()
    assert staging.endpoints is SimplyPrintBackend.STAGING.endpoints()
    assert production.endpoints != staging.endpoints


def test_test_backend_is_resolved_from_the_supplied_environment():
    backend, endpoints = resolve_backend_endpoints(environment={"IS_TESTING": "1"})

    assert backend is SimplyPrintBackend.TESTING
    assert endpoints is SimplyPrintBackend.TESTING.endpoints()
