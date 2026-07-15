"""Immutable SimplyPrint endpoint configuration.

Environment selection is a composition concern: :class:`ClientSettings`
resolves it once per app. Importing this module never changes process state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from os import environ
from typing import Mapping

from yarl import URL

__all__ = [
    "SimplyPrintBackend",
    "SimplyPrintEndpoints",
    "resolve_backend_endpoints",
    "default_connectivity_report",
]


@dataclass(frozen=True)
class SimplyPrintEndpoints:
    """The three backend roots one app uses for its entire lifetime."""

    main_url: URL
    api_url: URL
    websocket_url: URL


PRODUCTION_ENDPOINTS = SimplyPrintEndpoints(
    URL("https://simplyprint.io"),
    URL("https://api.simplyprint.io"),
    URL("wss://ws.simplyprint.io"),
)
TESTING_ENDPOINTS = SimplyPrintEndpoints(
    URL("https://test.simplyprint.io"),
    URL("https://testapi.simplyprint.io"),
    URL("wss://testws3.simplyprint.io"),
)
STAGING_ENDPOINTS = SimplyPrintEndpoints(
    URL("https://staging.simplyprint.io"),
    URL("https://apistaging.simplyprint.io"),
    URL("wss://wsstaging.simplyprint.io"),
)
PILOT_ENDPOINTS = SimplyPrintEndpoints(
    URL("https://pilot.simplyprint.io"),
    URL("https://pilotapi.simplyprint.io"),
    URL("wss://pilotws.simplyprint.io"),
)
LOCALHOST_ENDPOINTS = SimplyPrintEndpoints(
    URL("http://localhost:8080"),
    URL("http://localhost:8080/api"),
    URL("ws://localhost:8081"),
)


def _custom_endpoints(environment: Mapping[str, str]) -> SimplyPrintEndpoints:
    return SimplyPrintEndpoints(
        URL(environment.get("SIMPLYPRINT_MAIN_URL", "http://localhost:8080")),
        URL(environment.get("SIMPLYPRINT_API_URL", "http://localhost:8080/api")),
        URL(environment.get("SIMPLYPRINT_WS_URL", "ws://localhost:8081")),
    )


class SimplyPrintBackend(Enum):
    PRODUCTION = "production"
    TESTING = "test"
    STAGING = "staging"
    LOCALHOST = "local"
    PILOT = "pilot"
    CUSTOM = "custom"

    def endpoints(
        self, environment: Mapping[str, str] | None = None
    ) -> SimplyPrintEndpoints:
        if self is self.CUSTOM:
            return _custom_endpoints(environ if environment is None else environment)

        endpoints = {
            SimplyPrintBackend.PRODUCTION: PRODUCTION_ENDPOINTS,
            SimplyPrintBackend.TESTING: TESTING_ENDPOINTS,
            SimplyPrintBackend.STAGING: STAGING_ENDPOINTS,
            SimplyPrintBackend.LOCALHOST: LOCALHOST_ENDPOINTS,
            SimplyPrintBackend.PILOT: PILOT_ENDPOINTS,
        }.get(self)
        if endpoints is None:
            raise ValueError(f"Invalid backend: {self}")
        return endpoints


def resolve_backend_endpoints(
    backend: SimplyPrintBackend | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[SimplyPrintBackend, SimplyPrintEndpoints]:
    """Resolve one app's backend and endpoints from one environment snapshot."""

    environment = dict(environ if environment is None else environment)
    if backend is None:
        if value := environment.get("SIMPLYPRINT_BACKEND"):
            backend = SimplyPrintBackend(value)
        elif {
            "SIMPLYPRINT_WS_URL",
            "SIMPLYPRINT_API_URL",
            "SIMPLYPRINT_MAIN_URL",
        } & environment.keys():
            backend = SimplyPrintBackend.CUSTOM
        elif any(environment.get(key) for key in ("IS_TESTING", "DEV_MODE", "DEBUG")):
            backend = SimplyPrintBackend.TESTING
        else:
            backend = SimplyPrintBackend.PRODUCTION

    return backend, backend.endpoints(environment)


def default_connectivity_report(**kwargs):
    """Generate a connectivity report probing the public service backends."""

    from simplyprint_ws_client.common.debug.connectivity import ConnectivityReport

    endpoints = tuple(
        backend.endpoints()
        for backend in (
            SimplyPrintBackend.PRODUCTION,
            SimplyPrintBackend.STAGING,
            SimplyPrintBackend.TESTING,
        )
    )
    return ConnectivityReport.generate(
        [str(item.websocket_url) for item in endpoints],
        [str(item.api_url) for item in endpoints],
        ["1.1.1.1", "google.com"],
        **kwargs,
    )
