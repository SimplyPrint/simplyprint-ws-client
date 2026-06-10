from enum import Enum
from os import environ
from typing import NamedTuple

from yarl import URL

from simplyprint_ws_client.const import IS_TESTING


class SimplyPrintWsVersion(Enum):
    VERSION_0_1 = "0.1"
    VERSION_0_2 = "0.2"


class SimplyPrintURLCollection(NamedTuple):
    web: URL
    api: URL
    ws: URL


PRODUCTION_URLS = SimplyPrintURLCollection(
    URL("https://simplyprint.io"),
    URL("https://api.simplyprint.io"),
    URL("wss://ws.simplyprint.io"),
)

TESTING_URLS = SimplyPrintURLCollection(
    URL("https://test.simplyprint.io"),
    URL("https://testapi.simplyprint.io"),
    URL("wss://testws3.simplyprint.io"),
)

STAGING_URLS = SimplyPrintURLCollection(
    URL("https://staging.simplyprint.io"),
    URL("https://apistaging.simplyprint.io"),
    URL("wss://wsstaging.simplyprint.io"),
)

PILOT_URLS = SimplyPrintURLCollection(
    URL("https://pilot.simplyprint.io"),
    URL("https://pilotapi.simplyprint.io"),
    URL("wss://pilotws.simplyprint.io"),
)

LOCALHOST_URLS = SimplyPrintURLCollection(
    URL("http://localhost:8080"),
    URL("http://localhost:8080/api"),
    URL("ws://localhost:8081"),
)


def get_custom_urls() -> SimplyPrintURLCollection:
    """Parse the following environment variables as yarl.URL and convert into SimplyPrintURLs

    SIMPLYPRINT_WS_URL
    SIMPLYPRINT_API_URL
    SIMPLYPRINT_MAIN_URL
    Default to _localhost if not set
    """

    return SimplyPrintURLCollection(
        URL(environ.get("SIMPLYPRINT_MAIN_URL", "http://localhost:8080")),
        URL(environ.get("SIMPLYPRINT_API_URL", "http://localhost:8080/api")),
        URL(environ.get("SIMPLYPRINT_WS_URL", "ws://localhost:8081")),
    )


class SimplyPrintBackend(Enum):
    PRODUCTION = "production"
    TESTING = "test"
    STAGING = "staging"
    LOCALHOST = "local"
    PILOT = "pilot"
    CUSTOM = "custom"

    def urls(self) -> SimplyPrintURLCollection:
        if self is self.CUSTOM:
            return get_custom_urls()

        if urls := {
            SimplyPrintBackend.PRODUCTION: PRODUCTION_URLS,
            SimplyPrintBackend.TESTING: TESTING_URLS,
            SimplyPrintBackend.STAGING: STAGING_URLS,
            SimplyPrintBackend.LOCALHOST: LOCALHOST_URLS,
            SimplyPrintBackend.PILOT: PILOT_URLS,
        }.get(self):
            return urls

        raise ValueError(f"Invalid backend: {self}")


class SimplyPrintURL:
    _active_backend: SimplyPrintBackend = SimplyPrintBackend.PRODUCTION

    @staticmethod
    def set_backend(backend: SimplyPrintBackend):
        SimplyPrintURL._active_backend = backend

    @staticmethod
    def backend_urls() -> SimplyPrintURLCollection:
        return SimplyPrintURL._active_backend.urls()

    @property
    def main_url(self) -> URL:
        return self.backend_urls().web

    @property
    def api_url(self) -> URL:
        return self.backend_urls().api

    @property
    def ws_url(self) -> URL:
        return self.backend_urls().ws / SimplyPrintWsVersion.VERSION_0_2.value


def default_connectivity_report(**kwargs):
    """Generate a :class:`ConnectivityReport` probing every SimplyPrint backend.

    Lives here (not on the report) so the leaf ``common.debug`` module never
    looks up the SimplyPrint endpoints itself -- core, the layer that owns
    them, injects the URL lists. Lazy import: the debug module pulls in heavy
    optional networking deps that url_builder consumers never need.
    """
    from simplyprint_ws_client.common.debug.connectivity import ConnectivityReport

    backends = (
        SimplyPrintBackend.PRODUCTION,
        SimplyPrintBackend.STAGING,
        SimplyPrintBackend.TESTING,
    )
    return ConnectivityReport.generate(
        [str(backend.urls().ws) for backend in backends],
        [str(backend.urls().api) for backend in backends],
        ["1.1.1.1", "google.com"],
        **kwargs,
    )


if value := environ.get("SIMPLYPRINT_BACKEND"):
    SimplyPrintURL.set_backend(SimplyPrintBackend(value))
elif {
    "SIMPLYPRINT_WS_URL",
    "SIMPLYPRINT_API_URL",
    "SIMPLYPRINT_MAIN_URL",
} & environ.keys():
    # If custom urls are set, use them
    SimplyPrintURL.set_backend(SimplyPrintBackend.CUSTOM)
elif IS_TESTING:
    SimplyPrintURL.set_backend(SimplyPrintBackend.TESTING)
else:
    # Default to production
    SimplyPrintURL.set_backend(SimplyPrintBackend.PRODUCTION)
