"""Import-smoke guardrails for the whole library package."""

import importlib
import pkgutil

import pytest

import simplyprint_ws_client

pytestmark = pytest.mark.guardrail


def _walk_module_names():
    pkg = simplyprint_ws_client
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        yield info.name


def test_library_modules_import():
    """Every library module imports, while absent optional backends are tolerated."""
    failures = []

    for module_name in sorted(_walk_module_names()):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Heavy optional backends (paho-mqtt, aiomqtt, ...) may be absent in
            # the test environment; anything else is a real failure.
            if (
                exc.name
                and exc.name.split(".")[0] not in simplyprint_ws_client.__name__
            ):
                continue
            failures.append(f"{module_name}: {exc!r}")
        except Exception as exc:  # pragma: no cover - only reached on regressions
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []
