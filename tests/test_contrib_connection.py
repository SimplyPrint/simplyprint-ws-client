"""``contrib.connection`` package-level contracts (brand-free, wire-free).

The pool routing / lifecycle is now exercised against the v2 stack in
``test_pooled_connection_manager.py`` (the manager), ``test_pool_connection.py``
(the lease + router) and ``test_sync_mqtt_pool.py`` / ``test_sync_ws_pool.py``
(the wire families). What stays pinned *here* are the package-level invariants:

* the optional wire libraries (paho-mqtt, websocket-client, aiohttp) stay *lazy* --
  importing the package must not drag them into the import graph, so a base install
  without the extras still imports cleanly;
* ``contrib/__init__.py`` is import-free (no eager leaf / printer_client load); and
* the small always-on leaves (the ``ConnectionState`` vocabulary, the MQTT topic
  matcher) behave.
"""

import subprocess
import sys

from simplyprint_ws_client.contrib.connection import ConnectionState


def _import_is_clean(import_line: str, *forbidden: str) -> None:
    """Run ``import_line`` in a fresh interpreter; assert no forbidden module loaded."""
    checks = "\n".join(
        f"assert {mod!r} not in sys.modules, {mod!r}" for mod in forbidden
    )
    code = f"import sys\n{import_line}\n{checks}\nprint('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"import-purity failed for {import_line!r}:\n{result.stderr}"
    )


def test_connection_import_does_not_load_paho():
    _import_is_clean(
        "import simplyprint_ws_client.contrib.connection", "paho.mqtt.client", "paho"
    )


def test_transport_import_does_not_load_optional_libs():
    # The unified connection package must not eager-load any wire library: the
    # MQTT (paho), websocket-client, websockets and aiohttp leaves all stay lazy.
    _import_is_clean(
        "import simplyprint_ws_client.contrib.connection",
        "websocket",
        "aiohttp",
        "paho",
        "paho.mqtt.client",
    )


def test_contrib_init_is_import_free():
    # Importing the package root must not pull the high-level printer_client (which
    # imports core) nor the leaf submodules -- contrib/__init__.py is empty on purpose.
    _import_is_clean(
        "import simplyprint_ws_client.contrib",
        "simplyprint_ws_client.contrib.printer_client",
        "simplyprint_ws_client.contrib.connection",
    )


def test_connection_state_usable():
    assert ConnectionState.ONLINE.is_usable
    assert not ConnectionState.OFFLINE.is_usable
    assert not ConnectionState.AUTH_FAILED.is_usable
    assert not ConnectionState.CONFIG_INVALID.is_usable


def test_mqtt_topic_matcher_only_applies_mqtt_wildcard_suffix():
    from simplyprint_ws_client.contrib.connection.mqtt_topics import mqtt_topic_matches

    assert mqtt_topic_matches("devices/a/report", "devices/a/report")
    assert mqtt_topic_matches("devices/a/#", "devices/a/report")
    assert mqtt_topic_matches("devices/a/#", "devices/a")
    assert not mqtt_topic_matches("devices/a#", "devices/anything")
    assert not mqtt_topic_matches("devices/a/", "devices/a/report")
