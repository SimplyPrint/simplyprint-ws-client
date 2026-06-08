"""``contrib.connection.websocket`` -- the raw async WebSocket socket layer.

The dumb socket the SimplyPrint backend ``Connection`` drives (the :class:`WebSocket`
ABC + two async lib impls), living inside the WebSocket family. The contract pinned
here is import-purity: pulling the package (or its base ABC) must not eager-load
either wire library -- ``websockets`` and ``aiohttp`` stay lazy behind PEP 562
``__getattr__`` so selecting one never drags in the other.
"""

import subprocess
import sys

from simplyprint_ws_client.contrib.connection.websocket.base import (
    WebSocket,
    WebSocketClosed,
    WebSocketError,
)


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


def test_websocket_package_does_not_load_wire_libs():
    _import_is_clean(
        "import simplyprint_ws_client.contrib.connection.websocket",
        "websockets",
        "aiohttp",
    )


def test_base_abc_is_third_party_free():
    # The ABC + error vocabulary must be importable with neither wire lib present.
    _import_is_clean(
        "from simplyprint_ws_client.contrib.connection.websocket import base",
        "websockets",
        "aiohttp",
    )


def test_error_hierarchy():
    assert issubclass(WebSocketClosed, WebSocketError)


def test_abc_cannot_be_instantiated():
    import pytest

    with pytest.raises(TypeError):
        WebSocket()
