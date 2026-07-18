import ast
import pathlib

from simplyprint_ws_client.common.debug.connectivity import ConnectivityReport
from simplyprint_ws_client.core.api.url_builder import default_connectivity_report


def test_generate_default_report():
    report = default_connectivity_report()
    assert isinstance(report, ConnectivityReport)


def test_connectivity_module_does_not_import_core():
    """The leaf debug module gets its URL lists injected by core, never by
    importing upward itself."""
    path = (
        pathlib.Path(__file__).parent.parent
        / "simplyprint_ws_client"
        / "common"
        / "debug"
        / "connectivity.py"
    )
    tree = ast.parse(path.read_text(), str(path))
    offenders = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        for module in modules:
            if module.startswith("simplyprint_ws_client.core"):
                offenders.append(f"line {node.lineno}: {module}")
    assert offenders == [], f"connectivity.py imports core: {offenders}"
