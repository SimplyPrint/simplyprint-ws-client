"""Architecture invariants for the library package: brand-free and DAG-clean.

These tests enforce the rules from CLAUDE.md and decisions.md:
  LEAK  — no brand identifiers (NAME tokens) anywhere in the library package
  DAG   — integration never reaches host runtime

The production code is brand-free by hand; these tests turn that into a machine-checked invariant
so future edits (especially migrations from integrations) cannot regress the abstraction.

The scans cover every current architectural layer so new code cannot bypass a guard.
"""

import pathlib
import ast
import tokenize
from typing import List

import pytest

# The library package root
LIB_PKG = pathlib.Path(__file__).resolve().parents[2] / "simplyprint_ws_client"
pytestmark = pytest.mark.guardrail

# Brand names that must NEVER appear as NAME tokens in the library (case-insensitive substring)
BRANDS = ("bambu", "anycubic", "creality", "duet", "elegoo", "ultimaker", "centauri")


def test_discovery_has_no_process_global_service_accessor():
    active_module = LIB_PKG / "integration" / "discovery" / "active.py"
    assert not active_module.exists()

    forbidden = (
        "active_discovery_service",
        "set_active_discovery_service",
        "discover_from_active_service",
    )
    offenders = []
    for path in LIB_PKG.rglob("*.py"):
        source = path.read_text()
        for name in forbidden:
            if name in source:
                offenders.append(f"{path.relative_to(LIB_PKG)}::{name}")
    assert offenders == []


# The SimplyPrint wire protocol itself names hardware products (MultiMaterialSolution
# member ids, bed-plate types). These modules MIRROR that cloud-defined vocabulary —
# they are protocol constants, not machinery branching on a brand. Everything else
# in the package must stay brand-free.
PROTOCOL_VOCABULARY = ("core/state/models.py",)


class TestBrandFree:
    """LEAK — no brand identifiers anywhere in the library package.

    The protocol-vocabulary mirror is the one explicit, listed exception.
    """

    def test_library_is_brand_free(self):
        """The whole package contains zero brand NAME tokens (ignoring comments/strings)."""
        offenders = [
            o
            for o in self._find_brand_leaks(LIB_PKG)
            if not o.startswith(PROTOCOL_VOCABULARY)
        ]
        assert not offenders, f"Brand leak in library: {offenders}"

    def test_protocol_vocabulary_allowlist_is_not_stale(self):
        """Every allowlisted protocol-vocabulary module still exists and still
        carries protocol-defined brand identifiers (else the entry must be removed
        so the allowlist cannot silently grow stale)."""
        for entry in PROTOCOL_VOCABULARY:
            path = LIB_PKG / entry
            assert path.exists(), f"stale allowlist entry: {entry}"
            hits = self._find_brand_leaks(path.parent)
            assert any(h.startswith(entry) for h in hits), (
                f"{entry} no longer carries protocol vocabulary; remove it from the allowlist"
            )

    @staticmethod
    def _find_brand_leaks(root: pathlib.Path) -> List[str]:
        """Tokenize all .py files; yield 'file:line::brand' for each violation."""
        offenders = []
        if not root.exists():
            return offenders
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                with open(path, "rb") as fh:
                    toks = list(tokenize.tokenize(fh.readline))
            except (tokenize.TokenError, SyntaxError):
                continue
            for tok in toks:
                # Skip non-NAME tokens (strings, comments, keywords, etc are filtered)
                if tok.type != tokenize.NAME:
                    continue
                low = tok.string.lower()
                hit = next((b for b in BRANDS if b in low), None)
                if not hit:
                    continue
                # Found a brand identifier in a NAME token
                rel = path.relative_to(LIB_PKG)
                offenders.append(f"{rel}:{tok.start[0]}::{tok.string}")
        return offenders


class TestImportDAG:
    """DAG — integration code never depends on the host runtime."""

    def test_printer_client_imports_only_leftward_layers(self):
        """The authoring base may use protocol modules, never host-runtime modules."""
        base = LIB_PKG / "integration" / "client.py"
        # The authoring base may use the protocol half of core/, never its host half.
        host_modules = ("app", "settings", "scheduler", "manager", "host", "registry")
        allowed = ("common", "core", "events", "integration", "wire")
        offenders = []
        tree = ast.parse(base.read_text(), str(base))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            for module in modules:
                if not module.startswith("simplyprint_ws_client."):
                    continue
                parts = module.split(".")
                if parts[1] not in allowed:
                    offenders.append(f"line {node.lineno}: {module}")
                elif parts[1] == "core" and len(parts) > 2 and parts[2] in host_modules:
                    offenders.append(f"line {node.lineno}: {module} (host half)")
        assert not offenders, (
            f"integration/client.py imports non-leftward layers: {offenders}"
        )
