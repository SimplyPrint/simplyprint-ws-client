"""Architecture invariants for the library package — brand-free, DAG-clean, no shims.

These tests enforce the rules from CLAUDE.md and decisions.md:
  LEAK  — no brand identifiers (NAME tokens) anywhere in the library package
  SHIM  — no re-export aliases or star imports (both @ module-level) in the layer dirs
  DAG   — contrib/__init__ import-free; library never ships/imports contrib.printer_client

The production code is brand-free by hand; these tests turn that into a machine-checked invariant
so future edits (especially migrations from integrations) cannot regress the abstraction.

2.0 restructure: the scans cover the new layer dirs (common/, cloud/) alongside the
legacy ones (contrib/, shared/, core/) so no file escapes a guard mid-migration.
"""

import pathlib
import ast
import tokenize
from typing import List

import pytest

# The library package root
LIB_PKG = pathlib.Path(__file__).parent.parent / "simplyprint_ws_client"

# Brand names that must NEVER appear as NAME tokens in the library (case-insensitive substring)
BRANDS = ("bambu", "anycubic", "creality", "duet", "elegoo", "ultimaker", "centauri")

# The layer dirs the alias/star shim scans cover (root __init__.py's lazy PEP 562
# re-export hub is the one sanctioned exception and lives outside these dirs).
LAYER_DIRS = ("common", "core", "device", "integration")


# The SimplyPrint wire protocol itself names hardware products (MultiMaterialSolution
# member ids, bed-plate types). These modules MIRROR that cloud-defined vocabulary —
# they are protocol constants, not machinery branching on a brand. Everything else
# in the package must stay brand-free.
PROTOCOL_VOCABULARY = ("core/state/models.py",)


class TestBrandFree:
    """LEAK — no brand identifiers anywhere in the library package.

    2.0 restructure: widened from contrib/ + shared/ to the whole package (the
    assertion itself is unchanged: zero brand NAME tokens), with the cloud
    protocol-vocabulary mirror as the one explicit, listed exception.
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


# Brand ports / SSDP topic fragments that must never be hardcoded in the shared
# discovery machinery (they belong only in each integration's per-brand spec).
# A brand name as a NAME token is caught by TestBrandFree; these are string
# literals (ports, topic shapes), so they need a plain substring scan.
_DISCOVERY_LEAK_TOKENS = (
    "2021",
    "1900",
    "3030",
    "bambulab",
    "devmodel",
    "ac:3dprinter",
    "modelid",
)


def test_discovery_has_no_brand_ports_or_topics():
    """device/discovery is brand-agnostic machinery: even as plain strings, brand
    ports / SSDP topic shapes must not appear (they live only in each integration's
    per-brand discovery spec)."""
    root = LIB_PKG / "device" / "discovery"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in _DISCOVERY_LEAK_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(LIB_PKG)}: {token}")
    assert offenders == [], f"brand port/topic leak in device/discovery: {offenders}"


# Brand cloud field / login-type tokens that must never appear in the shared
# account surface. Brand NAMES are caught by TestBrandFree + the text scan; these
# are brand API field names (and the no-word-boundary "bambulab") those miss.
_ACCOUNT_LEAK_TOKENS = ("verifycode", "tfakey", "bambulab")


def test_accounts_has_no_brand_field_tokens():
    """device/accounts is the neutral cloud-account surface: no brand cloud API
    field / login-type token may appear (they live only in each integration's
    concrete provider)."""
    root = LIB_PKG / "device" / "accounts"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in _ACCOUNT_LEAK_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(LIB_PKG)}: {token}")
    assert offenders == [], f"brand field-token leak in device/accounts: {offenders}"


class TestNoShims:
    """SHIM — no re-export aliases or star imports in any layer dir."""

    @pytest.mark.parametrize("layer", LAYER_DIRS)
    def test_layer_has_no_shim_aliases(self, layer):
        """No module-level re-export aliases (Name = OtherName) in the layer."""
        offenders = self._find_shim_aliases(LIB_PKG / layer)
        assert not offenders, f"Shim aliases in {layer}/: {offenders}"

    @pytest.mark.parametrize("layer", LAYER_DIRS)
    def test_layer_has_no_star_imports(self, layer):
        """No star re-exports (from X import *) outside __init__.py in the layer."""
        offenders = self._find_star_imports(LIB_PKG / layer)
        assert not offenders, f"Star re-exports in {layer}/: {offenders}"

    def test_state_model_alias_removed(self):
        """SHIM (S-state): the StateModel re-export shim is gone for good.

        ``core/state/state_model.py`` used to expose ``StateModel = ReactiveModel``
        (and re-export Exclusive/Untracked). That alias was a re-export shim: every
        state class is now defined directly on ``ReactiveModel``. This guards against
        the file being resurrected and against the alias creeping back as a local name.
        """
        import importlib

        # 1. The shim file no longer exists (state/ lives under cloud/ since 2.0).
        shim = LIB_PKG / "core" / "state" / "state_model.py"
        assert not shim.exists(), f"state_model.py was resurrected: {shim}"

        # 2. Importing the dead module path raises ImportError.
        with pytest.raises(ImportError):
            importlib.import_module("simplyprint_ws_client.core.state.state_model")

        # 3. Every state class inherits from the real ReactiveModel, not a local alias.
        from simplyprint_ws_client.common.model.reactive import ReactiveModel
        import simplyprint_ws_client.core.state as state_pkg

        state_classes = [
            "TemperatureState",
            "AmbientTemperatureState",
            "FileProgressState",
            "JobInfoState",
            "NotificationEvent",
            "PrinterState",
        ]
        for cls_name in state_classes:
            cls = getattr(state_pkg, cls_name)
            assert ReactiveModel in cls.__mro__, (
                f"{cls_name} does not inherit from ReactiveModel: {cls.__mro__}"
            )

        # 4. No `StateModel` name re-exported from the state package.
        assert not hasattr(state_pkg, "StateModel"), (
            "cloud.state still re-exports a `StateModel` alias"
        )

    def test_job_lock_module_removed(self):
        """SHIM (B1): job_lock.py is gone; set_active_job absorbed into FileTransfer.

        ``transfer/job_lock.py`` (now under device/) exposed a free ``set_active_job()`` (a
        misnomer — it owned no lock, just three lines of active-job bookkeeping).
        That behavior now lives on the prepare-lifecycle-owning class as
        ``FileTransfer._set_active_job_for_prepare``. This guards against the
        module being resurrected and the free function creeping back into the
        ``device.transfer`` surface.
        """
        import importlib

        # 1. The module no longer exists.
        jl = LIB_PKG / "device" / "transfer" / "job_lock.py"
        assert not jl.exists(), (
            "job_lock.py must be deleted; set_active_job is now "
            "FileTransfer._set_active_job_for_prepare"
        )

        # 2. Importing the dead module path raises ImportError.
        with pytest.raises(ImportError):
            importlib.import_module("simplyprint_ws_client.device.transfer.job_lock")

        # 3. set_active_job is NOT exported from device.transfer.
        from simplyprint_ws_client.device import transfer

        assert not hasattr(transfer, "set_active_job"), (
            "device.transfer still exports a `set_active_job` free function"
        )

        # 4. The behavior lives on FileTransfer.
        assert hasattr(transfer.FileTransfer, "_set_active_job_for_prepare"), (
            "FileTransfer must own _set_active_job_for_prepare"
        )

    @staticmethod
    def _find_shim_aliases(root: pathlib.Path) -> List[str]:
        """AST-scan for module-level Name = OtherName; skip decorator RHS and class-level attrs."""
        offenders = []
        if not root.exists():
            return offenders

        SKIP_RHS = {"property", "staticmethod", "classmethod"}

        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(), str(path))
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    continue
                val = node.value
                if isinstance(val, ast.Name):
                    rhs = val.id
                elif isinstance(val, ast.Attribute):
                    rhs = val.attr
                else:
                    continue
                name = node.targets[0].id
                if name == rhs or rhs in SKIP_RHS:
                    continue
                # Treat as a shim only if RHS looks like a type (CapWord or builtin)
                looks_typeish = rhs[:1].isupper() or rhs in {
                    "float",
                    "int",
                    "str",
                    "bytes",
                }
                if not looks_typeish:
                    continue
                rel = path.relative_to(LIB_PKG)
                offenders.append(f"{rel}:{node.lineno}::{name}")
        return offenders

    @staticmethod
    def _find_star_imports(root: pathlib.Path) -> List[str]:
        """AST-scan for from X import * outside __init__.py; non-packages only."""
        offenders = []
        if not root.exists():
            return offenders
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or path.name == "__init__.py":
                continue
            try:
                tree = ast.parse(path.read_text(), str(path))
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and any(
                    a.name == "*" for a in node.names
                ):
                    rel = path.relative_to(LIB_PKG)
                    src = ("." * (node.level or 0)) + (node.module or "")
                    offenders.append(f"{rel}:{node.lineno}::star-from-{src}")
        return offenders


class TestImportDAG:
    """DAG — import-graph acyclicity: contrib/__init__ import-free; the authoring
    base lives in integration/ (promoted back in 2.0 slice C) and imports only
    leftward layers; the old ``contrib.printer_client`` stays a tombstone."""

    @pytest.mark.parametrize("layer", LAYER_DIRS)
    def test_layer_init_imports_no_library_modules_eagerly(self, layer):
        """Layer ``__init__`` discipline (generalizes the old contrib rule): a
        layer root imports no library module eagerly -- docstring-only or PEP 562
        lazy (stdlib imports are fine)."""
        init = LIB_PKG / layer / "__init__.py"
        assert init.exists(), f"{layer}/__init__.py missing"
        offenders = []
        tree = ast.parse(init.read_text(), str(init))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(".") or "simplyprint_ws_client" in node.module:
                    offenders.append(f"line {node.lineno}: {node.module}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if "simplyprint_ws_client" in a.name:
                        offenders.append(f"line {node.lineno}: {a.name}")
        assert not offenders, f"{layer}/__init__.py imports eagerly: {offenders}"

    def test_printer_client_promoted_out_of_library(self):
        """2.0 slice C inverts this contract: the authoring base now SHIPS in the
        library as ``integration/client.py`` (the cycle that forced its demotion
        died with the layer restructure — its imports are all leftward layers).

        The old ``contrib/printer_client.py`` stays a tombstone, and the base must
        import only from {cloud, common, device} + stdlib — never core/runtime.
        """
        tombstone = LIB_PKG / "contrib" / "printer_client.py"
        assert not tombstone.exists(), (
            "contrib/printer_client.py must stay deleted (the base lives in "
            "integration/client.py now; no re-export shim)."
        )

        base = LIB_PKG / "integration" / "client.py"
        assert base.exists(), "integration/client.py (the authoring base) is missing"

        # The authoring base may use the protocol half of core/ but never the
        # HOST half -- post-review, cloud+runtime share the core/ dir, so the
        # tooth moved from layer names to the host module list.
        host_modules = ("app", "settings", "scheduler", "manager", "host", "registry")
        allowed = ("common", "core", "device", "integration")
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

    @staticmethod
    def _find_printer_client_imports(root: pathlib.Path) -> List[str]:
        """Scan all .py files in root; yield 'file:line::printer_client' for cycle imports."""
        offenders = []
        if not root.exists():
            return offenders
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(), str(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if "printer_client" in node.module:
                        rel = path.relative_to(LIB_PKG)
                        offenders.append(f"{rel}:{node.lineno}::printer_client")
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if "printer_client" in a.name:
                            rel = path.relative_to(LIB_PKG)
                            offenders.append(f"{rel}:{node.lineno}::printer_client")
        return offenders
