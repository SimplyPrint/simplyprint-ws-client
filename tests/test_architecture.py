"""Architecture invariants for library contrib/ and shared/ — brand-free, DAG-clean, no shims.

These tests enforce the rules from CLAUDE.md and decisions.md:
  LEAK  — no brand identifiers (NAME tokens) in contrib/ + shared/
  SHIM  — no re-export aliases or star imports (both @ module-level)
  DAG   — contrib/__init__ import-free; core+leaves never import contrib.printer_client

The production code is brand-free by hand; these tests turn that into a machine-checked invariant
so future edits (especially migrations from integrations) cannot regress the abstraction.
"""

import pathlib
import ast
import tokenize
from typing import List

import pytest

# The library package root
LIB_PKG = pathlib.Path(__file__).parent.parent / "simplyprint_ws_client"

# Brand names that must NEVER appear as NAME tokens in contrib/ + shared/ (case-insensitive substring)
BRANDS = ("bambu", "anycubic", "creality", "duet", "elegoo", "ultimaker", "centauri")


class TestBrandFree:
    """LEAK — no brand identifiers in contrib/ + shared/."""

    def test_contrib_is_brand_free(self):
        """Contrib package contains zero brand NAME tokens (ignoring comments/strings)."""
        offenders = self._find_brand_leaks(LIB_PKG / "contrib")
        assert not offenders, f"Brand leak in contrib/: {offenders}"

    def test_shared_is_brand_free(self):
        """Shared package contains zero brand NAME tokens (ignoring comments/strings)."""
        offenders = self._find_brand_leaks(LIB_PKG / "shared")
        assert not offenders, f"Brand leak in shared/: {offenders}"

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
    """contrib/discovery is brand-agnostic machinery: even as plain strings, brand
    ports / SSDP topic shapes must not appear (they live only in each integration's
    per-brand discovery spec)."""
    root = LIB_PKG / "contrib" / "discovery"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in _DISCOVERY_LEAK_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(LIB_PKG)}: {token}")
    assert offenders == [], f"brand port/topic leak in contrib/discovery: {offenders}"


# Brand cloud field / login-type tokens that must never appear in the shared
# account surface. Brand NAMES are caught by TestBrandFree + the text scan; these
# are brand API field names (and the no-word-boundary "bambulab") those miss.
_ACCOUNT_LEAK_TOKENS = ("verifycode", "tfakey", "bambulab")


def test_accounts_has_no_brand_field_tokens():
    """contrib/accounts is the neutral cloud-account surface: no brand cloud API
    field / login-type token may appear (they live only in each integration's
    concrete provider)."""
    root = LIB_PKG / "contrib" / "accounts"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in _ACCOUNT_LEAK_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(LIB_PKG)}: {token}")
    assert offenders == [], f"brand field-token leak in contrib/accounts: {offenders}"


class TestNoShims:
    """SHIM — no re-export aliases or star imports in contrib/ + shared/."""

    def test_contrib_has_no_shim_aliases(self):
        """Contrib: no module-level re-export aliases (Name = OtherName)."""
        offenders = self._find_shim_aliases(LIB_PKG / "contrib")
        assert not offenders, f"Shim aliases in contrib/: {offenders}"

    def test_shared_has_no_shim_aliases(self):
        """Shared: no module-level re-export aliases (Name = OtherName)."""
        offenders = self._find_shim_aliases(LIB_PKG / "shared")
        assert not offenders, f"Shim aliases in shared/: {offenders}"

    def test_contrib_has_no_star_imports(self):
        """Contrib: no star re-exports (from X import *) outside __init__.py."""
        offenders = self._find_star_imports(LIB_PKG / "contrib")
        assert not offenders, f"Star re-exports in contrib/: {offenders}"

    def test_shared_has_no_star_imports(self):
        """Shared: no star re-exports (from X import *) outside __init__.py."""
        offenders = self._find_star_imports(LIB_PKG / "shared")
        assert not offenders, f"Star re-exports in shared/: {offenders}"

    def test_state_model_alias_removed(self):
        """SHIM (S-state): the StateModel re-export shim is gone for good.

        ``core/state/state_model.py`` used to expose ``StateModel = ReactiveModel``
        (and re-export Exclusive/Untracked). That alias was a re-export shim: every
        state class is now defined directly on ``ReactiveModel``. This guards against
        the file being resurrected and against the alias creeping back as a local name.
        """
        import importlib

        # 1. The shim file no longer exists.
        shim = LIB_PKG / "core" / "state" / "state_model.py"
        assert not shim.exists(), f"state_model.py was resurrected: {shim}"

        # 2. Importing the dead module path raises ImportError.
        with pytest.raises(ImportError):
            importlib.import_module("simplyprint_ws_client.core.state.state_model")

        # 3. Every state class inherits from the real ReactiveModel, not a local alias.
        from simplyprint_ws_client.contrib.model.reactive import ReactiveModel
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
            "core.state still re-exports a `StateModel` alias"
        )

    def test_job_lock_module_removed(self):
        """SHIM (B1): job_lock.py is gone; set_active_job absorbed into FileTransfer.

        ``contrib/transfer/job_lock.py`` exposed a free ``set_active_job()`` (a
        misnomer — it owned no lock, just three lines of active-job bookkeeping).
        That behavior now lives on the prepare-lifecycle-owning class as
        ``FileTransfer._set_active_job_for_prepare``. This guards against the
        module being resurrected and the free function creeping back into the
        ``contrib.transfer`` surface.
        """
        import importlib

        # 1. The module no longer exists.
        jl = LIB_PKG / "contrib" / "transfer" / "job_lock.py"
        assert not jl.exists(), (
            "job_lock.py must be deleted; set_active_job is now "
            "FileTransfer._set_active_job_for_prepare"
        )

        # 2. Importing the dead module path raises ImportError.
        with pytest.raises(ImportError):
            importlib.import_module("simplyprint_ws_client.contrib.transfer.job_lock")

        # 3. set_active_job is NOT exported from contrib.transfer.
        from simplyprint_ws_client.contrib import transfer

        assert not hasattr(transfer, "set_active_job"), (
            "contrib.transfer still exports a `set_active_job` free function"
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
                looks_typeish = rhs[:1].isupper() or rhs in {"float", "int", "str", "bytes"}
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
                if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                    rel = path.relative_to(LIB_PKG)
                    src = ("." * (node.level or 0)) + (node.module or "")
                    offenders.append(f"{rel}:{node.lineno}::star-from-{src}")
        return offenders


class TestImportDAG:
    """DAG — import-graph acyclicity: contrib/__init__ import-free, core+leaves never import PrinterClient."""

    def test_contrib_init_is_import_free(self):
        """contrib/__init__.py must not import anything (internal or relative)."""
        init = LIB_PKG / "contrib" / "__init__.py"
        if not init.exists():
            pytest.skip("contrib/__init__.py does not exist")
        offenders = []
        try:
            tree = ast.parse(init.read_text(), str(init))
        except SyntaxError:
            pytest.fail(f"contrib/__init__.py has syntax errors")
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(".") or "contrib" in node.module:
                    offenders.append(f"line {node.lineno}: {node.module}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(".") or "contrib" in a.name:
                        offenders.append(f"line {node.lineno}: {a.name}")
        assert not offenders, f"contrib/__init__.py is not import-free: {offenders}"

    def test_core_does_not_import_printer_client(self):
        """core/ must never import contrib.printer_client (cycle)."""
        offenders = self._find_printer_client_imports(LIB_PKG / "core")
        assert not offenders, f"core/ has cycle imports of printer_client: {offenders}"

    def test_contrib_leaves_do_not_import_printer_client(self):
        """contrib leaf modules must not import contrib.printer_client (cycle)."""
        leaves = [
            LIB_PKG / "contrib" / "transport",
            LIB_PKG / "contrib" / "connection",
            LIB_PKG / "contrib" / "model",
            LIB_PKG / "contrib" / "transfer",
            LIB_PKG / "contrib" / "onboarding",
            LIB_PKG / "contrib" / "logging",
        ]
        for leaf in leaves:
            if not leaf.exists():
                continue
            offenders = self._find_printer_client_imports(leaf)
            assert not offenders, f"{leaf.name}/ has cycle imports of printer_client: {offenders}"

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
