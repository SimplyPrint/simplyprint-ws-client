"""The library's Python 3.9 floor: grammar, and no runtime-evaluated PEP 604 unions.

``X | Y`` between types only works at runtime from Python 3.10. Under
``from __future__ import annotations`` an annotation stays a string, so it is
fine -- unless something evaluates it (pydantic does, for model fields). These
are the spots where a union is evaluated on import:

* any annotation on a function, module or class level in a module without
  ``from __future__ import annotations``;
* a module-level type alias (``Alias = str | None``);
* a field of a pydantic model, even with the ``__future__`` import;
* ``isinstance`` / ``issubclass`` with a union.

Use ``typing.Optional`` / ``typing.Union`` there instead.

The merge-and-diff engine and the G-code parser are re-exported from
``simplyprint_sdk`` (a dependency), so their source is no longer scanned here:
the SDK holds the same 3.9 floor with its own guard (its
``tests/test_python39.py``, run on 3.9 in its CI). The 3.9 CI job here imports
them on a real 3.9 all the same.
"""

import ast
import pathlib
from typing import Iterator, List, Set, Tuple

import pytest

import simplyprint_ws_client

pytestmark = pytest.mark.guardrail

_ROOTS = (
    pathlib.Path(simplyprint_ws_client.__file__).parent,
    pathlib.Path(__file__).resolve().parents[1],  # tests/
)
_BUILTIN_TYPES = {
    "bool", "bytearray", "bytes", "complex", "dict", "float", "frozenset", "int",
    "list", "object", "set", "str", "tuple", "type",
}  # fmt: skip
_PYDANTIC_ROOTS = {"BaseModel", "RootModel"}


def _modules() -> Iterator[Tuple[pathlib.Path, str, ast.Module]]:
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            yield path, source, ast.parse(source, str(path), feature_version=(3, 9))


def _has_union(node) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return True
    if isinstance(node, ast.Subscript) and _base_name(node.value) == "Literal":
        return False  # Literal["a | b"] holds values, not types
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:  # a quoted annotation is evaluated just the same
            return _has_union(ast.parse(node.value.strip(), mode="eval").body)
        except SyntaxError:
            return False
    return any(_has_union(child) for child in ast.iter_child_nodes(node))


def _is_type_union(node) -> bool:
    """``a | b`` with an operand that can only be a type (not an int flag)."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)):
        return False

    def type_only(operand) -> bool:
        if isinstance(operand, ast.BinOp) and isinstance(operand.op, ast.BitOr):
            return _is_type_union(operand)
        return (
            (isinstance(operand, ast.Constant) and operand.value is None)
            or (isinstance(operand, ast.Name) and operand.id in _BUILTIN_TYPES)
            or isinstance(operand, ast.Subscript)
        )

    return type_only(node.left) or type_only(node.right)


def _base_name(base) -> str:
    if isinstance(base, ast.Subscript):
        base = base.value
    if isinstance(base, ast.Attribute):
        return base.attr
    return base.id if isinstance(base, ast.Name) else ""


def _pydantic_model_names(trees: List[ast.Module]) -> Set[str]:
    """Class names that (transitively) derive from a pydantic model."""
    classes = [
        n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    ]
    names = set(_PYDANTIC_ROOTS)
    while True:
        found = {
            c.name for c in classes if any(_base_name(b) in names for b in c.bases)
        }
        if found <= names:
            return names
        names |= found


def _violations() -> List[str]:
    modules = list(_modules())
    models = _pydantic_model_names([tree for _, _, tree in modules])
    found = []

    for path, source, tree in modules:
        lazy = "from __future__ import annotations" in source

        def report(node, what: str, path: pathlib.Path = path) -> None:
            found.append(f"{path}:{node.lineno}: {what}")

        for node in tree.body:
            value = (
                node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
            )
            if value is not None and _is_type_union(value):
                report(node, "module-level type alias uses `|`")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not lazy:
                args = node.args
                for arg in args.posonlyargs + args.args + args.kwonlyargs:
                    if _has_union(arg.annotation):
                        report(arg, f"annotation of `{arg.arg}` uses `|`")
                for arg in (args.vararg, args.kwarg):
                    if arg is not None and _has_union(arg.annotation):
                        report(arg, f"annotation of `{arg.arg}` uses `|`")
                if _has_union(node.returns):
                    report(node, f"return annotation of `{node.name}` uses `|`")
            elif isinstance(node, (ast.Module, ast.ClassDef)):
                is_model = isinstance(node, ast.ClassDef) and any(
                    _base_name(b) in models for b in node.bases
                )
                for statement in node.body:
                    if (
                        isinstance(statement, ast.AnnAssign)
                        and (is_model or not lazy)
                        and _has_union(statement.annotation)
                    ):
                        report(statement, "evaluated field annotation uses `|`")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("isinstance", "issubclass")
                and len(node.args) == 2
                and _has_union(node.args[1])
            ):
                report(node, f"`{node.func.id}` with a `|` union")

    return found


def test_every_module_parses_with_the_python_39_grammar() -> None:
    assert list(_modules())  # parsing raises SyntaxError on 3.10+ grammar


def test_no_runtime_evaluated_pep604_unions() -> None:
    assert _violations() == []
