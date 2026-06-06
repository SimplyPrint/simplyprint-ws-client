"""Performance-regression guards for the reactive-model hot paths.

These are not precise benchmarks (the project has no pytest-benchmark dep); they
pin a *generous* wall-clock ceiling so a gross regression -- an accidental O(n)
per ``__setattr__``, a quadratic changeset walk, a merge that recompares the
world -- trips the suite, while wide margins (10x+ over real throughput) keep
them non-flaky on slow or loaded runners. Measured timings are printed so a
human can eyeball drift even when the assertion still passes.

The three guarded paths are exactly the ones an integration runs thousands of
times per printer per minute:

* tracked ``__setattr__`` -- every state mutation,
* ``model_recursive_changeset`` -- drained once per client tick,
* ``SimpleUpdateModel.update_model`` -- run on every inbound device report.
"""

import time
from typing import List, Optional

from pydantic import BaseModel, Field

from simplyprint_ws_client.contrib.model import ReactiveModel, SimpleUpdateModel


class _Ctx:
    """Minimal change context: hands out monotonic ids, ignores signals."""

    def __init__(self) -> None:
        self._id = 0

    def next_msg_id(self) -> int:
        self._id += 1
        return self._id

    def signal(self) -> None:
        pass


class _Leaf(ReactiveModel):
    a: int = 0
    b: float = 0.0
    c: Optional[str] = None


class _Tree(ReactiveModel):
    name: Optional[str] = None
    leaf: _Leaf = Field(default_factory=_Leaf)
    leaves: List[_Leaf] = Field(default_factory=list)


def _tracked_tree(n_leaves: int = 20) -> _Tree:
    tree = _Tree(leaves=[_Leaf() for _ in range(n_leaves)])
    tree.provide_context(lambda: _Ctx())
    return tree


def test_setattr_tracking_throughput():
    tree = _tracked_tree()
    leaf = tree.leaf
    iterations = 50_000

    start = time.perf_counter()
    for i in range(iterations):
        leaf.a = i
    elapsed = time.perf_counter() - start

    print(f"\ntracked __setattr__: {iterations / elapsed:,.0f}/s ({elapsed:.3f}s)")
    # Real throughput is ~10^5-10^6/s; 5s for 50k is a >10x cushion.
    assert elapsed < 5.0, f"setattr tracking regressed: {elapsed:.3f}s for {iterations}"


def test_recursive_changeset_throughput():
    tree = _tracked_tree(n_leaves=20)
    # Dirty a field on every leaf so the walk has real work to roll up.
    for i, leaf in enumerate(tree.leaves):
        leaf.a = i
    tree.leaf.b = 1.0

    iterations = 5_000
    start = time.perf_counter()
    for _ in range(iterations):
        changeset = tree.model_recursive_changeset
    elapsed = time.perf_counter() - start

    # Sanity: the walk actually produced the expected wildcard + nested keys.
    assert "leaves.*.a" in changeset
    assert "leaf.b" in changeset

    print(
        f"\nmodel_recursive_changeset: {iterations / elapsed:,.0f}/s ({elapsed:.3f}s)"
    )
    assert elapsed < 5.0, f"changeset walk regressed: {elapsed:.3f}s for {iterations}"


class _DiffLeaf(BaseModel, SimpleUpdateModel):
    x: Optional[int] = None
    y: Optional[str] = None


class _DiffRoot(BaseModel, SimpleUpdateModel):
    title: Optional[str] = None
    child: Optional[_DiffLeaf] = None
    items: Optional[List[_DiffLeaf]] = None


def test_update_model_merge_throughput():
    base = _DiffRoot(
        title="a",
        child=_DiffLeaf(x=0, y="a"),
        items=[_DiffLeaf(x=0) for _ in range(20)],
    )

    iterations = 5_000
    start = time.perf_counter()
    for i in range(iterations):
        incoming = _DiffRoot(
            title=f"t{i}",
            child=_DiffLeaf(x=i, y=f"y{i}"),
            items=[_DiffLeaf(x=i) for _ in range(20)],
        )
        diff = base.update_model(incoming)
    elapsed = time.perf_counter() - start

    # Sanity: a real diff came back with the old/new leaf semantics.
    assert "title" in diff and diff["title"].has_changed()

    print(f"\nupdate_model merge: {iterations / elapsed:,.0f}/s ({elapsed:.3f}s)")
    assert elapsed < 5.0, (
        f"update_model merge regressed: {elapsed:.3f}s for {iterations}"
    )
