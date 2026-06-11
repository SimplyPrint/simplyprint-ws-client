"""Regression coverage for predicate and listener behavior."""

from simplyprint_ws_client.events.emitter import Emitter  # noqa: F401 - used by string annotations


def test_predicate_tree_evaluates_all_matching_branches():
    from simplyprint_ws_client.events.event_bus_predicate_tree import (
        EventBusPredicateTree,
    )
    from simplyprint_ws_client.events.predicate import Eq, Gt, Sel

    tree = EventBusPredicateTree()
    # Two sibling chains that can both match the same input.
    a = tree.add("starts-positive", Sel(0) | Gt(0))
    b = tree.add("is-five", Sel(0) | Eq(5))

    matches = {tree.resources[rid] for rid in tree.evaluate(5)}

    # Previously only the first matching sibling was descended.
    assert matches == {"starts-positive", "is-five"}

    tree.remove_resource_id(a)
    matches = {tree.resources[rid] for rid in tree.evaluate(5)}
    assert matches == {"is-five"}

    tree.remove_resource_id(b)
    assert list(tree.evaluate(5)) == []
    assert tree.root.predicates == []


def test_pipe_or_does_not_mutate_the_original_chain():
    from simplyprint_ws_client.events.predicate import Eq, Gt, Sel

    base = Sel(0)
    first = base | Eq(1)
    second = base | Gt(10)

    # Previously building ``second`` silently extended ``base``/``first``.
    assert base.output is None
    assert first(1) is True
    assert first(11) is False
    assert second(11) is True
    assert second(1) is False


def test_reduce_equality_respects_lambda_constants():
    from simplyprint_ws_client.events.predicate import Reduce

    assert Reduce(lambda x: x > 5) != Reduce(lambda x: x > 99)
    assert Reduce(lambda x: x > 5) == Reduce(lambda x: x > 5)


def test_reduce_equality_respects_closures():
    from simplyprint_ws_client.events.predicate import Reduce

    def make(n):
        return lambda x: x > n

    assert Reduce(make(5)) != Reduce(make(99))


def test_forward_emitter_detected_with_string_annotations():
    from simplyprint_ws_client.events.event_bus_listeners import (
        EventBusListener,
        ListenerLifetimeForever,
    )

    # PEP 563-style string annotation, as produced by
    # ``from __future__ import annotations``. ``Emitter`` is importable from
    # this module's globals, exactly like a real handler module.
    def handler(event, emitter: "Emitter"): ...

    listener = EventBusListener(ListenerLifetimeForever(), 0, handler)

    assert listener.forward_emitter == "emitter"
