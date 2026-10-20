"""The changeset walks must read the same fields as the per-field predicate did.

``model_recursive_changeset`` / ``model_changed_fields`` now walk a per-class
list of tracked fields instead of asking
``is_pydantic_change_detect_annotation`` for every field on every call. The
oracles below are the old loops, and both are run over a mutated
``PrinterState`` (including its dict-valued peripherals and notifications)
and over small models with tuple, dict and ``Untracked``-redefined fields.
"""

import random
from itertools import repeat
from typing import Dict, Optional, Set, Tuple

import pytest
from pydantic import Field

from simplyprint_ws_client.common.model import ReactiveModel
from simplyprint_ws_client.common.model.annotations import Untracked
from simplyprint_ws_client.common.model.reactive import _TRACKED_FIELDS
from simplyprint_ws_client.core.client import Client, ClientContext
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.state.notifications import NotificationEventPayload


def _reference_changeset(model: ReactiveModel) -> Dict[str, int]:
    changed = model.model_self_changed_fields.copy()

    def put(field: str, change_id: int) -> None:
        changed[field] = max(changed.get(field, -1), change_id)

    for name, info in type(model).model_fields.items():
        value = model.__dict__[name]
        if not value or not model.is_pydantic_change_detect_annotation(field_info=info):
            continue
        if isinstance(value, ReactiveModel) and (inner := _reference_changeset(value)):
            for key, v in inner.items():
                put(f"{name}.{key}", v)
            put(name, max(inner.values()))
            continue
        if isinstance(value, list):
            items = zip(repeat("*"), value)
        elif isinstance(value, tuple):
            items = enumerate(value)
        elif isinstance(value, dict):
            items = value.items()
        else:
            continue
        for index, item in items:
            if isinstance(item, ReactiveModel) and (inner := _reference_changeset(item)):
                for key, v in inner.items():
                    put(f"{name}.{index}.{key}", v)
                top = max(inner.values())
                put(f"{name}.{index}", top)
                put(name, top)
    return changed


def _reference_changed_fields(model: ReactiveModel) -> Set[str]:
    changed = set(model.model_self_changed_fields)
    for name, info in type(model).model_fields.items():
        value = model.__dict__[name]
        if isinstance(value, ReactiveModel) and value.model_has_changed:
            changed.add(name)
        if value and model.is_pydantic_change_detect_annotation(field_info=info):
            if isinstance(value, (list, tuple)):
                items = list(enumerate(value))
            elif isinstance(value, dict):
                items = list(value.items())
            else:
                continue
            if any(
                isinstance(item, ReactiveModel) and item.model_has_changed
                for _, item in items
            ):
                changed.add(name)
    return changed


def _walk(model, seen=None):
    yield model
    for value in model.__dict__.values():
        children = value if isinstance(value, (list, tuple)) else (
            value.values() if isinstance(value, dict) else (value,)
        )
        for child in children:
            if isinstance(child, ReactiveModel):
                yield from _walk(child)


def test_changeset_walks_match_the_per_field_predicate():
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    printer = client.printer
    rng = random.Random(3)

    for step in range(300):
        what = rng.random()
        if what < 0.3:
            printer.bed.temperature.actual = float(rng.randint(50, 70))
        elif what < 0.5:
            printer.tools[0].temperature.target = float(rng.randint(0, 3))
        elif what < 0.6:
            printer.job_info.progress = float(rng.randint(0, 100))
        elif what < 0.7:
            printer.info.machine = rng.choice(["a", "b"])
        elif what < 0.75:
            printer.webcam_info.connected = rng.random() < 0.5
        elif what < 0.8:
            # Dict[str, PeripheralState]: new entries and entry updates.
            printer.peripherals.update(
                f"light:{rng.randint(0, 3)}", value=rng.randint(0, 2)
            )
        elif what < 0.85:
            # Dict[UUID, NotificationEvent]: new events, nested payload
            # changes and resolutions.
            events = list(printer.notifications.notifications.values())
            if not events or rng.random() < 0.3:
                printer.notifications.new(
                    payload=NotificationEventPayload(title=str(step))
                )
            elif rng.random() < 0.5:
                rng.choice(events).payload.title = rng.choice(["a", "b"])
            else:
                rng.choice(events).resolve()
        elif what < 0.9:
            client.consume()

        for model in _walk(printer):
            assert model.model_recursive_changeset == _reference_changeset(model)
            assert model.model_changed_fields == _reference_changed_fields(model)


class _Ctx:
    def __init__(self) -> None:
        self.n = 0

    def next_msg_id(self) -> int:
        self.n += 1
        return self.n

    def signal(self) -> None:
        pass


class _Leaf(ReactiveModel):
    v: Optional[int] = None


class _Box(ReactiveModel):
    leaf: _Leaf = Field(default_factory=_Leaf)
    pair: Tuple[_Leaf, _Leaf] = Field(default_factory=lambda: (_Leaf(), _Leaf()))
    many: Tuple[_Leaf, ...] = ()
    by_key: Dict[str, _Leaf] = Field(default_factory=dict)
    plain: Optional[int] = None


class _QuietBox(_Box):
    """Redefines tracked fields as ``Untracked``: the walk must not recurse."""

    leaf: Untracked[_Leaf] = Field(default_factory=_Leaf)
    by_key: Untracked[Dict[str, _Leaf]] = Field(default_factory=dict)


@pytest.mark.parametrize("cls", [_Box, _QuietBox])
def test_tuple_dict_and_untracked_fields_walk_like_the_predicate(cls):
    ctx = _Ctx()
    box = cls(many=(_Leaf(), _Leaf(), _Leaf()))
    box.provide_context(lambda: ctx)
    rng = random.Random(11)

    def fresh_leaf() -> _Leaf:
        leaf = _Leaf()
        leaf.provide_context(box)
        return leaf

    for _ in range(200):
        what = rng.random()
        if what < 0.15:
            box.leaf.v = rng.randint(0, 3)
        elif what < 0.3:
            rng.choice(box.pair).v = rng.randint(0, 3)
        elif what < 0.45:
            if box.many:
                rng.choice(box.many).v = rng.randint(0, 3)
        elif what < 0.5:
            box.many = tuple(fresh_leaf() for _ in range(rng.randint(0, 3)))
        elif what < 0.6:
            key = str(rng.randint(0, 3))
            if key not in box.by_key:
                box.by_key[key] = fresh_leaf()
                box.model_set_changed("by_key")
        elif what < 0.75:
            if box.by_key:
                rng.choice(list(box.by_key.values())).v = rng.randint(0, 3)
        elif what < 0.85:
            box.plain = rng.randint(0, 3)
        else:
            for model in _walk(box):
                model.model_reset_changed()

        for model in _walk(box):
            assert model.model_recursive_changeset == _reference_changeset(model)
            assert model.model_changed_fields == _reference_changed_fields(model)

    tracked = set(_TRACKED_FIELDS[cls][1])
    if cls is _QuietBox:
        assert tracked == {"pair", "many"}
    else:
        assert tracked == {"leaf", "pair", "many", "by_key"}


def test_a_rebuilt_model_gets_fresh_tracked_fields():
    """A forward reference completed by ``model_rebuild`` swaps the field dict.

    While ``_LateLeaf`` is undefined, ``child`` is not a change-detect
    annotation and the walk skips it; once rebuilt it must recurse. A tracked
    set cached before the rebuild would keep skipping it.
    """

    class _LateLeafImpl(ReactiveModel):
        v: Optional[int] = None

    # ``_LateLeaf`` is bound only by the rebuild namespace below.
    class _Holder(ReactiveModel):
        child: "Optional[_LateLeaf]" = None

    unresolved = _Holder.__pydantic_fields__
    child = _LateLeafImpl()
    child.model_self_changed_fields["v"] = 4
    # An incomplete model cannot validate, but model_construct works on it.
    holder = _Holder.model_construct(child=child)

    assert holder.model_recursive_changeset == _reference_changeset(holder) == {}
    assert holder.model_changed_fields == _reference_changed_fields(holder)
    assert _TRACKED_FIELDS[_Holder][1] == {}

    assert _Holder.model_rebuild(_types_namespace={"_LateLeaf": _LateLeafImpl})
    assert _Holder.__pydantic_fields__ is not unresolved

    expected = {"child.v": 4, "child": 4}
    assert holder.model_recursive_changeset == _reference_changeset(holder) == expected
    assert holder.model_changed_fields == _reference_changed_fields(holder)
    assert list(_TRACKED_FIELDS[_Holder][1]) == ["child"]
