"""The changeset walks must read the same fields as the per-field predicate did.

``model_recursive_changeset`` / ``model_changed_fields`` now walk a per-class
list of tracked fields instead of asking
``is_pydantic_change_detect_annotation`` for every field on every call. The
oracles below are the old loops, and both are run over a mutated
``PrinterState``.
"""

import random
from itertools import repeat
from typing import Dict, Set

from simplyprint_ws_client.common.model import ReactiveModel
from simplyprint_ws_client.core.client import Client, ClientContext
from simplyprint_ws_client.core.config import PrinterConfig


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
        elif what < 0.8:
            printer.webcam_info.connected = rng.random() < 0.5
        elif what < 0.9:
            client.consume()

        for model in _walk(printer):
            assert model.model_recursive_changeset == _reference_changeset(model)
            assert model.model_changed_fields == _reference_changed_fields(model)
