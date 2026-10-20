"""The pre-optimisation ``SimpleUpdateModel.update_model``, frozen as an oracle.

Kept verbatim (bar recursion, which stays inside the oracle) so the equivalence
tests and ``scripts/bench_model_diff.py`` can compare the fast merge against the
walk it replaced: same diff, same resulting model, same fields-set.
"""

import functools
from typing import Any, Dict, List, Optional

from simplyprint_ws_client.common.model import annotations
from simplyprint_ws_client.common.model.diff import (
    SimpleUpdateModel,
    UpdatedField,
    UpdatedFieldsType,
    UpdateModel,
)

# The predicates as they were: a default (128-entry) LRU each, which a model
# wider than 128 fields cycles through and misses on every call.
is_atomic = functools.lru_cache(annotations.is_atomic.__wrapped__)
is_extra = functools.lru_cache(annotations.is_extra.__wrapped__)


def _recurse(value1: Any, value2: Any) -> UpdatedFieldsType:
    if type(value1).update_model is SimpleUpdateModel.update_model:
        return reference_update_model(value1, value2)
    return value1.update_model(value2)


def reference_update_model(self: Any, other: Any) -> UpdatedFieldsType:
    updated_fields: Dict[str, Any] = {}
    self_fields = self.__class__.model_fields
    other_fields = other.__class__.model_fields

    for name, field in other_fields.items():
        if name not in self_fields or is_extra(field):
            continue

        value2 = getattr(other, name)

        if value2 is None:
            continue

        value1 = getattr(self, name)

        if not is_atomic(field):
            if isinstance(value1, UpdateModel):
                if changes := _recurse(value1, value2):
                    updated_fields[name] = changes
                continue

            if isinstance(value1, list):
                if not isinstance(value2, list):
                    continue

                changes: List[Optional[Any]] = []
                any_touched = False
                old_length, new_length = len(value1), len(value2)

                for i in range(max(old_length, new_length)):
                    if i >= new_length:
                        changes.append(UpdatedField(value1[i], None))
                        any_touched = True
                    elif i >= old_length:
                        value1.append(value2[i])
                        changes.append(UpdatedField(None, value2[i]))
                        any_touched = True
                    elif isinstance(value1[i], UpdateModel):
                        element_diff = _recurse(value1[i], value2[i])
                        changes.append(element_diff or None)
                        any_touched = any_touched or bool(element_diff)
                    else:
                        changes.append(UpdatedField(value1[i], value2[i]))
                        value1[i] = value2[i]
                        any_touched = True

                if old_length > new_length:
                    del value1[new_length:]

                if any_touched:
                    updated_fields[name] = changes

                continue

        setattr(self, name, value2)
        updated_fields[name] = UpdatedField(value1, value2)

    return updated_fields
