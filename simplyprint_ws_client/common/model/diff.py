"""Merge-and-diff for device models.

A device model mirrors a printer's last-known report; when a new (partial) report
arrives, :meth:`SimpleUpdateModel.update_model` folds it in and returns a recursive
:data:`UpdatedFieldsType` describing exactly what changed (old -> new at each leaf).
Integrations read that diff to drive side-effects (e.g. "this field flipped, so ask
the printer for X"). This is the brand-side counterpart to the library's
accumulate-changeset engine and shares the annotation vocabulary in
:mod:`.annotations` (``Atomic`` / ``ExtraInfo``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

try:
    from typing import Self
except ImportError:  # Python < 3.11
    from typing_extensions import Self

# Generic NamedTuple classes need Python 3.11+; typing_extensions backports them
# to the library's 3.9 floor.
from typing_extensions import NamedTuple

from pydantic import BaseModel

from simplyprint_ws_client.common.model.annotations import is_atomic, is_extra

__all__ = [
    "SimpleUpdateModel",
    "UpdatedFieldsType",
    "UpdatedField",
    "UpdateModel",
    "updated_fields_substitute_new",
    "updated_fields_pretty_print",
]

UpdatedT = TypeVar("UpdatedT")


class UpdatedField(NamedTuple, Generic[UpdatedT]):
    old: UpdatedT | None
    new: UpdatedT | None

    def __repr__(self):
        return f"UpdatedField(old={self.old}, new={self.new})"

    def has_changed(self) -> bool:
        return self.old != self.new


#: List diffs are index-aligned with the merged list: ``None`` marks an element
#: that did not change, a dict is a nested-model diff, an ``UpdatedField`` a
#: scalar element change.
UpdatedListType = List[Union["UpdatedFieldsType", UpdatedField, None]]

UpdatedFieldsType = Dict[
    str, Union[UpdatedField, "UpdatedFieldsType", UpdatedListType, None]
]


def updated_fields_substitute_new(
    value: Union["UpdatedFieldsType", List["UpdatedFieldsType"], None],
):
    if isinstance(value, dict):
        new_dict = {}

        for k, v in value.items():
            new_v = updated_fields_substitute_new(v)

            if new_v is None:
                continue

            new_dict[k] = new_v

        return new_dict or None

    if isinstance(value, list):
        new_list = []

        for v in value:
            new_list.append(updated_fields_substitute_new(v))

        return new_list

    if isinstance(value, UpdatedField):
        if not value.has_changed():
            return None

        return value.new

    return value


def updated_fields_pretty_print(items: UpdatedFieldsType):
    items = updated_fields_substitute_new(items)
    return repr(items)


class UpdateModel(ABC):
    """Define how to update a model with another model."""

    @abstractmethod
    def update_model(self, other: Self | BaseModel) -> UpdatedFieldsType: ...


class SimpleUpdateModel(UpdateModel):
    def update_model(
        self: Self | BaseModel, other: Self | BaseModel
    ) -> UpdatedFieldsType:
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
                    if changes := value1.update_model(value2):
                        updated_fields[name] = changes
                    continue

                if isinstance(value1, list):
                    if not isinstance(value2, list):
                        continue

                    # Index-aligned merge that also grows/shrinks the list,
                    # reporting touched elements only (filtered by has_changed()).
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
                            element_diff = value1[i].update_model(value2[i])
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
