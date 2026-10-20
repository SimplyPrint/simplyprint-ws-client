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
from itertools import compress, repeat
from operator import is_not, itemgetter
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

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


class _MergePlan:
    """The per-``(self class, other class)`` field walk of ``update_model``.

    ``names`` / ``atomic`` list, in the other class's field order, the fields
    a merge may touch: present on self and not ``ExtraInfo``. ``read`` pulls
    their values out of an instance ``__dict__`` in one C call.
    """

    __slots__ = ("self_fields", "other_fields", "names", "atomic", "read")

    def __init__(self, self_fields: Any, other_fields: Any) -> None:
        self.self_fields = self_fields
        self.other_fields = other_fields
        fields = [
            (name, field)
            for name, field in other_fields.items()
            if name in self_fields and not is_extra(field)
        ]
        self.names = tuple(name for name, _ in fields)
        self.atomic = tuple(is_atomic(field) for _, field in fields)
        if len(self.names) > 1:
            self.read = itemgetter(*self.names)
        elif self.names:
            only = self.names[0]
            self.read = lambda values: (values[only],)
        else:
            self.read = lambda values: ()

    def present(self, self_: Any, other: Any) -> Iterator[Tuple[int, Any]]:
        """``(index, value)`` of every non-None mergeable field of ``other``.

        The fast path snapshots ``other.__dict__`` up front and drops the None
        values in C (``is_not``: identity only, so a value whose ``__eq__``
        raises or returns a non-bool is never compared). It falls back to the
        old field-by-field read, interleaved with the merge, whenever a
        snapshot could observe something the old walk did not:

        * a field missing from ``other.__dict__`` (``model_construct`` without
          a required field): the earlier fields are merged first, then the
          ``getattr`` of the missing one raises ``AttributeError``;
        * a self-merge (``other is self``): a ``validate_assignment`` model
          validator may rebind later fields mid-walk, and each one must be
          read after the earlier ones were assigned.

        The one remaining difference: when ``other`` is a distinct object
        whose fields are rebound *during* the walk (only possible if a
        validator on ``self`` reaches into ``other``), the snapshot keeps the
        values ``other`` had when the merge started.
        """
        if other is not self_:
            try:
                values = self.read(other.__dict__)
            except (KeyError, AttributeError):
                pass
            else:
                return compress(enumerate(values), map(is_not, values, repeat(None)))
        return _read_one_by_one(other, self.names)


def _read_one_by_one(
    other: Any, names: Tuple[str, ...]
) -> Iterator[Tuple[int, Any]]:
    """The old walk's read: one ``getattr`` per field, as the merge goes."""
    for index, name in enumerate(names):
        value = getattr(other, name)
        if value is not None:
            yield index, value


#: Plans are computed once per class pair. The field dicts are kept so a
#: ``model_rebuild`` (which swaps them) invalidates the entry.
_MERGE_PLANS: Dict[Tuple[type, type], _MergePlan] = {}


def _merge_plan(self_cls: type, other_cls: type) -> _MergePlan:
    plan = _MERGE_PLANS.get((self_cls, other_cls))
    if (
        plan is not None
        and plan.self_fields is self_cls.__pydantic_fields__
        and plan.other_fields is other_cls.__pydantic_fields__
    ):
        return plan
    plan = _MERGE_PLANS[(self_cls, other_cls)] = _MergePlan(
        self_cls.model_fields, other_cls.model_fields
    )
    return plan


class SimpleUpdateModel(UpdateModel):
    def update_model(
        self: Self | BaseModel, other: Self | BaseModel
    ) -> UpdatedFieldsType:
        updated_fields: Dict[str, Any] = {}
        # Hot path: runs on every inbound device report, over every field of
        # wide models, most of them None in a partial report. The per-class
        # plan replaces the per-field annotation checks, and the None fields
        # are skipped in C (see _MergePlan.present). An empty sub-model (a
        # default-factory one riding along in a partial report) yields
        # nothing, so the loop body never runs.
        plan = _merge_plan(self.__class__, other.__class__)
        names, atomics = plan.names, plan.atomic

        for i, value2 in plan.present(self, other):
            name = names[i]
            atomic = atomics[i]

            value1 = getattr(self, name)

            if not atomic:
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
