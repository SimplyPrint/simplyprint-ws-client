"""The unified reactive/diffable model layer.

One home for the two change-tracking engines an integration builds on and the
single annotation vocabulary they share:

* :class:`ReactiveModel` -- the accumulate-changeset base (``PrinterState`` and
  every nested state model build on it; mutations roll up into a recursive
  changeset the client drains into messages).
* :class:`SimpleUpdateModel` -- the merge-and-diff base a brand device model
  builds on (folds a new report in and returns an :data:`UpdatedFieldsType`).
* annotations: ``Exclusive`` / ``Untracked`` (changeset) and ``Atomic`` /
  ``ExtraInfo`` (merge/diff).

Both engines are deliberately kept distinct -- they solve genuinely different
problems (accumulate-until-drained vs compare-on-ingest) and share the base
vocabulary, not one record type.
"""

from simplyprint_ws_client.common.model.annotations import (
    Atomic,
    Exclusive,
    ExtraInfo,
    Untracked,
    is_atomic,
    is_extra,
)
from simplyprint_ws_client.common.model.diff import (
    SimpleUpdateModel,
    UpdatedField,
    UpdatedFieldsType,
    UpdateModel,
    updated_fields_pretty_print,
    updated_fields_substitute_new,
)
from simplyprint_ws_client.common.model.reactive import ReactiveModel

__all__ = [
    "ReactiveModel",
    "Exclusive",
    "Untracked",
    "Atomic",
    "ExtraInfo",
    "is_atomic",
    "is_extra",
    "SimpleUpdateModel",
    "UpdateModel",
    "UpdatedField",
    "UpdatedFieldsType",
    "updated_fields_pretty_print",
    "updated_fields_substitute_new",
]
