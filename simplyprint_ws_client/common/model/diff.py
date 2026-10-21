"""Merge-and-diff for device models: re-exported from ``simplyprint_sdk.pydantic``.

A device model mirrors a printer's last-known report; when a new (partial) report
arrives, :meth:`SimpleUpdateModel.update_model` folds it in and returns a recursive
:data:`UpdatedFieldsType` describing exactly what changed (old -> new at each leaf).
Integrations read that diff to drive side-effects (e.g. "this field flipped, so ask
the printer for X").

The engine lives in the open SDK, which brands build on. It used to be copied
there; a copy made two ``UpdatedField`` classes and two ``Atomic`` markers, so an
``isinstance`` or ``is_atomic`` from one never matched the other's. Re-exporting
keeps one copy: every name below *is* the SDK's object, under this module's old
import path, so nothing changes for code importing it from here.
"""

from simplyprint_sdk.pydantic import (
    SimpleUpdateModel,
    UpdatedField,
    UpdatedFieldsType,
    UpdatedListType,
    UpdateModel,
    updated_fields_pretty_print,
    updated_fields_substitute_new,
)

__all__ = [
    "SimpleUpdateModel",
    "UpdatedFieldsType",
    "UpdatedField",
    "UpdatedListType",
    "UpdateModel",
    "updated_fields_substitute_new",
    "updated_fields_pretty_print",
]
