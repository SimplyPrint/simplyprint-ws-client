"""The shared field-annotation vocabulary for the reactive model layer.

Four ``Annotated`` markers, one home. Two drive the library's accumulate-changeset
engine (:class:`~simplyprint_ws_client.common.model.reactive.ReactiveModel`):

* :data:`Exclusive` -- always signal a change, even when the value compares equal
  (mutually-exclusive job states rely on re-signalling).
* :data:`Untracked` -- never tracked, never recursed (config, scratch fields).

Two drive the merge/diff engine (:class:`SimpleUpdateModel`):

* :data:`Atomic` -- treat a collection / nested model as a single leaf value.
* :data:`ExtraInfo` -- exclude from a bulk ``update_model`` merge (timestamps,
  buffers folded in by hand).

``Atomic`` / ``ExtraInfo`` and their predicates are the open SDK's
(``simplyprint_sdk.pydantic``), re-exported: the merge engine lives there, and a
marker only works with the predicate built for its sentinel, so there must be
one copy of each. ``Exclusive`` / ``Untracked`` belong to the framework's own
changeset engine and stay here.

The cached predicates keep the hot path cheap (a field's policy is computed once
per ``FieldInfo``).
"""

from __future__ import annotations

import functools
from typing import Any, TypeVar

from typing import Annotated

from pydantic.fields import FieldInfo

from simplyprint_sdk.pydantic import Atomic, ExtraInfo, is_atomic, is_extra


_ExclusiveSentinel = object()
_TExclusive = TypeVar("_TExclusive", bound=Any)
Exclusive = Annotated[_TExclusive, _ExclusiveSentinel]


@functools.lru_cache(maxsize=None)
def _is_exclusive(field_info: FieldInfo) -> bool:
    return any(v is _ExclusiveSentinel for v in field_info.metadata)


_UntrackedSentinel = object()
_TUntracked = TypeVar("_TUntracked", bound=Any)
Untracked = Annotated[_TUntracked, _UntrackedSentinel]


@functools.lru_cache(maxsize=None)
def _is_untracked(field_info: FieldInfo) -> bool:
    return any(v is _UntrackedSentinel for v in field_info.metadata)


__all__ = [
    "Exclusive",
    "Untracked",
    "Atomic",
    "ExtraInfo",
    "is_atomic",
    "is_extra",
]
