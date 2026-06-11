"""Regression coverage for model diff and reactive annotation behavior."""

from typing import List, Mapping, Optional

from pydantic import BaseModel

from simplyprint_ws_client.common.model.diff import SimpleUpdateModel, UpdatedField
from simplyprint_ws_client.common.model.reactive import ReactiveModel


class _Inner(SimpleUpdateModel, BaseModel):
    a: Optional[int] = None


class _Outer(SimpleUpdateModel, BaseModel):
    items: Optional[List[_Inner]] = None
    nums: Optional[List[int]] = None


def test_no_false_positive_diff_for_untouched_model_lists():
    """A partial update whose list elements carry nothing must not report the list."""
    left = _Outer(items=[_Inner(a=1), _Inner(a=2)])
    right = _Outer(items=[_Inner(), _Inner()])

    assert left.update_model(right) == {}
    assert [item.a for item in left.items] == [1, 2]


def test_model_list_diff_is_index_aligned():
    left = _Outer(items=[_Inner(a=1), _Inner(a=2)])
    right = _Outer(items=[_Inner(), _Inner(a=3)])

    changes = left.update_model(right)

    assert changes == {"items": [None, {"a": UpdatedField(2, 3)}]}
    assert left.items[1].a == 3


def test_scalar_list_element_changes_are_reported():
    left = _Outer(nums=[1, 2, 3])
    right = _Outer(nums=[1, 5, 3])

    changes = left.update_model(right)

    # Scalar elements follow the engine's "touched" semantics: every element
    # the update carried is reported; has_changed() distinguishes real changes.
    assert changes == {
        "nums": [UpdatedField(1, 1), UpdatedField(2, 5), UpdatedField(3, 3)]
    }
    assert left.nums == [1, 5, 3]
    nums_diff = changes["nums"]
    assert [f.has_changed() for f in nums_diff] == [False, True, False]


def test_updated_field_is_generic():
    field: UpdatedField[int] = UpdatedField(1, 2)
    assert field.has_changed()


class _Leaf(ReactiveModel):
    x: int = 0


def test_mapping_fields_are_change_detected():
    detect = ReactiveModel.is_pydantic_change_detect_annotation
    assert detect(Mapping[str, _Leaf]) is True
    assert detect(List[_Leaf]) is True
    assert detect(Optional[_Leaf]) is True
    assert detect(Mapping[str, int]) is False
