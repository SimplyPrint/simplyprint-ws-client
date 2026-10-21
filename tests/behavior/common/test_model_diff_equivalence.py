"""The fast ``update_model`` must merge exactly like the walk it replaced.

Randomized and representative partial reports are folded into twin copies of a
device model, one through ``SimpleUpdateModel.update_model`` and one through
the frozen oracle in ``_reference_update_model``; the diffs, the resulting
models and their fields-sets must match after every report.
"""

import copy
import random
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from simplyprint_sdk.pydantic import _MERGE_PLANS  # the engine's own plan cache

from simplyprint_ws_client.common.model import (
    Atomic,
    ExtraInfo,
    SimpleUpdateModel,
    updated_fields_pretty_print,
    updated_fields_substitute_new,
)
from simplyprint_ws_client.common.model.annotations import is_extra
from simplyprint_ws_client.common.model.diff import UpdatedField

from tests._reference_update_model import reference_update_model


class _Tray(BaseModel, SimpleUpdateModel):
    id: Optional[str] = None
    temp: Optional[float] = None
    color: Optional[str] = None
    k: Optional[float] = None


class _Unit(BaseModel, SimpleUpdateModel):
    id: Optional[str] = None
    humidity: Optional[int] = None
    tray: Optional[List[_Tray]] = None


class _Ams(BaseModel, SimpleUpdateModel):
    ams: Optional[List[_Unit]] = None
    tray_now: Optional[str] = None
    version: int = 0


class _Cam(BaseModel, SimpleUpdateModel):
    url: Optional[str] = None
    on: Optional[bool] = None


class _Custom(BaseModel, SimpleUpdateModel):
    """Overrides the merge: the walk must still call the override."""

    hits: int = 0
    seen: Optional[int] = None

    def update_model(self, other):
        self.hits += 1
        if other.seen is not None and other.seen != self.seen:
            old, self.seen = self.seen, other.seen
            return {"seen": UpdatedField(old, other.seen)}
        return {}


def _wide_fields() -> Dict[str, Any]:
    return {f"f{i}": (Optional[float], None) for i in range(60)}


class _PrintBase(BaseModel, SimpleUpdateModel):
    nozzle_temper: Optional[float] = None
    bed_temper: Optional[float] = None
    gcode_state: Optional[str] = None
    mc_percent: Optional[int] = None
    stage: str = "idle"  # a non-None default is merged on every report
    ams: _Ams = Field(default_factory=_Ams)
    cam: Optional[_Cam] = None
    custom: _Custom = Field(default_factory=_Custom)
    hms: Atomic[Optional[List[Dict[str, int]]]] = None
    pos: Atomic[Optional[_Cam]] = None
    nums: Optional[List[int]] = None
    lights: Optional[List[_Cam]] = None
    t_utc: ExtraInfo[Optional[int]] = None
    raw: ExtraInfo[Optional[str]] = None
    mixed: Optional[Any] = None


# ~75 fields, like a real brand model, so the wide-model path is exercised.
_Print = create_model("_Print", __base__=_PrintBase, **_wide_fields())


class _Root(BaseModel, SimpleUpdateModel):
    print: Optional[_Print] = None
    info: Optional[_Cam] = None
    seq: ExtraInfo[Optional[int]] = None


class _PrintNarrow(BaseModel, SimpleUpdateModel):
    """A different report class: only some fields shared with ``_Print``."""

    nozzle_temper: Optional[float] = None
    unknown: Optional[int] = None
    t_utc: Optional[int] = None  # ExtraInfo only on the other side
    stage: Optional[str] = None


def _maybe(rng: random.Random, p: float, make):
    return make() if rng.random() < p else None


def _tray(rng):
    return _Tray(
        id=_maybe(rng, 0.5, lambda: str(rng.randint(0, 3))),
        temp=_maybe(rng, 0.5, lambda: float(rng.randint(180, 220))),
        color=_maybe(rng, 0.3, lambda: rng.choice(["FF0000", "00FF00"])),
        k=_maybe(rng, 0.2, lambda: rng.choice([0.02, 0.03])),
    )


def _unit(rng):
    return _Unit(
        id=_maybe(rng, 0.6, lambda: str(rng.randint(0, 1))),
        humidity=_maybe(rng, 0.4, lambda: rng.randint(1, 5)),
        tray=_maybe(rng, 0.5, lambda: [_tray(rng) for _ in range(rng.randint(0, 4))]),
    )


def _cam(rng):
    return _Cam(
        url=_maybe(rng, 0.5, lambda: rng.choice(["a", "b", "c"])),
        on=_maybe(rng, 0.5, lambda: rng.random() < 0.5),
    )


def _print_report(rng) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if rng.random() < 0.8:
        data["nozzle_temper"] = float(rng.randint(195, 205))
    if rng.random() < 0.8:
        data["bed_temper"] = float(rng.randint(58, 62))
    if rng.random() < 0.2:
        data["gcode_state"] = rng.choice(["IDLE", "RUNNING", "FINISH"])
    if rng.random() < 0.2:
        data["mc_percent"] = rng.randint(0, 100)
    if rng.random() < 0.2:
        data["stage"] = rng.choice(["idle", "printing"])
    if rng.random() < 0.3:
        data["ams"] = _Ams(
            ams=_maybe(rng, 0.7, lambda: [_unit(rng) for _ in range(rng.randint(0, 3))]),
            tray_now=_maybe(rng, 0.5, lambda: str(rng.randint(0, 254))),
            version=rng.randint(0, 2),
        )
    if rng.random() < 0.2:
        data["cam"] = _cam(rng)
    if rng.random() < 0.2:
        data["custom"] = _Custom(seen=_maybe(rng, 0.7, lambda: rng.randint(0, 3)))
    if rng.random() < 0.2:
        data["hms"] = [{"attr": rng.randint(0, 2)} for _ in range(rng.randint(0, 2))]
    if rng.random() < 0.1:
        data["pos"] = _cam(rng)
    if rng.random() < 0.2:
        data["nums"] = [rng.randint(0, 3) for _ in range(rng.randint(0, 4))]
    if rng.random() < 0.15:
        data["lights"] = [_cam(rng) for _ in range(rng.randint(0, 3))]
    if rng.random() < 0.5:
        data["t_utc"] = rng.randint(0, 10**9)
    if rng.random() < 0.2:
        data["raw"] = "x"
    if rng.random() < 0.15:
        # Type flips between scalar / list / model across reports.
        data["mixed"] = rng.choice(
            [1, "s", [1, 2], [_cam(rng)], _cam(rng), {"k": 1}]
        )
    for i in range(60):
        if rng.random() < 0.05:
            data[f"f{i}"] = float(rng.randint(0, 3))
    return data


def _report(rng) -> _Root:
    return _Root(
        print=_maybe(rng, 0.9, lambda: _Print(**_print_report(rng))),
        info=_maybe(rng, 0.1, lambda: _cam(rng)),
        seq=_maybe(rng, 0.5, lambda: rng.randint(0, 100)),
    )


def _fields_sets(model: Any) -> Any:
    if isinstance(model, BaseModel):
        return (
            type(model).__name__,
            sorted(model.model_fields_set),
            {name: _fields_sets(getattr(model, name)) for name in type(model).model_fields},
        )
    if isinstance(model, list):
        return [_fields_sets(v) for v in model]
    return model


def _run(merge, state, report):
    try:
        return merge(state, copy.deepcopy(report))
    except Exception as error:  # noqa: BLE001 - compared against the oracle
        return error


def _merge_both(fast, ref, report) -> bool:
    """Merge ``report`` into both twins and compare; False once both raised."""
    fast_diff = _run(type(fast).update_model, fast, report)
    ref_diff = _run(reference_update_model, ref, report)
    if isinstance(ref_diff, Exception) or isinstance(fast_diff, Exception):
        # A merge the old walk rejected (e.g. a model field now holding a
        # list) must be rejected the same way, after the same partial merge.
        assert type(fast_diff) is type(ref_diff)
        assert fast.model_dump() == ref.model_dump()
        return False
    _assert_same(fast, ref, fast_diff, ref_diff)
    return True


def _assert_same(fast_state, ref_state, fast_diff, ref_diff):
    assert fast_diff == ref_diff
    # Same keys in the same order, same leaf types (UpdatedField vs dict vs list).
    assert repr(fast_diff) == repr(ref_diff)
    assert fast_state.model_dump() == ref_state.model_dump()
    assert _fields_sets(fast_state) == _fields_sets(ref_state)


@pytest.mark.parametrize("seed", range(40))
def test_randomized_reports_merge_like_the_reference(seed):
    rng = random.Random(seed)
    fast = _Root(print=_Print())
    ref = copy.deepcopy(fast)

    for _ in range(60):
        if not _merge_both(fast, ref, _report(rng)):
            fast = _Root(print=_Print())
            ref = copy.deepcopy(fast)


@pytest.mark.parametrize("seed", range(10))
def test_cross_class_reports_merge_like_the_reference(seed):
    rng = random.Random(1000 + seed)
    fast = _Print(**_print_report(rng))
    ref = copy.deepcopy(fast)

    for _ in range(30):
        report = _PrintNarrow(
            nozzle_temper=_maybe(rng, 0.7, lambda: float(rng.randint(195, 205))),
            unknown=_maybe(rng, 0.7, lambda: 1),
            t_utc=_maybe(rng, 0.7, lambda: rng.randint(0, 9)),
            stage=_maybe(rng, 0.5, lambda: rng.choice(["idle", "printing"])),
        )
        assert _merge_both(fast, ref, report)


def test_temperature_only_report_merges_like_the_reference():
    base = _Root(print=_Print(nozzle_temper=200.0, bed_temper=60.0, stage="printing"))
    fast, ref = copy.deepcopy(base), copy.deepcopy(base)
    for temp in (200.0, 201.0, 201.0, 199.5):
        report = _Root(print=_Print(nozzle_temper=temp, t_utc=1))
        assert _merge_both(fast, ref, report)


def test_a_bambu_like_session_merges_like_the_reference():
    """A full push, temperature ticks, an AMS swap, list growth and shrink, an
    equal atomic value, and an overridden merge, in the order a printer sends
    them. (From the client's former SDK-vs-framework parity test.)"""
    full = _Root(
        print=_Print(
            nozzle_temper=200.0,
            bed_temper=60.0,
            gcode_state="RUNNING",
            stage="printing",
            ams=_Ams(
                ams=[
                    _Unit(
                        id="0",
                        tray=[_Tray(id="0", color="FF0000") for _ in range(4)],
                    )
                ],
                tray_now="0",
            ),
            hms=[{"attr": 1}],
            lights=[_Cam(url="a", on=True)],
            t_utc=1,
        ),
        seq=1,
    )
    reports = [
        full,
        _Root(print=_Print(nozzle_temper=201.0, t_utc=2)),
        _Root(print=_Print(nozzle_temper=201.0, t_utc=3)),  # no change
        _Root(
            print=_Print(
                ams=_Ams(ams=[_Unit(tray=[_Tray(), _Tray(color="00FF00")])])
            )
        ),
        _Root(print=_Print(ams=_Ams(ams=[_Unit(id="0"), _Unit(id="1")]))),  # grows
        _Root(print=_Print(ams=_Ams(ams=[]))),  # shrinks
        _Root(print=_Print(hms=[{"attr": 1}])),  # atomic, equal value
        _Root(print=_Print(hms=[])),
        _Root(print=_Print(lights=[_Cam(on=False)], gcode_state="FINISH")),
        _Root(print=_Print(custom=_Custom(seen=3)), info=_Cam(url="b")),
    ]
    fast = _Root(print=_Print())
    ref = copy.deepcopy(fast)
    for report in reports:
        assert _merge_both(fast, ref, report)


@pytest.mark.parametrize("seed", range(10))
def test_change_set_helpers_drop_what_did_not_change(seed):
    """``updated_fields_substitute_new`` keeps exactly the leaves whose value
    changed, as their new value; ``pretty_print`` is its ``repr``."""
    rng = random.Random(2000 + seed)
    state = _Root(print=_Print())

    def expected(diff):
        if isinstance(diff, UpdatedField):
            return diff.new if diff.old != diff.new else None
        if isinstance(diff, list):
            return [expected(v) for v in diff]
        if isinstance(diff, dict):
            kept = {k: expected(v) for k, v in diff.items()}
            return {k: v for k, v in kept.items() if v is not None} or None
        return diff

    for _ in range(10):
        diff = _run(type(state).update_model, state, _report(rng))
        if isinstance(diff, Exception):
            return
        assert updated_fields_substitute_new(diff) == expected(diff)
        assert updated_fields_pretty_print(diff) == repr(expected(diff))


def test_self_merge_matches_the_reference():
    rng = random.Random(7)
    fast = _Root(print=_Print(**_print_report(rng)))
    ref = copy.deepcopy(fast)
    _assert_same(fast, ref, fast.update_model(fast), reference_update_model(ref, ref))


def test_models_with_zero_and_one_mergeable_field():
    class _Empty(BaseModel, SimpleUpdateModel):
        only_extra: ExtraInfo[Optional[int]] = None

    class _One(BaseModel, SimpleUpdateModel):
        a: Optional[int] = None

    empty = _Empty(only_extra=1)
    assert empty.update_model(_Empty(only_extra=2)) == {}
    assert empty.only_extra == 1

    one = _One()
    assert one.update_model(_One(a=3)) == {"a": UpdatedField(None, 3)}
    assert one.update_model(_One()) == {}
    assert one.a == 3


def test_a_rebuilt_model_gets_a_fresh_plan():
    """A forward reference completed by ``model_rebuild`` swaps the field dict.

    While ``_Later`` is undefined the ``late`` annotation cannot be evaluated,
    so its FieldInfo carries no ``ExtraInfo`` marker and the merge walks it.
    Once rebuilt, ``late`` is ``ExtraInfo`` and must be skipped: a plan
    cached before the rebuild would still merge it.
    """

    class _LaterImpl(BaseModel):
        x: int = 0

    # ``_Later`` is bound only by the rebuild namespace below.
    class _Fwd(BaseModel, SimpleUpdateModel):
        a: Optional[int] = None
        late: "ExtraInfo[Optional[_Later]]" = None

    unresolved = _Fwd.__pydantic_fields__
    assert not is_extra(unresolved["late"]), "the forward ref resolved early"

    # An incomplete model cannot validate, but model_construct works on it.
    fast, ref = _Fwd.model_construct(), _Fwd.model_construct()
    seven = _LaterImpl(x=7)
    report = _Fwd.model_construct(a=1, late=seven)
    expected = {"a": UpdatedField(None, 1), "late": UpdatedField(None, seven)}
    assert fast.update_model(report) == expected
    assert reference_update_model(ref, report) == expected
    stale = _MERGE_PLANS[(_Fwd, _Fwd)]
    assert stale.names == ("a", "late")

    assert _Fwd.model_rebuild(_types_namespace={"_Later": _LaterImpl})
    assert _Fwd.__pydantic_fields__ is not unresolved
    assert is_extra(_Fwd.__pydantic_fields__["late"])

    report = _Fwd(a=2, late=_LaterImpl(x=1))
    _assert_same(
        fast, ref, fast.update_model(report), reference_update_model(ref, report)
    )
    assert fast.late is seven  # ExtraInfo now: not merged
    fresh = _MERGE_PLANS[(_Fwd, _Fwd)]
    assert fresh is not stale
    assert fresh.names == ("a",)


class _Angry:
    """An ``Any`` value that cannot be compared (``__eq__`` raises)."""

    __hash__ = object.__hash__

    def __eq__(self, other):
        raise TypeError("cannot compare")


class _ArrayLike:
    """numpy-like: ``==`` returns an object that refuses ``bool()``."""

    __hash__ = object.__hash__

    def __eq__(self, other):
        return _Ambiguous()


class _Ambiguous:
    def __bool__(self):
        raise ValueError("truth value is ambiguous")


@pytest.mark.parametrize("make", [_Angry, _ArrayLike])
def test_values_that_cannot_be_compared_are_merged(make):
    class _M(BaseModel, SimpleUpdateModel):
        a: Optional[int] = None
        blob: Any = None
        b: Optional[int] = None

    fast, ref = _M(), _M()
    blob = make()
    for report in (_M(blob=blob), _M(a=1, blob=blob), _M(), _M(b=2, blob=make())):
        fast_diff = fast.update_model(report)
        ref_diff = reference_update_model(ref, report)
        assert list(fast_diff) == list(ref_diff)
        for name, change in fast_diff.items():
            assert change.old is ref_diff[name].old
            assert change.new is ref_diff[name].new
        assert fast.blob is ref.blob and (fast.a, fast.b) == (ref.a, ref.b)


class _Required(BaseModel, SimpleUpdateModel):
    a: Optional[int] = None
    b: int
    c: Optional[int] = None


def test_a_report_missing_a_field_merges_the_earlier_fields_first():
    """``model_construct`` without a required field: like the old walk, the
    fields before the missing one are merged, then ``AttributeError``."""
    fast, ref = _Required(b=0), _Required(b=0)
    report = _Required.model_construct(a=1, c=3)

    with pytest.raises(AttributeError):
        fast.update_model(report)
    with pytest.raises(AttributeError):
        reference_update_model(ref, report)
    assert (fast.a, fast.b, fast.c) == (1, 0, None)
    assert fast.model_dump() == ref.model_dump()
    assert _fields_sets(fast) == _fields_sets(ref)


class _Linked(BaseModel, SimpleUpdateModel):
    """Assigning ``a`` rewrites ``b`` (a validator that rebinds later fields)."""

    model_config = ConfigDict(validate_assignment=True)

    a: Optional[int] = None
    b: Optional[int] = None

    @model_validator(mode="after")
    def _derive_b(self):
        if self.a is not None:
            # Not ``self.b = ...``: that would re-enter validate_assignment.
            self.__dict__["b"] = self.a * 10
        return self


def test_a_self_merge_reads_each_field_after_the_earlier_ones_merged():
    """The old walk read ``b`` after assigning ``a`` (which rewrote ``b``)."""
    fast = _Linked.model_construct(a=1, b=2)
    ref = _Linked.model_construct(a=1, b=2)

    fast_diff = fast.update_model(fast)
    ref_diff = reference_update_model(ref, ref)
    _assert_same(fast, ref, fast_diff, ref_diff)
    assert fast_diff == {"a": UpdatedField(1, 1), "b": UpdatedField(10, 10)}
    assert (fast.a, fast.b) == (1, 10)


def test_a_distinct_report_is_read_before_the_merge_starts():
    """Pinned difference: a distinct ``other`` is snapshotted up front, so a
    rebind of its fields during the walk is not seen (the old walk saw it)."""

    class _Poke(BaseModel, SimpleUpdateModel):
        model_config = ConfigDict(validate_assignment=True)

        a: Optional[int] = None
        b: Optional[int] = None

        @model_validator(mode="after")
        def _poke_report(self):
            target = self.__dict__.get("a")
            if isinstance(target, int) and _POKED:
                _POKED[0].__dict__["b"] = 99
            return self

    _POKED: List[Any] = []
    state = _Poke()
    report = _Poke(a=1, b=2)
    _POKED.append(report)
    assert state.update_model(report) == {
        "a": UpdatedField(None, 1),
        "b": UpdatedField(None, 2),  # the old walk would have read 99
    }
    assert report.b == 99
