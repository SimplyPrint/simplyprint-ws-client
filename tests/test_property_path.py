import pytest
from dataclasses import dataclass, field
from typing import List

from simplyprint_ws_client.common.events.property_path import PropertyPath, as_path, p


def test_simple():
    d = {"a": "a", "b": "g", "c": {"q": [1, 2, 3]}}

    path_for_a = as_path(p["a"])
    path_for_g = as_path(p["b"])
    path_for_3 = as_path(p["c"]["q"][2])

    assert path_for_a.resolve(d) == "a"
    assert path_for_g.resolve(d) == "g"
    assert path_for_3.resolve(d) == 3


def test_object_hash():
    class MyCustomClass: ...

    instance = MyCustomClass()

    d = {instance: 10}

    assert d.get(MyCustomClass()) is None
    assert d.get(instance) == 10

    path = PropertyPath().idx(instance)

    assert path.resolve(d) == 10
    assert as_path(p[instance]).resolve(d) == 10

    with pytest.raises(KeyError):
        as_path(p[MyCustomClass()]).resolve(d)


def test_path_hash():
    p1 = p.cool_path[1]
    p2 = p.cool_path[1]
    p3 = as_path(p).attr("cool_path").idx(1)
    p4 = p.cool_path

    assert hash(p1) == hash(p2)
    assert hash(p1) == hash(p3)
    assert hash(p1) != hash(p4)

    d = {p1: "value"}

    assert d[p1] == d[p2]


def test_class():
    @dataclass
    class C:
        list: List["A"]
        field: int = field(default=10)

    @dataclass
    class B:
        c: C
        list: List[int]

    @dataclass
    class A:
        b: B
        list: List[str]

    obj = A(list=["a", "b", "c"], b=B(list=[1, 2, 3], c=C(list=[])))
    obj.b.c.list.append(obj)

    path = PropertyPath().attr("b").attr("c").attr("list").idx(0)

    assert path.resolve(obj) == obj
