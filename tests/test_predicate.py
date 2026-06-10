from simplyprint_ws_client.events import Event
from simplyprint_ws_client.events.predicate import (
    Constant,
    Reduce,
    Eq,
    Extract,
    Sel,
    And,
    EmptyPipe,
)
from simplyprint_ws_client.events.property_path import p


def test_simple():
    predicate = Constant(True)
    assert predicate()

    predicate = Constant(False)
    assert not predicate()


def test_conditional():
    a = [1, 2, 3]
    b = [4, 5, 6]

    predicate = (
        Sel(1)
        | Reduce(lambda x: filter(lambda y: y > 4, x))
        | Reduce(list)
        | Reduce(len)
        | Eq(2)
    )

    assert predicate(a, b)
    assert not predicate(b, a)


def test_property():
    class CustomEvent(Event):
        id: int = 1337
        name: str

        def __init__(self, name: str):
            self.name = name

    event_a = CustomEvent("a")
    event_b = CustomEvent("b")

    event_is_a = Sel(0) | And.chain(Extract(p.name) | Eq("a"), Extract(p.id) | Eq(1337))

    assert event_is_a(event_a)
    assert not event_is_a(event_b)


def test_function_chains():
    def square(x):
        return x * x

    def add(x, y):
        return x + y

    def is_even(x):
        return x % 2 == 0

    predicate = EmptyPipe() | square | Eq(4)
    assert predicate(2)

    predicate = EmptyPipe() | add | Eq(4)
    assert predicate(2, 2)


def test_predicate_equality():
    predicate = Constant(True)

    assert predicate == Constant(True)
    assert predicate != Constant(False)

    predicate = Reduce(lambda x: x) | Reduce(lambda x: x)

    assert predicate == (Reduce(lambda x: x) | Reduce(lambda x: x))
    assert predicate != (
        Reduce(lambda x: x) | Reduce(lambda x: x) | Reduce(lambda x: x)
    )

    predicate = Sel(0) | Reduce(lambda x: x)

    assert predicate == (Sel(0) | Reduce(lambda x: x))
    assert predicate != (Sel(1) | Reduce(lambda x: x))

    predicate = Extract(p.name) | Eq("a")

    assert predicate == (Extract(p.name) | Eq("a"))
    assert predicate != (Extract(p.id) | Eq(1337))

    predicate = And(Constant(True), Constant(True))

    assert predicate == And(Constant(True), Constant(True))
    assert predicate != And(Constant(True), Constant(False))
