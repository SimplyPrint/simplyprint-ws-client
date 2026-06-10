from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Generic, TypeVar, Callable, Any

from simplyprint_ws_client.events.property_path import (
    PropertyPath,
    PropertyPathBuilder,
    as_path,
    Indexable,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


def _code_fingerprint(code) -> tuple:
    """The behavior-relevant parts of a code object (source position excluded,
    constants included - ``lambda x: x > 5`` must differ from ``x > 99``)."""
    return (
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_freevars,
        code.co_cellvars,
    )


def _callables_equivalent(a: Any, b: Any) -> bool:
    """Whether two callables are interchangeable as predicates.

    Distinct-but-identical lambdas compare equal only when their code
    fingerprints match and neither captures free variables - a closure's
    behavior lives in its cells, which the code object misses. Defaults are
    compared separately for the same reason.
    """
    if a == b:
        return True

    code_a = getattr(a, "__code__", None)
    code_b = getattr(b, "__code__", None)

    if code_a is None or code_b is None:
        return False

    return (
        not code_a.co_freevars
        and _code_fingerprint(code_a) == _code_fingerprint(code_b)
        and getattr(a, "__defaults__", None) == getattr(b, "__defaults__", None)
        and getattr(a, "__kwdefaults__", None) == getattr(b, "__kwdefaults__", None)
    )


@dataclass
class Predicate(ABC):
    """Evaluates an input and returns a boolean."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError()


@dataclass
class Constant(Predicate):
    """Constant true or false."""

    value: bool

    def __call__(self, *args, **kwargs) -> bool:
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


@dataclass
class Lambda(Predicate):
    """A lambda predicate."""

    func: Callable

    def __call__(self, *args, **kwargs) -> bool:
        return self.func(*args, **kwargs)


@dataclass
class Unary(Predicate, ABC):
    predicate: Predicate

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.predicate)})"


@dataclass
class Not(Unary):
    def __call__(self, *args, **kwargs) -> bool:
        return not self.predicate(*args, **kwargs)


@dataclass
class Compare(Predicate, ABC):
    value: Any

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.value)})"


@dataclass
class Eq(Compare):
    """Whether the first argument is equal to the value."""

    def __call__(self, *args, **kwargs) -> bool:
        return args[0] == self.value


@dataclass
class Gt(Compare):
    """Whether the first argument is greater than the value."""

    def __call__(self, *args, **kwargs) -> bool:
        return args[0] > self.value


@dataclass
class Lt(Compare):
    """Whether the first argument is less than the value."""

    def __call__(self, *args, **kwargs) -> bool:
        return args[0] < self.value


@dataclass
class Gte(Compare):
    """Whether the first argument is greater than or equal to the value."""

    def __call__(self, *args, **kwargs) -> bool:
        return args[0] >= self.value


@dataclass
class Lte(Compare):
    """Whether the first argument is less than or equal to the value."""

    def __call__(self, *args, **kwargs) -> bool:
        return args[0] <= self.value


@dataclass
class IsInstance(Compare):
    """Whether the first argument is an instance of the value."""

    def __call__(self, *args, **kwargs) -> bool:
        return isinstance(args[0], self.value)


@dataclass
class Binary(Predicate, ABC):
    left: Predicate
    right: Predicate

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.left)}, {repr(self.right)})"

    @classmethod
    def chain(cls, *predicates: Predicate) -> Predicate:
        """Create a chain of predicates from left to right."""
        if not predicates:
            predicate = Constant(True)
        else:
            predicate = predicates[0]
            for pred in predicates[1:]:
                predicate = cls(predicate, pred)
        return predicate


@dataclass
class And(Binary):
    def __call__(self, *args, **kwargs) -> bool:
        return self.left(*args, **kwargs) and self.right(*args, **kwargs)


@dataclass
class Or(Binary):
    def __call__(self, *args, **kwargs) -> bool:
        return self.left(*args, **kwargs) or self.right(*args, **kwargs)


_TValue = TypeVar("_TValue")


@dataclass
class Pipe(Generic[_TValue], Predicate, ABC):
    value: _TValue
    output: Predicate = None

    def __or__(self, other: Union[Predicate, Callable]) -> Self:
        # We transform all to reduce, some pipes implement custom calls
        # others just provide a callable function.

        # We convert functions into plain reduce pipes.
        if not isinstance(other, Predicate) and callable(other):
            other = Reduce(other)

        # Copy every Pipe node down to the open tail: a head-only copy would
        # share its tail with ``self``, and assigning into that shared node
        # silently extended the original chain.
        head = node = self.__class__(self.value, self.output)

        while isinstance(node, Pipe):
            if node.output is None:
                node.output = other
                return head

            if isinstance(node.output, Pipe):
                node.output = node.output.__class__(
                    node.output.value, node.output.output
                )

            node = node.output

        raise TypeError(f"Cannot reduce {self} with {other}")

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.value)}, {repr(self.output)})"


@dataclass
class Reduce(Pipe[Callable]):
    """Reduces the input to a single value."""

    def __call__(self, *args, **kwargs) -> bool:
        return self.output(self.value(*args, **kwargs))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (
            _callables_equivalent(self.value, other.value)
            and self.output == other.output
        )


@dataclass
class Extract(Pipe[PropertyPath]):
    """Extracts a property from the first argument and evaluates it with the predicate."""

    def __init__(
        self, value: Union[PropertyPath, PropertyPathBuilder], output: Predicate = None
    ):
        if isinstance(value, PropertyPathBuilder):
            value = as_path(value)

        super().__init__(value, output)

    def __call__(self, *args, **kwargs) -> bool:
        try:
            return self.output(self.value.resolve(args[0]))
        except (AttributeError, KeyError, IndexError):
            return False


@dataclass
class Sel(Pipe[Indexable]):
    """Select either argument by index or kwarg by name."""

    def __call__(self, *args, **kwargs) -> bool:
        try:
            if isinstance(self.value, str):
                return self.output(kwargs[self.value])

            return self.output(args[self.value])
        except (KeyError, IndexError):
            return False


@dataclass
class EmptyPipe(Pipe[None]):
    def __init__(self, value=None, output: Predicate = None):
        super().__init__(value, output)

    def __call__(self, *args, **kwargs) -> bool:
        return self.output(*args, **kwargs)
