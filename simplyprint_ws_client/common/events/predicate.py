from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Generic, TypeVar, Callable, Any

from simplyprint_ws_client.common.events.property_path import (
    PropertyPath,
    PropertyPathBuilder,
    as_path,
    Indexable,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


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

        original = output = self.__class__(self.value, self.output)

        while isinstance(output, Pipe):
            if output.output is None:
                output.output = other
                return original

            output = output.output

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

        # If the callable objects are not strictly equal, we compare their code objects.
        if (
            self.value != other.value
            and (
                hasattr(self.value, "__code__")
                and hasattr(self.value.__code__, "co_code")
            )
            and (
                hasattr(other.value, "__code__")
                and hasattr(other.value.__code__, "co_code")
            )
        ):
            return (
                self.output == other.output
                and self.value.__code__.co_code == other.value.__code__.co_code
            )

        return self.value == other.value and self.output == other.output


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
