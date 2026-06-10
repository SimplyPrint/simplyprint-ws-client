from typing import List, Hashable, cast, Any, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class _Attribute(str):
    """A part of a path - field name"""

    ...


class AnyIdx:
    """Special index sentinel that is reserved for derive.

    When used with resolve it will throw an KeyError unless defined.
    """

    __slots__ = []


Indexable = Union[Hashable, int, slice]
Shard = Union[_Attribute, Indexable]


class PropertyPath(object):
    """Representation of a property path where parts are called "Shards"

    Can be used to query objects for values. Immutable builder functions.
    """

    __slots__ = ("__path", "__hash")

    __path: List[Shard]
    __hash: int

    def __init__(self, path=None):
        self.__path = path or []
        self.__hash = hash(str(self))

    def __str__(self) -> str:
        return "".join(
            [
                f".{shard}" if isinstance(shard, _Attribute) else f"[{repr(shard)}]"
                for shard in self.__path
            ]
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {str(self)}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, PropertyPath):
            return False

        return self.__hash == other.__hash

    def __hash__(self) -> int:
        return self.__hash

    def as_path(self):
        return PropertyPath(self.__path)

    def pop(self, *args, **kwargs) -> Self:
        items = self.__path.copy()
        items.pop(*args, **kwargs)
        return self.__class__(items)

    def push(self, item) -> Self:
        items = self.__path.copy()
        items.append(item)
        return self.__class__(items)

    def resolve(self, current: object) -> Any:
        for shard in self.__path:
            if isinstance(shard, _Attribute):
                current = getattr(current, shard)
            else:
                current = current[shard]

        return current

    def attr(self, item: str) -> Self:
        return self.push(_Attribute(item))

    def idx(self, item: Indexable) -> Self:
        return self.push(item)


class PropertyPathBuilder(PropertyPath):
    """A path to a property, with builder functions.

    Allows for this object to be given to a function to replicate
    all the property accesses it received.
    """

    def __getattr__(self, item: str) -> Self:
        return super().push(_Attribute(item))

    def __getitem__(self, item: Indexable) -> Self:
        return super().push(item)


def as_path(builder: PropertyPathBuilder) -> PropertyPath:
    return cast(PropertyPath, super(type(builder), builder)).as_path()


p = PropertyPathBuilder()
