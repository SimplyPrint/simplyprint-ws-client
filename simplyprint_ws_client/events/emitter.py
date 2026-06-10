from abc import ABC, abstractmethod
from typing import Union, Hashable, TypeVar, Generic

TEvent = TypeVar("TEvent", bound=object)


class Emitter(Generic[TEvent], ABC):
    @abstractmethod
    async def emit(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None: ...

    @abstractmethod
    def emit_sync(self, event: Union[Hashable, TEvent], *args, **kwargs) -> None: ...
