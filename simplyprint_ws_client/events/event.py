__all__ = ["Event", "sync_only"]

from typing import Type, Union, ClassVar


class EventTraits:
    def __eq__(self: Union[Type["Event"], "Event"], other: object) -> bool:
        if isinstance(other, str):
            return self.get_name() == other

        if isinstance(other, Event) or isinstance(other, EventType):
            return self.get_name() == other.get_name()

        return self is other

    def __str__(self: Union[Type["Event"], "Event"]) -> str:
        return self.get_name()

    def __hash__(self: Union[Type["Event"], "Event"]) -> int:
        return hash(
            self.__name__ if isinstance(self, type) else self.__class__.__name__
        )


class EventType(EventTraits, type):
    @classmethod
    def get_name(cls):
        raise NotImplementedError()


def sync_only(cls: Type["Event"]) -> Type["Event"]:
    """Mark an event as sync-only, this will prevent it from being listened to with an async handler."""
    if isinstance(cls, type) and not issubclass(cls, Event):
        raise TypeError("sync_only decorator can only be used on Event subclasses.")

    cls._Event__sync_only = True
    return cls


class Event(EventTraits, metaclass=EventType):
    """
    Base event class for type-hinting, not required to be used.
    """

    __stopped: bool = False
    __sync_only: ClassVar[bool] = False

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @classmethod
    def is_sync_only(cls) -> bool:
        return cls.__sync_only

    # Allow for propagation control of events.
    def is_stopped(self) -> bool:
        return self.__stopped

    def stop_event(self) -> None:
        self.__stopped = True
