__all__ = ["Event", "sync_only"]

from typing import ClassVar, Type


def sync_only(cls: Type["Event"]) -> Type["Event"]:
    """Mark an event as sync-only, this will prevent it from being listened to with an async handler."""
    if isinstance(cls, type) and not issubclass(cls, Event):
        raise TypeError("sync_only decorator can only be used on Event subclasses.")

    cls.synchronous_only = True
    return cls


class Event:
    """Canonical base for class-keyed events and propagation control."""

    __stopped: bool = False
    synchronous_only: ClassVar[bool] = False

    @classmethod
    def is_sync_only(cls) -> bool:
        return cls.synchronous_only

    # Allow for propagation control of events.
    def is_stopped(self) -> bool:
        return self.__stopped

    def stop_event(self) -> None:
        self.__stopped = True
