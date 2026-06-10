import asyncio
from typing import Callable, Optional, TypeVar, Generic

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

TEventLoop = TypeVar("TEventLoop", bound=asyncio.AbstractEventLoop)
TEventLoopFactory = Callable[[], TEventLoop]


class EventLoopProvider(Generic[TEventLoop]):
    """Manages an event loop for external access

    Useful when you have a class that uses an event loop
    to perform its tasks, and you want to invoke it from
    another loop or thread.
    """

    __event_loop: Optional[TEventLoop]
    __event_loop_factory: Optional[TEventLoopFactory]

    def __init__(
        self,
        loop: Optional[TEventLoop] = None,
        factory: Optional[TEventLoopFactory] = None,
        provider: Optional[Self] = None,
        **kwargs,
    ):
        # TODO: raise an error if no loops are provided / no default is used?
        self.__event_loop = loop
        self.__event_loop_factory = factory or (
            (lambda *args, **_: provider.event_loop) if provider is not None else None
        )

    def use_running_loop(self):
        self.__event_loop = asyncio.get_running_loop()

    def use_existing_loop(self):
        self.__event_loop = asyncio.get_event_loop()

    def use_new_loop(self):
        self.__event_loop = asyncio.new_event_loop()

    def reset_event_loop(self):
        self.__event_loop = None

    def event_loop_is_running(self):
        if self.__event_loop is None and self.__event_loop_factory is None:
            return False

        try:
            loop = self.event_loop
            return loop.is_running()
        except RuntimeError:
            return False

    def event_loop_is_not_closed(self) -> bool:
        if self.__event_loop is None and self.__event_loop_factory is None:
            return False

        try:
            loop = self.event_loop
            return loop.is_closed()
        except RuntimeError:
            return False

    @property
    def event_loop(self) -> TEventLoop:
        if not self.__event_loop and self.__event_loop_factory:
            self.__event_loop = self.__event_loop_factory()

        if not self.__event_loop:
            raise RuntimeError("No event loop available")

        return self.__event_loop

    @event_loop.setter
    def event_loop(self, loop: TEventLoop) -> None:
        self.__event_loop = loop

    @staticmethod
    def default():
        """Returns an EventLoopProvider that uses the running loop."""
        return EventLoopProvider(factory=asyncio.get_running_loop)
