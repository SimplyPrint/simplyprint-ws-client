import asyncio
import functools
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Tuple,
    Dict,
    TypeVar,
    Generic,
    Optional,
    NamedTuple,
    TYPE_CHECKING,
    Final,
)

from simplyprint_ws_client.common.events.event_bus_listeners import (
    EventBusListener,
    ListenerLifetimeForever,
)
from simplyprint_ws_client.common.events.event_bus_predicate_tree import (
    EventBusPredicateTree,
)
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.events.predicate import Predicate

if TYPE_CHECKING:
    from simplyprint_ws_client.common.events.event_bus import EventBus


class EventBusMiddleware(EventBusListener, ABC):
    """An event bus middleware is a synchronous listener that is always called before any other listener."""

    @classmethod
    def setup(cls, event_bus: "EventBus", *args, **kwargs):
        """Instantiate and ensure the middleware is added to the event bus."""
        instance = cls(*args, **kwargs)
        event_bus.middleware.add(instance)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(
            lifetime=ListenerLifetimeForever(**{}), priority=0, handler=self.handle
        )

        assert not self.is_async, "EventBusMiddleware must be synchronous."

    @abstractmethod
    def handle(self, *args, **kwargs):
        raise NotImplementedError()


class EventBusResponseMiddleware(EventBusMiddleware, EventLoopProvider, ABC):
    """Query events as responses by using asyncio wait primitives such as futures and queues"""

    def __init__(self, *args, **kwargs):
        EventLoopProvider.__init__(self, *args, **kwargs)
        EventBusMiddleware.__init__(self, *args, **kwargs)
        ABC.__init__(self)

    @staticmethod
    def _on_response(
        loop: asyncio.AbstractEventLoop, callback: Callable[[Tuple[Tuple, Dict]], None]
    ):
        """Bind a loop and a callback as the response to a future."""
        return functools.partial(loop.call_soon, callback)

    def _create_response_queue(self, maxsize=0):
        """Creates a queue that can be submitted to with the first
        function and closed with the second function which also
        invokes the cleanup function."""

        loop = self.event_loop
        queue = asyncio.Queue(maxsize=maxsize)

        return queue, self._on_response(loop, queue.put_nowait)

    def _create_response(self, cleanup: Callable[[asyncio.Future], None]):
        """Create a future that takes a cleanup function and returns its
        trigger function. This future can be awaited for the result given to
        the trigger function.
        """
        loop = self.event_loop
        future = loop.create_future()
        future.add_done_callback(cleanup)
        return future, self._on_response(loop, future.set_result)

    @abstractmethod
    def create_response(self, *args, **kwargs) -> asyncio.Future:
        """Creates a future that can be awaited for a response."""
        pass

    @abstractmethod
    async def wait_for_response(self, *args, **kwargs) -> Tuple[Tuple, Dict]:
        """Creates a future that can be awaited for a response. And waits for it."""
        pass

    @abstractmethod
    async def create_response_queue(
        self, *args, **kwargs
    ) -> Tuple[asyncio.Queue, Callable]:
        """Creates a queue that will be filled with responses."""
        pass


_THash = TypeVar("_THash")


class EventBusKeyResponseMiddleware(EventBusResponseMiddleware, Generic[_THash]):
    """Simple response middleware that passes all events through a hash function to map
    to a potential oneshot callback.
    """

    class _HashBucketEntry(NamedTuple):
        oneshot: bool
        callback: Callable

    hash_function: Final[Callable[..., _THash]]
    hash_bucket: Dict[_THash, _HashBucketEntry]

    def __init__(self, hash_function: Callable = hash, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hash_function = hash_function
        self.hash_bucket = {}

    def create_response(self, index: _THash) -> asyncio.Future:
        future, trigger = self._create_response(
            lambda _: self.hash_bucket.pop(index, None)
        )
        self.hash_bucket[index] = self._HashBucketEntry(True, trigger)
        return future

    async def wait_for_response(
        self, index: _THash, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[Tuple, Dict]:
        future = self.create_response(index)
        return await asyncio.wait_for(future, timeout=timeout)

    async def create_response_queue(
        self, index: _THash, maxsize=0
    ) -> Tuple[asyncio.Queue, Callable]:
        queue, trigger = self._create_response_queue(maxsize)
        self.hash_bucket[index] = self._HashBucketEntry(False, trigger)
        return queue, functools.partial(self.hash_bucket.pop, index, None)

    def handle(self, *args, **kwargs):
        index = self.hash_function(*args, **kwargs)

        if index is None:
            return

        entry = self.hash_bucket.get(index)

        if not entry:
            return

        entry.callback((args, kwargs))

        if entry.oneshot:
            self.hash_bucket.pop(index, None)


class EventBusPredicateResponseMiddleware(EventBusResponseMiddleware):
    """Allow for smart waiting of responses based on predicates that are evaluated on the event arguments."""

    predicate_tree: EventBusPredicateTree[Callable]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predicate_tree = EventBusPredicateTree()

    def create_response(self, *predicates: Predicate) -> asyncio.Future:
        """WARNING: not threadsafe."""
        resource_id = None
        future, trigger = self._create_response(
            lambda _: self.predicate_tree.remove_resource_id(resource_id)
        )
        resource_id = self.predicate_tree.add(trigger, *predicates)
        return future

    async def wait_for_response(
        self, *predicates: Predicate, timeout: Optional[float] = None, **kwargs
    ):
        future = self.create_response(*predicates)
        return await asyncio.wait_for(future, timeout=timeout)

    async def create_response_queue(self, *predicates: Predicate, maxsize=0, **kwargs):
        queue, trigger = self._create_response_queue(maxsize)
        resource_id = self.predicate_tree.add(trigger, *predicates)
        return queue, functools.partial(
            self.predicate_tree.remove_resource_id, resource_id
        )

    def handle(self, *args, **kwargs):
        for resource_id in self.predicate_tree.evaluate(*args, **kwargs):
            callback = self.predicate_tree.resources.get(resource_id)

            # TODO: Log when a resource id in the tree exists without an attached resource.
            if not callback:
                continue

            callback((args, kwargs))
