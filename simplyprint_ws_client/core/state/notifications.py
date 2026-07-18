"""The notification center: persistent printer events relayed to SimplyPrint."""

import asyncio
import datetime
import uuid
from typing import (
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

from pydantic import Field, PrivateAttr

from simplyprint_ws_client.core.state.models import (
    NotificationEventActions,
    NotificationEventEffect,
    NotificationEventSeverity,
    NotificationEventType,
)
from simplyprint_ws_client.common.model.reactive import ReactiveModel
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider

if TYPE_CHECKING:
    from simplyprint_ws_client.core.protocol.messages import (
        ResolveNotificationDemandData,
    )

try:
    from typing import Unpack, TypedDict
except ImportError:
    from typing_extensions import Unpack, TypedDict


class NotificationEventPayload(ReactiveModel):
    title: Optional[str] = None
    message: Optional[str] = None
    url: Optional[str] = None
    image_url: Optional[str] = None
    effect: Optional[NotificationEventEffect] = None
    actions: Optional[Dict[str, NotificationEventActions]] = None  # action_id -> action
    data: Optional[dict] = None  # any data to attach to the event


class NotificationEvent(ReactiveModel):
    event_id: Optional[uuid.UUID] = None
    type: NotificationEventType = NotificationEventType.GENERIC
    severity: NotificationEventSeverity = NotificationEventSeverity.INFO
    payload: NotificationEventPayload = Field(default_factory=NotificationEventPayload)
    issued_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    resolved_at: Optional[datetime.datetime] = None

    # Integrated response handler, as an alternative pattern to implementing
    # a large on_resolve_notification handler.
    _response_future: Optional[asyncio.Future] = PrivateAttr(None)

    def __await__(self):
        return self.wait_for_response().__await__()

    async def wait_for_response(
        self, timeout: Optional[float] = None
    ) -> Optional["ResolveNotificationDemandData"]:
        ctx = self.ctx()
        if not ctx or not isinstance(ctx, EventLoopProvider):
            raise RuntimeError("NotificationEvent is not bound to a Client context.")
        self._response_future = ctx.event_loop.create_future()
        try:
            return await asyncio.wait_for(self._response_future, timeout)
        except asyncio.TimeoutError:
            return None

    def resolve(self, when: Optional[datetime.datetime] = None):
        """Mark the event as resolved."""
        self.resolved_at = when or datetime.datetime.now(datetime.timezone.utc)

    def respond(self, data: Optional["ResolveNotificationDemandData"]):
        if not self._response_future or self._response_future.done():
            return
        ctx = self.ctx()
        if not ctx or not isinstance(ctx, EventLoopProvider):
            raise RuntimeError("NotificationEvent is not bound to a Client context.")
        ctx.event_loop.call_soon(self._response_future.set_result, data)


class NotificationEventKwargs(TypedDict, total=False):
    """The keyword arguments accepted when building a :class:`NotificationEvent`.

    Mirrors the model's fields so ``**kwargs`` call sites can be checked with
    :data:`typing.Unpack` (a pydantic model itself is not a valid ``Unpack``
    operand).
    """

    event_id: Optional[uuid.UUID]
    type: NotificationEventType
    severity: NotificationEventSeverity
    payload: NotificationEventPayload
    issued_at: datetime.datetime
    resolved_at: Optional[datetime.datetime]


class NotificationsState(ReactiveModel):
    """Notification and event center for printer, relayed to SimplyPrint"""

    notifications: Dict[uuid.UUID, NotificationEvent] = Field(default_factory=dict)

    # Map hashable objects to an event for persistent references.
    __idempotency_keys: Dict[Hashable, uuid.UUID] = PrivateAttr(default_factory=dict)

    def new(self, **kwargs: Unpack[NotificationEventKwargs]) -> NotificationEvent:
        """Create a new unmanaged persistent (with event_id) notification event which can be updated and resolved async"""
        if "event_id" not in kwargs:
            kwargs["event_id"] = uuid.uuid4()
        event = NotificationEvent(**kwargs)
        event.provide_context(self)
        event.model_set_changed(
            "issued_at"
        )  # mark as changed when we create the event.
        self.notifications[event.event_id] = event
        return event

    def keyed(
        self,
        idempotency_key: Hashable,
        **kwargs: Unpack[NotificationEventKwargs],
    ) -> NotificationEvent:
        """Create new managed persistent reference to event, which is tied to some external object state (args)"""
        if "event_id" in kwargs:
            raise TypeError(
                "keyed() does not accept 'event_id'; it is managed by the idempotency key"
            )
        with self:
            if (
                event_id := self.__idempotency_keys.get(idempotency_key)
            ) and event_id in self.notifications:
                event = self.notifications[event_id]
                if event.resolved_at is None:
                    return event

            event = self.new(**kwargs)
            self.__idempotency_keys[idempotency_key] = event.event_id
            return event

    def rekey(self, idempotency_key: Hashable, event_id: uuid.UUID):
        if event_id not in self.notifications:
            return
        with self:
            self.__idempotency_keys[idempotency_key] = event_id

    def keys(self) -> List[Hashable]:
        """Return a (copied) list of all keys currently in use."""
        with self:
            return list(self.__idempotency_keys.keys())

    def retain(self, *uuids: uuid.UUID, remove_immediately=False):
        """Clean up all events except the ones provided."""
        for key in list(self.notifications.keys()):
            if key in uuids:
                continue
            if remove_immediately:
                self.remove(key)
                continue
            if key in self.notifications:
                self.notifications[key].resolve()

    def retain_keys(self, *keys: Hashable, remove_immediately=False):
        """
        Retain only the notifications with the given keys, remove all others.
        Useful when downstream APIs mark resolved events as no longer appearing.
        """
        for key in self.keys():
            if key in keys:
                continue
            if remove_immediately:
                self.remove(key)
                continue
            with self:
                uuid_key = self.__idempotency_keys.get(key)
            if uuid_key and uuid_key in self.notifications:
                self.notifications[uuid_key].resolve()

    def filter_retain_keys(
        self,
        func: Callable[[Hashable], bool],
        *keys: Hashable,
        remove_immediately=False,
    ):
        """
        Retain all keys, except for the subset of keys where func(key) is True - your kept keys.
        Useful when having multiple basis types for keys and wanting to retain only a subset.
        """
        all_keys = self.keys()
        all_keys_without_subset = set(all_keys) - set(k for k in all_keys if func(k))
        all_keys_without_subset = all_keys_without_subset.union(
            set(keys)
        )  # Add the explicit keys to retain
        self.retain_keys(
            *all_keys_without_subset, remove_immediately=remove_immediately
        )

    def clear(self, remove_immediately=False):
        """Reset all notifications"""
        self.retain(remove_immediately=remove_immediately)

    def get(self, key: Union[Hashable, uuid.UUID]) -> Optional[NotificationEvent]:
        """Get the notification with the given key."""
        if not isinstance(key, uuid.UUID):
            with self:
                key = self.__idempotency_keys.get(key)

        return self.notifications.get(key)

    def remove(self, key: Union[Hashable, uuid.UUID]):
        """Remove the notification with the given key."""
        if not isinstance(key, uuid.UUID):
            with self:
                key = self.__idempotency_keys.pop(key, None)
        else:
            # If we are removing an uuid, also remove all keys pointing to it.
            with self:
                for k, v in list(self.__idempotency_keys.items()):
                    if v != key:
                        continue
                    self.__idempotency_keys.pop(k, None)

        self.notifications.pop(key, None)

    def __contains__(self, item) -> bool:
        if isinstance(item, NotificationEvent):
            return item in self.notifications.values()

        if isinstance(item, bytes):
            item = item.hex()

        return item in self.notifications.keys()
