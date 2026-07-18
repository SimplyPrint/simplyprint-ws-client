import asyncio
from enum import Enum, auto
import pytest

from simplyprint_ws_client.events.event import Event
from simplyprint_ws_client.events.event_bus import EventBus, EventBusListeners
from simplyprint_ws_client.events.event_bus_listeners import (
    ListenerLifetime,
    ListenerUniqueness,
)


class CustomEvent(Event):
    def __init__(self, data=None) -> None:
        self.data = data


class CustomChildEvent(CustomEvent): ...


class ClientEvent(Event): ...


class ServerEvent(Event): ...


class ConnectEvent(ClientEvent):
    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


class EventKey(Enum):
    TEST = auto()
    OTHER = auto()
    FUNC1 = auto()
    FUNC2 = auto()


class EventBusTestHelper:
    def __init__(self):
        self.called_test = 0
        self.called_other = 0
        self.called_custom = 0
        self.called_always = 0

        self.custom_event_bus = EventBus()
        self.custom_event_bus.on(EventKey.TEST, self.on_test)
        self.custom_event_bus.on(EventKey.OTHER, self.on_other)
        self.custom_event_bus.on(CustomEvent, self.on_custom)
        self.custom_event_bus.on(CustomEvent, self.on_always)

        self.default_event_bus = EventBus()
        self.default_event_bus.on(ClientEvent, self.on_client_event)
        self.default_event_bus.on(ServerEvent, self.on_server_event)

    @staticmethod
    async def on_client_event(event: ClientEvent):
        if not isinstance(event, ClientEvent):
            raise Exception("Event is not a ClientEvent")

    @staticmethod
    async def on_server_event(event: ServerEvent):
        if not isinstance(event, ServerEvent):
            raise Exception("Event is not a ServerEvent")

    async def on_test(self, event: CustomEvent):
        self.called_test += 1

    def on_other(self, event: CustomEvent):
        self.called_other += 1

    def on_custom(self, event: CustomEvent):
        self.called_custom += 1

    def on_always(self, event: CustomEvent):
        self.called_always += 1


@pytest.fixture
def event_helper() -> EventBusTestHelper:
    return EventBusTestHelper()


@pytest.mark.asyncio
async def test_custom_event_bus(event_helper: EventBusTestHelper):
    assert len(event_helper.custom_event_bus.listeners[CustomEvent]) == 2

    await event_helper.custom_event_bus.emit(EventKey.TEST, CustomEvent())
    assert event_helper.called_test == 1

    await event_helper.custom_event_bus.emit(EventKey.OTHER, CustomEvent())
    assert event_helper.called_other == 1

    await event_helper.custom_event_bus.emit(EventKey.OTHER, CustomEvent())
    await event_helper.custom_event_bus.emit(EventKey.OTHER, CustomEvent())

    assert event_helper.called_other == 3

    await event_helper.custom_event_bus.emit(CustomEvent())
    assert event_helper.called_custom == 1
    await event_helper.custom_event_bus.emit(CustomEvent())
    await event_helper.custom_event_bus.emit(CustomEvent())

    assert event_helper.called_always == 3

    await event_helper.custom_event_bus.emit(CustomChildEvent())
    assert event_helper.called_custom == 3
    assert event_helper.called_always == 3


@pytest.mark.asyncio
async def test_default_event_bus(event_helper: EventBusTestHelper):
    await event_helper.default_event_bus.emit(ClientEvent())
    await event_helper.default_event_bus.emit(ConnectEvent("connected"))
    await event_helper.default_event_bus.emit(CustomEvent())


@pytest.mark.asyncio
async def test_chained_event_bus():
    called_func1 = 0
    called_func2 = 0
    event_bus = EventBus()

    def func1():
        nonlocal called_func1
        called_func1 += 1

        event_bus.emit_sync(EventKey.FUNC2)

    def func2():
        nonlocal called_func2
        called_func2 += 1

    event_bus.on(EventKey.FUNC1, func1)
    event_bus.on(EventKey.FUNC2, func2)

    await event_bus.emit(EventKey.FUNC1)

    assert called_func1 == 1
    assert called_func2 == 1


@pytest.mark.asyncio
async def test_event_bus_fast_path_preserves_priority_order():
    event_bus = EventBus()
    calls = []

    event_bus.on(CustomEvent, lambda _event: calls.append("low"), priority=0)
    event_bus.on(CustomEvent, lambda _event: calls.append("high"), priority=10)

    await event_bus.emit(CustomEvent())

    assert calls == ["high", "low"]


@pytest.mark.asyncio
async def test_event_bus_fast_path_preserves_return_chaining():
    event_bus = EventBus()
    got = []
    emitted_event = CustomEvent()

    def first(_event: CustomEvent):
        return "next"

    async def second(event: CustomEvent, value: str):
        got.append((event, value))

    event_bus.on(CustomEvent, first, priority=10)
    event_bus.on(CustomEvent, second, priority=0)

    await event_bus.emit(emitted_event)

    assert got == [(emitted_event, "next")]


@pytest.mark.asyncio
async def test_event_bus_fast_path_preserves_stop_event():
    event_bus = EventBus()
    calls = []

    def first(event: CustomEvent):
        calls.append("first")
        event.stop_event()

    def second(_event: CustomEvent):
        calls.append("second")

    event_bus.on(CustomEvent, first, priority=10)
    event_bus.on(CustomEvent, second, priority=0)

    await event_bus.emit(CustomEvent())

    assert calls == ["first"]


@pytest.mark.asyncio
async def test_one_shot_listener():
    event_bus = EventBus()
    called = 0

    def func1():
        nonlocal called
        called += 1

    event_bus.on(EventKey.TEST, func1, lifetime=ListenerLifetime.ONCE)

    await event_bus.emit(EventKey.TEST)

    assert called == 1

    await event_bus.emit(EventKey.TEST)

    assert called == 1

    assert len(event_bus.listeners[EventKey.TEST]) == 0


@pytest.mark.asyncio
async def test_one_shot_listener_ret():
    event_bus = EventBus()
    loop = event_bus.event_loop_provider.event_loop

    async def expensive_task():
        await asyncio.sleep(0.0)
        return 1337

    async def func1(f: asyncio.Future):
        task = loop.create_task(expensive_task())
        task.add_done_callback(lambda _: f.set_result(task.result()))

    event_bus.on(EventKey.TEST, func1, lifetime=ListenerLifetime.ONCE)

    fut = loop.create_future()

    await event_bus.emit(EventKey.TEST, fut)

    result = await fut

    assert result == 1337


@pytest.mark.asyncio
async def test_distinct_classes_with_same_name_do_not_share_listeners():
    """Two different Event classes with the same class NAME are distinct keys."""
    klass_a = type("DuplicateNamedEvent", (Event,), {})
    klass_b = type("DuplicateNamedEvent", (Event,), {})
    assert klass_a is not klass_b

    event_bus = EventBus()
    calls = []

    event_bus.on(klass_a, lambda _event: calls.append("a"))
    event_bus.on(klass_b, lambda _event: calls.append("b"))

    # Each class keeps its own listener list (no silent merge).
    assert len(event_bus.listeners[klass_a]) == 1
    assert len(event_bus.listeners[klass_b]) == 1

    await event_bus.emit(klass_a())
    assert calls == ["a"]

    await event_bus.emit(klass_b())
    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_event_does_not_alias_string_key():
    event = CustomEvent()
    assert event != "CustomEvent"
    assert CustomEvent != "CustomEvent"

    event_bus = EventBus()
    calls = []
    event_bus.on("CustomEvent", lambda *args: calls.append(args))

    await event_bus.emit(event)
    assert calls == []


def test_event_listener_adding():
    event_listeners = EventBusListeners()

    def func1():
        pass

    def func2():
        pass

    event_listeners.add(
        func1,
        lifetime=ListenerLifetime.FOREVER,
        priority=0,
        unique=ListenerUniqueness.NONE,
    )
    event_listeners.add(
        func2,
        lifetime=ListenerLifetime.FOREVER,
        priority=0,
        unique=ListenerUniqueness.NONE,
    )

    assert len(event_listeners) == 2

    event_bus = EventBus()

    event_bus.on(CustomEvent, func1)
    event_bus.on(CustomEvent, func2)

    assert len(event_bus.listeners[CustomEvent]) == 2


def test_listener_lifetime_variants_are_distinct():
    assert ListenerLifetime.ONCE is not ListenerLifetime.FOREVER
