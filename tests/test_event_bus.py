import asyncio
from typing import Optional

import pytest

from simplyprint_ws_client.events.event import Event
from simplyprint_ws_client.events.event_bus import EventBus, EventBusListeners
from simplyprint_ws_client.events.event_bus_listeners import (
    ListenerUniqueness,
    ListenerLifetimeForever,
    ListenerLifetimeOnce,
)


class CustomEvent(Event):
    def __init__(self, data=None) -> None:
        self.data = data

    @classmethod
    def get_name(cls) -> str:
        return "custom"


class CustomEventBus(EventBus[CustomEvent]): ...


class DefaultEventBus(EventBus[Event]): ...


class ClientEvent(Event): ...


class ServerEvent(Event): ...


class ConnectEvent(Event):
    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


class EventBusTestHelper:
    def __init__(self):
        self.called_test = 0
        self.called_other = 0
        self.called_custom = 0
        self.called_always = 0

        self.custom_event_bus = CustomEventBus()
        self.custom_event_bus.on("test", self.on_test)
        self.custom_event_bus.on("other", self.on_other)
        self.custom_event_bus.on(CustomEvent, self.on_custom)
        self.custom_event_bus.on(CustomEvent, self.on_always, generic=True)

        self.default_event_bus = DefaultEventBus()
        self.default_event_bus.on(ClientEvent, self.on_client_event, generic=True)
        self.default_event_bus.on(ServerEvent, self.on_server_event, generic=True)

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

    await event_helper.custom_event_bus.emit("test", CustomEvent())
    assert event_helper.called_test == 1

    await event_helper.custom_event_bus.emit("other", CustomEvent())
    assert event_helper.called_other == 1

    await event_helper.custom_event_bus.emit("other", CustomEvent())
    await event_helper.custom_event_bus.emit("other", CustomEvent())

    assert event_helper.called_other == 3

    await event_helper.custom_event_bus.emit(CustomEvent())
    assert event_helper.called_custom == 1
    await event_helper.custom_event_bus.emit(CustomEvent())
    await event_helper.custom_event_bus.emit(CustomEvent())

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

    def func1(dispatcher: Optional[EventBus] = None):
        nonlocal called_func1
        called_func1 += 1

        dispatcher.emit_sync("func2")

    def func2():
        nonlocal called_func2
        called_func2 += 1

    event_bus = EventBus()
    event_bus.on("func1", func1)
    event_bus.on("func2", func2)

    await event_bus.emit("func1")

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

    def first(_event: CustomEvent):
        return "next"

    async def second(event: CustomEvent, value: str):
        got.append((event.get_name(), value))

    event_bus.on(CustomEvent, first, priority=10)
    event_bus.on(CustomEvent, second, priority=0)

    await event_bus.emit(CustomEvent())

    assert got == [("custom", "next")]


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

    event_bus.on("test", func1, lifetime=ListenerLifetimeOnce(**{}))

    await event_bus.emit("test")

    assert called == 1

    await event_bus.emit("test")

    assert called == 1

    assert len(event_bus.listeners["test"]) == 0


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

    event_bus.on("test", func1, lifetime=ListenerLifetimeOnce(**{}))

    fut = loop.create_future()

    await event_bus.emit("test", fut)

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
async def test_event_string_comparability_preserved():
    """An event (class or instance) still compares equal to its get_name()
    string, and explicitly string-keyed listeners still fire on string emits."""
    assert CustomEvent == "custom"
    assert CustomEvent() == "custom"
    assert ClientEvent == "ClientEvent"

    event_bus = EventBus()
    calls = []
    event_bus.on("custom", lambda *args: calls.append(args))

    await event_bus.emit("custom", 42)
    assert calls == [(42,)]


def test_event_listener_adding():
    event_listeners = EventBusListeners()

    def func1():
        pass

    def func2():
        pass

    event_listeners.add(
        func1,
        lifetime=ListenerLifetimeForever(**{}),
        priority=0,
        unique=ListenerUniqueness.NONE,
    )
    event_listeners.add(
        func2,
        lifetime=ListenerLifetimeForever(**{}),
        priority=0,
        unique=ListenerUniqueness.NONE,
    )

    assert len(event_listeners) == 2

    event_bus = EventBus()

    event_bus.on(CustomEvent, func1)
    event_bus.on(CustomEvent, func2, generic=True)

    assert len(event_bus.listeners[CustomEvent]) == 2


def test_listener_lifetime_variants_are_distinct():
    """The lifetime tagged-union variants must not compare equal across types
    (they used to, as empty NamedTuples)."""
    assert ListenerLifetimeOnce() != ListenerLifetimeForever()
    assert ListenerLifetimeForever() != ListenerLifetimeOnce()
    # Same-variant instances stay interchangeable.
    assert ListenerLifetimeOnce() == ListenerLifetimeOnce()
    assert ListenerLifetimeForever() == ListenerLifetimeForever()
