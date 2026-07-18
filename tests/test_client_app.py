import threading
import time

import pytest
from unittest.mock import AsyncMock

from simplyprint_ws_client import (
    ClientSettings,
    IntegrationId,
    IntegrationSpec,
    Client,
    PrinterConfig,
    ClientApp,
)
from simplyprint_ws_client.integration.spec import ProductMetadata
from simplyprint_ws_client.integration.discovery import DiscoveryService
from simplyprint_ws_client.core.client import (
    ClientConfigChangedEvent,
    ClientStateChangeEvent,
)
from simplyprint_ws_client.core.api.url_builder import SimplyPrintBackend
from simplyprint_ws_client.integration.camera.pool import (
    DEFAULT_CAMERA_PROCESS_WORKERS,
)
from tests._fakes import FakeTransport


class FirstClient(Client[PrinterConfig]): ...


class SecondClient(Client[PrinterConfig]): ...


class SecondConfig(PrinterConfig):
    marker: str | None = None


_METADATA = ProductMetadata(
    display_name="Test",
    image_url="/test.png",
    supported_transports=(),
    capabilities=(),
)


def _integration(integration_id, client_factory, config_factory):
    return IntegrationSpec(
        id=IntegrationId(integration_id),
        client_factory=client_factory,
        config_factory=config_factory,
        metadata=_METADATA,
    )


def _fixture_transport_factory(url, provider, logger):
    """A real protocol transport boundary with a deterministic server script."""
    transport = FakeTransport(url, provider, logger, first_message_timeout=1.0)
    transport.queue_message(
        '{"type":"new_token","data":{"token":"fixture-token","short_id":"TEST"}}'
    )
    transport.queue_message(
        '{"type":"connected","data":'
        '{"region":"fixture","short_id":"TEST","in_setup":true}}'
    )
    return transport


def _wait_for_client(client, predicate, description, timeout=5.0):
    """Wait on client lifecycle events, with the timeout only as a deadlock guard."""
    changed = threading.Event()
    client.event_bus.on(ClientConfigChangedEvent, changed.set)
    client.event_bus.on(ClientStateChangeEvent, changed.set)
    deadline = time.monotonic() + timeout
    try:
        while True:
            # Clear before observing state: an edge immediately after this check
            # remains set, while an edge immediately before it is reflected by
            # the predicate. No lifecycle transition can be missed.
            changed.clear()
            if predicate():
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0 or not changed.wait(remaining):
                pytest.fail(
                    f"client did not reach {description}; "
                    f"state={client.state!r}, short_id={client.config.short_id!r}, "
                    f"active={client.active!r}"
                )
    finally:
        client.event_bus.off(ClientConfigChangedEvent, changed.set)
        client.event_bus.off(ClientStateChangeEvent, changed.set)


@pytest.fixture
def app():
    settings = ClientSettings(
        integrations=(_integration("test", Client, PrinterConfig),),
        camera_workers=0,
    )

    app = ClientApp(
        settings,
        discovery_service=DiscoveryService(),
        account_providers={},
        transport_factory=_fixture_transport_factory,
    )
    app.run_detached()
    yield app
    app.stop()


def test_virtual_client(app: ClientApp):
    config = PrinterConfig.get_new()
    client = app.add(config)
    _wait_for_client(
        client,
        lambda: client.is_added() and client.config.short_id == "TEST",
        "the scripted connected handshake",
    )
    assert client.config.token == "fixture-token"


def test_single_connection_active_flag(app: ClientApp):
    config = PrinterConfig.get_new()
    client = app.add(config)
    _wait_for_client(
        client,
        lambda: client.is_added() and client.config.short_id == "TEST",
        "the scripted connected handshake",
    )
    assert client.config.token == "fixture-token"
    assert client.is_added(), "Client should be added after being activated."

    client.active = False
    _wait_for_client(client, client.is_removed, "the deallocated state")
    assert not client.is_added(), "Client should be removed after being deactivated."

    client.active = True
    _wait_for_client(client, client.is_added, "the reconnected state")
    assert client.is_added(), "Client should be added after being re-activated."


def test_multiple_integrations_require_explicit_routing():
    settings = ClientSettings(
        name="multi",
        integrations=(
            _integration("first", FirstClient, PrinterConfig),
            _integration("second", SecondClient, SecondConfig),
        ),
    )
    app = ClientApp(
        settings, discovery_service=DiscoveryService(), account_providers={}
    )

    try:
        first_config = PrinterConfig.get_new()
        second_config = SecondConfig.get_new()

        first_client = app.add(first_config, integration_id="first")
        second_client = app.add(second_config, integration_id="second")

        assert isinstance(first_client, FirstClient)
        assert isinstance(second_client, SecondClient)
        assert app.get_config_manager(integration_id="first").contains(first_config)
        assert app.get_config_manager(integration_id="second").contains(second_config)
        assert app.config_manager is app.get_config_manager(integration_id="first")
    finally:
        app.stop()


def test_factory_receives_only_its_late_started_background_service():
    received = []

    def factory(config, *, context):
        received.append(context)
        return Client(config, context=context)

    settings = ClientSettings(
        integrations=(_integration("test", factory, PrinterConfig),)
    )
    services = {}
    app = ClientApp(
        settings,
        discovery_service=DiscoveryService(),
        account_providers={},
        background_services=services,
    )
    service = object()
    services["test"] = service

    try:
        app.add(PrinterConfig.get_new())

        assert app.client_context.background_service is None
        assert len(received) == 1
        assert received[0].background_service is service
    finally:
        app.stop()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(0, DEFAULT_CAMERA_PROCESS_WORKERS), (1, 1)],
)
def test_camera_worker_setting_controls_process_budget(configured, expected):
    app = ClientApp(
        ClientSettings(
            integrations=(_integration("test", Client, PrinterConfig),),
            camera_workers=configured,
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    try:
        assert app.camera_pool is not None
        assert app.camera_pool.process_workers == expected
    finally:
        app.stop()


def test_config_change_triggers_the_flusher_not_an_inline_flush():
    settings = ClientSettings(
        integrations=(_integration("test", Client, PrinterConfig),)
    )
    app = ClientApp(
        settings, discovery_service=DiscoveryService(), account_providers={}
    )
    try:
        config = PrinterConfig.get_new()
        integration = app._get_integration()
        manager = app.config_managers[str(integration.id)]

        direct = {"n": 0}
        real_flush = manager.flush

        def counting_flush(cfg=None):
            direct["n"] += 1
            return real_flush(cfg)

        manager.flush = counting_flush

        client = app.add(config)
        assert direct["n"] == 1  # add() flushes directly (registration-critical)

        flusher = app.config_flushers[str(integration.id)]
        triggered = {"n": 0}
        flusher.trigger = lambda: triggered.__setitem__("n", triggered["n"] + 1)

        client.event_bus.emit_sync(ClientConfigChangedEvent)
        assert triggered["n"] == 1  # the change path triggers the coalesced flusher
        assert direct["n"] == 1  # and does not flush inline again
    finally:
        app.stop()


def test_multiple_integrations_require_id_even_for_the_same_config_type():
    settings = ClientSettings(
        name="multi",
        integrations=(
            _integration("first", FirstClient, PrinterConfig),
            _integration("second", SecondClient, PrinterConfig),
        ),
    )
    app = ClientApp(
        settings, discovery_service=DiscoveryService(), account_providers={}
    )

    try:
        config = PrinterConfig.get_new()

        with pytest.raises(ValueError):
            app.add(config)

        assert isinstance(app.add(config, integration_id="second"), SecondClient)
    finally:
        app.stop()


def test_apps_own_and_close_transport_registries_once():
    settings = ClientSettings(
        integrations=(_integration("test", Client, PrinterConfig),),
        camera_workers=None,
    )
    first = ClientApp(
        settings, discovery_service=DiscoveryService(), account_providers={}
    )
    second = ClientApp(
        settings, discovery_service=DiscoveryService(), account_providers={}
    )
    assert first.client_context.mqtt_pools is not second.client_context.mqtt_pools
    assert (
        first.client_context.websocket_pools
        is not second.client_context.websocket_pools
    )

    first_mqtt_close = AsyncMock(wraps=first.client_context.mqtt_pools.close)
    first_ws_close = AsyncMock(wraps=first.client_context.websocket_pools.close)
    second_mqtt_close = AsyncMock(wraps=second.client_context.mqtt_pools.close)
    first.client_context.mqtt_pools.close = first_mqtt_close
    first.client_context.websocket_pools.close = first_ws_close
    second.client_context.mqtt_pools.close = second_mqtt_close

    try:
        first.run_detached()
        first.stop()
        first.stop()

        first_mqtt_close.assert_awaited_once_with()
        first_ws_close.assert_awaited_once_with()
        second_mqtt_close.assert_not_awaited()
    finally:
        first.stop()
        second.stop()


def test_stop_closes_never_started_loop_and_registries_once():
    app = ClientApp(
        ClientSettings(
            integrations=(_integration("test", Client, PrinterConfig),),
            camera_workers=None,
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    loop = app._app_event_loop
    mqtt_close = AsyncMock(wraps=app.client_context.mqtt_pools.close)
    websocket_close = AsyncMock(wraps=app.client_context.websocket_pools.close)
    app.client_context.mqtt_pools.close = mqtt_close
    app.client_context.websocket_pools.close = websocket_close

    app.stop()
    app.stop()

    assert loop.is_closed()
    mqtt_close.assert_awaited_once_with()
    websocket_close.assert_awaited_once_with()


def test_immediate_stop_does_not_deadlock_startup_config_replay():
    app = ClientApp(
        ClientSettings(
            integrations=(_integration("test", Client, PrinterConfig),),
            camera_workers=None,
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    config = PrinterConfig.get_new()
    app.config_manager.persist(config)

    replay_ready = threading.Event()
    allow_replay = threading.Event()
    stop_has_app_lock = threading.Event()
    replay_lock_timed_out = threading.Event()
    stop_finished = threading.Event()
    stop_errors = []
    real_get_all = app.config_manager.get_all
    real_add = app.add

    class ObservedLock:
        """Expose when the stopper owns the app lock to order the regression."""

        def __init__(self, lock):
            self._lock = lock

        def acquire(self, blocking=True, timeout=-1):
            acquired = self._lock.acquire(blocking, timeout)
            if acquired and threading.current_thread().name == "app-stopper":
                stop_has_app_lock.set()
            return acquired

        def release(self):
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *args):
            self.release()

    app._app_lock = ObservedLock(app._app_lock)

    def gated_get_all():
        replay_ready.set()
        if not allow_replay.wait(5.0):
            raise AssertionError("startup replay was never released")
        return real_get_all()

    def guarded_add(*args, **kwargs):
        # The timeout is only a deadlock guard. On the correct path this probe
        # acquires immediately because stop released the app lock before join.
        if not app._app_lock.acquire(timeout=1.0):
            replay_lock_timed_out.set()
            return None
        app._app_lock.release()
        return real_add(*args, **kwargs)

    def stop_app():
        try:
            app.stop()
        except BaseException as error:
            stop_errors.append(error)
        finally:
            stop_finished.set()

    app.config_manager.get_all = gated_get_all
    app.add = guarded_add
    stopper = threading.Thread(target=stop_app, name="app-stopper")

    try:
        app.run_detached()
        assert replay_ready.wait(5.0)

        stopper.start()
        assert stop_has_app_lock.wait(5.0)
        allow_replay.set()

        assert stop_finished.wait(5.0)
        stopper.join()
        assert stop_errors == []
        assert not replay_lock_timed_out.is_set()
        assert config.unique_id in app.client_list
        assert app._app_event_loop.is_closed()
    finally:
        allow_replay.set()
        if stopper.is_alive():
            stopper.join(5.0)
        app.stop()


@pytest.mark.asyncio
async def test_two_apps_isolate_backend_api_websocket_and_telemetry(monkeypatch):
    import simplyprint_ws_client.core.app as app_module

    readings = iter(({"cpu": 11}, {"cpu": 22}))

    def telemetry_factory():
        reading = next(readings)

        async def read():
            return reading

        return read

    monkeypatch.setattr(app_module, "make_host_telemetry_reader", telemetry_factory)
    integration = _integration("test", Client, PrinterConfig)
    first = ClientApp(
        ClientSettings(
            integrations=(integration,),
            backend=SimplyPrintBackend.PRODUCTION,
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    second = ClientApp(
        ClientSettings(
            integrations=(integration,),
            backend=SimplyPrintBackend.STAGING,
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    first_loop = first._app_event_loop
    second_loop = second._app_event_loop

    try:
        assert first.simplyprint_api is first.client_context.simplyprint_api
        assert second.simplyprint_api is second.client_context.simplyprint_api
        assert first.simplyprint_api is not second.simplyprint_api
        assert str(first.simplyprint_api.api_url) == "https://api.simplyprint.io"
        assert (
            str(second.simplyprint_api.api_url) == "https://apistaging.simplyprint.io"
        )
        assert str(first.scheduler.manager.websocket_base) == "wss://ws.simplyprint.io"
        assert (
            str(second.scheduler.manager.websocket_base)
            == "wss://wsstaging.simplyprint.io"
        )

        first_telemetry = first.client_context.host_telemetry
        second_telemetry = second.client_context.host_telemetry
        assert first_telemetry is not None and second_telemetry is not None
        assert first_telemetry is not second_telemetry
        assert await first_telemetry() == {"cpu": 11}
        assert await second_telemetry() == {"cpu": 22}
    finally:
        first.stop()
        second.stop()

    assert first_loop.is_closed()
    assert second_loop.is_closed()
