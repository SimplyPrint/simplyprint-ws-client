from simplyprint_ws_client.core.autowire import AutowireClientMeta, configure, autowire
from simplyprint_ws_client.core.protocol.messages import WebcamSnapshotDemandData
from simplyprint_ws_client.core.protocol.models import ServerMsgType, DemandMsgType


class AutowireClient1(metaclass=AutowireClientMeta):
    def __init__(self):
        self.event_bus = None

    def on_error(self):
        pass

    def on_pause(self):
        pass

    @configure(event="custom_event")
    def custom_handler(self):
        pass


class AutowireClient2(metaclass=AutowireClientMeta):
    def __init__(self):
        self.event_bus = None

    def on_connected(self):
        pass

    def regular_method(self):
        pass


class ClientWithArgs(metaclass=AutowireClientMeta):
    def __init__(self):
        self.event_bus = None

    def no_args_handler(self):
        pass

    def one_arg_handler(self, data):
        pass

    def multi_arg_handler(self, arg1, arg2):
        pass


def test_autoconfigure_on_server_message(client):
    """Test that on_* methods are auto-configured for server message types"""
    test_client = AutowireClient1()
    test_client.event_bus = client.event_bus

    autowire(test_client)

    # Check that the event bus has listeners for the auto-configured events
    assert ServerMsgType.ERROR in test_client.event_bus.listeners
    assert DemandMsgType.PAUSE in test_client.event_bus.listeners
    assert "custom_event" in test_client.event_bus.listeners


def test_autoconfigure_mixed_handlers(client):
    """Test mix of auto-configured and regular methods"""
    test_client = AutowireClient2()
    test_client.event_bus = client.event_bus

    autowire(test_client)

    # Should have listener for on_connected
    assert ServerMsgType.CONNECTED in test_client.event_bus.listeners
    # Should not have listener for regular_method
    assert "regular_method" not in test_client.event_bus.listeners


def test_autoconfigure_argument_handling():
    """Test autoconfiguration with different argument counts"""
    # No args should get _event_bus_wrap = True
    assert hasattr(ClientWithArgs.no_args_handler, "_event_bus_wrap")
    assert ClientWithArgs.no_args_handler._event_bus_wrap is True

    # One arg without type annotation should not be auto-configured
    assert not hasattr(ClientWithArgs.one_arg_handler, "_event_bus_event")

    # Multiple args should not be auto-configured
    assert not hasattr(ClientWithArgs.multi_arg_handler, "_event_bus_event")


def test_configure_decorator():
    """Test the configure decorator sets the right attributes"""

    @configure(event="test_event", priority=5)
    def test_func():
        pass

    assert test_func._event_bus_event == "test_event"
    assert test_func._event_bus_listeners_args["priority"] == 5


def test_metaclass_autoconfiguration():
    """Test that the metaclass auto-configures methods on class creation"""
    assert hasattr(AutowireClient1.on_error, "_event_bus_event")
    assert AutowireClient1.on_error._event_bus_event == ServerMsgType.ERROR

    assert hasattr(AutowireClient1.on_pause, "_event_bus_event")
    assert AutowireClient1.on_pause._event_bus_event == DemandMsgType.PAUSE

    assert hasattr(AutowireClient1.custom_handler, "_event_bus_event")
    assert AutowireClient1.custom_handler._event_bus_event == "custom_event"


class ClientWithPrivateHelpers(metaclass=AutowireClientMeta):
    """Private helpers must never be inferred into listeners by signature."""

    def __init__(self):
        self.event_bus = None

    def _helper_with_demand_annotation(self, data: WebcamSnapshotDemandData):
        # Returns a non-None value: were this registered, the event bus would
        # replace the emit args with it for every later listener.
        return object()

    @configure(ServerMsgType.STREAM_RECEIVED)
    def _explicit_private_handler(self): ...

    def on_webcam_snapshot(self, data=None): ...


def test_private_methods_are_not_inferred_as_listeners():
    """A private single-arg method annotated with a demand-data model is an
    implementation detail, not a listener. (Production bug: a private helper
    taking WebcamSnapshotDemandData was auto-registered, and its non-None
    return value replaced the demand data for the real handler.)"""
    helper = ClientWithPrivateHelpers._helper_with_demand_annotation
    assert getattr(helper, "_event_bus_event", None) is None


def test_explicitly_configured_private_handler_still_works():
    """@configure on a private method remains honored, including the zero-arg
    wrap that drops the event payload."""
    handler = ClientWithPrivateHelpers._explicit_private_handler
    assert handler._event_bus_event == ServerMsgType.STREAM_RECEIVED
    assert getattr(handler, "_event_bus_wrap", False) is True


def test_camera_mixin_private_cache_helper_is_not_a_listener():
    """The exact production regression: ClientCameraMixin._allowed_cache_age
    takes WebcamSnapshotDemandData and returns a timedelta; it must not be
    wired as a WEBCAM_SNAPSHOT listener."""
    from simplyprint_ws_client.integration.camera.mixin import ClientCameraMixin

    helper = ClientCameraMixin._allowed_cache_age
    assert getattr(helper, "_event_bus_event", None) is None
