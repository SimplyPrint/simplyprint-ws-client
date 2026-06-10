from simplyprint_ws_client.cloud.autowire import AutowireClientMeta, configure, autowire
from simplyprint_ws_client.cloud.protocol.models import ServerMsgType, DemandMsgType


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
