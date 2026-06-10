import pytest

from simplyprint_ws_client import (
    ClientSettings,
    ClientSpec,
    Client,
    PrinterConfig,
    ClientApp,
)


class FirstClient(Client[PrinterConfig]): ...


class SecondClient(Client[PrinterConfig]): ...


class SecondConfig(PrinterConfig):
    marker: str | None = None


@pytest.fixture
def app():
    settings = ClientSettings(
        Client,
        PrinterConfig,
        camera_workers=0,
    )

    app = ClientApp(settings)
    app.run_detached()
    yield app
    app.stop()


def test_virtual_client(app: ClientApp):
    config = PrinterConfig.get_new()
    client = app.add(config)
    app.wait(2)
    assert client.config.short_id is not None


def test_single_connection_active_flag(app: ClientApp):
    config = PrinterConfig.get_new()
    client = app.add(config)
    app.wait(2)
    assert client.config.short_id is not None
    assert client.is_added(), "Client should be added after being activated."

    client.active = False
    app.wait(2)
    assert not client.is_added(), "Client should be removed after being deactivated."

    client.active = True
    app.wait(2)
    assert client.is_added(), "Client should be added after being re-activated."


def test_multi_client_specs_route_configs_by_type():
    settings = ClientSettings(
        name="multi",
        client_specs=(
            ClientSpec("first", FirstClient, PrinterConfig),
            ClientSpec("second", SecondClient, SecondConfig),
        ),
    )
    app = ClientApp(settings)

    try:
        first_config = PrinterConfig.get_new()
        second_config = SecondConfig.get_new()

        first_client = app.add(first_config)
        second_client = app.add(second_config)

        assert isinstance(first_client, FirstClient)
        assert isinstance(second_client, SecondClient)
        assert app.get_config_manager(client_key="first").contains(first_config)
        assert app.get_config_manager(client_key="second").contains(second_config)
        assert app.config_manager is app.get_config_manager(client_key="first")
    finally:
        app.stop()


def test_multi_client_specs_require_key_for_ambiguous_config_type():
    settings = ClientSettings(
        name="multi",
        client_specs=(
            ClientSpec("first", FirstClient, PrinterConfig),
            ClientSpec("second", SecondClient, PrinterConfig),
        ),
    )
    app = ClientApp(settings)

    try:
        config = PrinterConfig.get_new()

        with pytest.raises(ValueError):
            app.add(config)

        assert isinstance(app.add(config, client_key="second"), SecondClient)
    finally:
        app.stop()
