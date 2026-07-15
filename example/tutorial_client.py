from simplyprint_ws_client import (
    Client,
    PrinterConfig,
    ClientSettings,
    ConfigManagerType,
    ClientApp,
    IntegrationId,
    IntegrationSpec,
)
from simplyprint_ws_client.integration.spec import ProductMetadata
from simplyprint_ws_client.integration.discovery import DiscoveryService


class MyPrinterClient(Client[PrinterConfig]): ...


if __name__ == "__main__":
    client_settings = ClientSettings(
        integrations=(
            IntegrationSpec(
                id=IntegrationId("tutorial"),
                client_factory=MyPrinterClient,
                config_factory=PrinterConfig,
                metadata=ProductMetadata(
                    display_name="Tutorial",
                    image_url="/tutorial.png",
                    supported_transports=(),
                    capabilities=(),
                ),
            ),
        ),
        config_manager_t=ConfigManagerType.JSON,  # save the configuration to a JSON file
    )
    client_app = ClientApp(
        client_settings,
        discovery_service=DiscoveryService(),
        account_providers={},
    )

    # Check if we already have added a client.
    if len(client_app.config_manager.get_all()) > 0:
        my_config = client_app.config_manager.get_all()[0]
    else:
        my_config = PrinterConfig.get_new()

    my_client = client_app.add(my_config)
    print(my_client.config)
    client_app.run_blocking()
