from simplyprint_ws_client import (
    Client,
    PrinterConfig,
    ClientSettings,
    ConfigManagerType,
    ClientApp,
)


class MyPrinterClient(Client[PrinterConfig]): ...


if __name__ == "__main__":
    client_settings = ClientSettings(
        MyPrinterClient,
        PrinterConfig,
        config_manager_t=ConfigManagerType.JSON,  # save the configuration to a JSON file
    )
    client_app = ClientApp(client_settings)

    # Check if we already have added a client.
    if len(client_app.config_manager.get_all()) > 0:
        my_config = client_app.config_manager.get_all()[0]
    else:
        my_config = PrinterConfig.get_new()

    my_client = client_app.add(my_config)
    print(my_client.config)
    client_app.run_blocking()
