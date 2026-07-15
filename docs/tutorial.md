# Build a printer integration

An integration has three explicit pieces:

1. a persisted `PrinterConfig` type;
2. a `PrinterClient` that translates device state and commands;
3. one immutable `IntegrationSpec` value that connects the two.

The following can live in a single `client.py` while an integration is small.

```python
from __future__ import annotations

from simplyprint_ws_client import (
    ClientApp,
    ClientContext,
    ClientSettings,
    ConfigManagerType,
    ConnectedMsg,
    FileDemandData,
    GcodeDemandData,
    IntegrationCapability,
    IntegrationId,
    IntegrationSpec,
    IntegrationTransport,
    PrinterConfig,
    PrinterStatus,
)
from simplyprint_ws_client.integration import DevicePoller, PrinterClient
from simplyprint_ws_client.integration.discovery import DiscoveryService
from simplyprint_ws_client.integration.spec import ProductMetadata


class MyPrinterConfig(PrinterConfig):
    address: str | None = None
    serial: str | None = None

    def hardware_identity(self) -> str | None:
        return self.serial

    def network_addresses(self) -> tuple[str, ...]:
        return (self.address,) if self.address else ()


class MyPrinterClient(PrinterClient[MyPrinterConfig]):
    def __init__(
        self,
        config: MyPrinterConfig,
        *,
        context: ClientContext,
    ) -> None:
        super().__init__(config, context=context)
        self.printer.set_info("My Printer", "0.1.0")
        self.printer.tool_count = 1
        # Replace this polling driver with WsDriver or MqttDriver for a device
        # that pushes state.
        self.driver = self.attach_driver(
            DevicePoller(self, interval=1.0, offline_after=10.0)
        )

    async def poll_device(self) -> None:
        # Read your device API here, then update the neutral printer state.
        self.printer.bed.temperature.actual = 22.0
        self.printer.tool0.temperature.actual = 24.0
        self.apply_status(PrinterStatus.OPERATIONAL)

    async def on_connected(self, _message: ConnectedMsg) -> None:
        print("Connected to SimplyPrint; setup code:", self.config.short_id)

    async def on_gcode(self, data: GcodeDemandData) -> None:
        print("Execute G-code on the device:", data.list)

    async def on_file(self, data: FileDemandData) -> None:
        print("Download and prepare this file:", data)


INTEGRATION = IntegrationSpec(
    id=IntegrationId("my-printer"),
    client_factory=MyPrinterClient,
    config_factory=MyPrinterConfig,
    metadata=ProductMetadata(
        display_name="My Printer",
        image_url="/img/my-printer.png",
        supported_transports=(IntegrationTransport.HTTP,),
        capabilities=(IntegrationCapability.FILE_UPLOAD,),
    ),
)
```

`PrinterClient` declares the typed command hooks (`on_gcode`, `on_file`,
`on_pause`, `on_resume`, `on_cancel`, and the other supported demands). The
runtime registers that fixed contract explicitly; arbitrary method names and
annotations do not create listeners.

Every device driver must be attached in the concrete client's constructor,
after `super().__init__()`, with `self.attach_driver(driver)`. Construction is
the only driver-registration phase: there is no `device_drivers()` discovery
hook and drivers added later miss the client's owned lifecycle. The read-only
`client.drivers` tuple is available for inspection; retain the return value from
`attach_driver()` when brand code needs to send through a particular driver.

## Run and persist the integration

Pass the integration value to `ClientSettings`. JSON storage keeps the same
printer slot across restarts.

```python
def main() -> None:
    settings = ClientSettings(
        integrations=(INTEGRATION,),
        name="my-printers",
        config_manager_t=ConfigManagerType.JSON,
    )
    app = ClientApp(
        settings,
        discovery_service=DiscoveryService(),
        account_providers={},
    )

    stored = app.config_manager.get_all()
    if stored:
        config = stored[0]
    else:
        config = MyPrinterConfig.get_new()
        config.address = "http://192.168.1.42"

    client = app.add(config)
    print(client.config)
    app.run_blocking()


if __name__ == "__main__":
    main()
```

With one integration, `app.add(config)` and `app.get_config_manager()` are
unambiguous. A process hosting several integrations must pass the exact id:

```python
app.add(config, integration_id="my-printer")
manager = app.get_config_manager(integration_id="my-printer")
```

No config-class or inheritance matching is performed.

## Push-based devices

For a device that publishes updates, attach a `WsDriver` or `MqttDriver` in
`__init__` exactly like the poller above, then implement the typed device edges:

```python
async def on_device_connected(self, driver) -> None:
    self.apply_status(PrinterStatus.OPERATIONAL)

async def on_device_disconnected(self, driver, reason) -> None:
    self.apply_status(PrinterStatus.OFFLINE)

async def on_device_message(self, message, driver) -> None:
    # Decode the brand payload and update self.printer.
    ...
```

For discovery, guided onboarding, cameras, accounts, or background services,
set the corresponding optional field on `IntegrationSpec`. See
[`example/headless_vendor.py`](../example/headless_vendor.py) for a complete
headless composition with discovery and an add-printer flow.
