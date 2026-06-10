To begin using the client library to connect to SimplyPrint create a file `client.py` to define
your [client](concepts/client.md) class.

```python
from simplyprint_ws_client import *


class MyPrinterClient(Client[PrinterConfig]):
    ...
```

Every client has an associated configuration that is saved automatically you can either use the builtin `PrinterConfig`
class or extend it to store more data for your printer client, for now we will use the builtin config class.

To use your Printer Client class you must create a `ClientApp` instance, typically you would want this to also
be the entrypoint to your python script, so we put it in the `__main__` block (only run it when executed directly).

```python
from simplyprint_ws_client import *

...  # your client definition here

if __name__ == '__main__':
    client_settings = ClientSettings(MyPrinterClient, PrinterConfig)
    client_app = ClientApp(client_settings)
    client_app.run_blocking()
```

If you run your `client.py` file now, nothing happens! Use CTRL+C to stop the client.

We haven't yet added any client to our app instance. To add a client we called the `add` method on the `client_app` instance with a new configuration.


```python
from simplyprint_ws_client import *

...  # your client definition here

if __name__ == '__main__':
    client_settings = ClientSettings(MyPrinterClient, PrinterConfig)
    client_app = ClientApp(client_settings)
    my_client = client_app.add(PrinterConfig.get_new())
    print(my_client.config)
    client_app.run_blocking()
```

This now outputs a new empty configuration!

```bash
$ python client.py
PrinterConfig(id=0, token='0', name=None, in_setup=None, short_id=None, public_ip=None, unique_id='04be1d2a-248c-425d-a08f-d5892287b288')
CTRL+C
```

But there is an issue, everytime we restart the application we get a new configuration, we would really like to persist the configuration we create between runs. So lets do that.

```python
from simplyprint_ws_client import *

...  # your client definition here

if __name__ == '__main__':
    client_settings = ClientSettings(
        MyPrinterClient, 
        PrinterConfig, 
        config_manager_t=ConfigManagerType.JSON # save the configuration to a JSON file
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
```

Suddenly now our unique id is the same, and the config has some additional values it received from SimplyPrint on our second run.

```bash
$ python client.py
PrinterConfig(id=0, token='0', name=None, in_setup=None, short_id=None, public_ip=None, unique_id='08086eeb-812b-4bda-9159-99852aff509b')
CTRL+C
$ python client.py
PrinterConfig(id=0, token='595a482a-745f-41a6-88c7-1402cffa1e9c', name=None, in_setup=True, short_id='9ZZM', public_ip=None, unique_id='08086eeb-812b-4bda-9159-99852aff509b')
CTRL+C
```

Already at this point our printer client has been recognized by SimplyPrint, and we could follow the [setup guide](https://simplyprint.io/setup-guide) and add the printer to SimplyPrint with the `short_id` as the setup code.

But most likely we also want to be able to both send and receive some messages to be able to do something interesting. So lets extend our MyPrinterClient:

```python
from simplyprint_ws_client import *


class MyPrinterClient(Client[PrinterConfig]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set basic information
        self.printer.set_info('MyPrinterClient', "0.0.1")
        self.printer.nozzle_count = 1
        self.printer.material_count = 1
        self.printer.populate_info_from_physical_machine()

    def on_connected(self, msg: ConnectedMsg):
        print("Connected to SimplyPrint! Setup code is: ", msg.data.short_id)

    def on_gcode(self, data: GcodeDemandData):
        print("Got GCODE from SimplyPrint to run", data.list)

    def on_file(self, data: FileDemandData):
        print("Got file to start from SimplyPrint", data)

# main entrypoint code below
```

Here we have added some functions prefixed with `on_` that handle some common messages and demands from SimplyPrint, see [messages](concepts/messages.md) for an exhaustive overview.

How to extend this to be able to receive updates from my printer?
```python
from simplyprint_ws_client import *

# Custom logic to talk with printer in another class
class CustomPrinterConnection:
    def __init__(self, url):
        self.url = url
        
    ... # logic to talk with printer
    
    def get_printer_status(self) -> PrinterStatus:
        my_status = "custom_printing"
        
        if my_status == "custom_printing":
            return PrinterStatus.PRINTING
        else:
            return PrinterStatus.OPERATIONAL

    def get_bed_temperature(self) -> float:
        ...
    
    def get_bed_target_temperature(self) -> float:
        ...

# Custom config class that extends base config class
# to store custom data
class MyPrinterConfig(PrinterConfig):
    my_printer_url: str | None = None

# Extend client class to hold instance of custom printer connection
# and use the custom config.
class MyPrinterClient(Client[MyPrinterConfig]): # Specify custom config
    printer_connection: CustomPrinterConnection
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ... # other basic setup
        self.printer_connection = CustomPrinterConnection(self.config.my_printer_url)
    
    ... # messages handlers

    async def tick(self, _):
        # Update SimplyPrint printer state by setting fields on the printer-state field.
        self.printer.status = self.printer_connection.get_printer_status()
        self.printer.bed_temperature.actual = self.printer_connection.get_bed_temperature()
        self.printer.bed_temperature.target = self.printer_connection.get_bed_target_temperature()
        ... # Etc.

# Updated to use MyPrinterConfig instead of generic config.
if __name__ == '__main__':
    client_settings = ClientSettings(
        MyPrinterClient,
        MyPrinterConfig,
        config_manager_t=ConfigManagerType.JSON
    )
    client_app = ClientApp(client_settings)

    if len(client_app.config_manager.get_all()) > 0:
        my_config = client_app.config_manager.get_all()[0]
    else:
        my_config = MyPrinterConfig.get_new()
        my_config.my_printer_url = "per printer setup here"

    my_client = client_app.add(my_config)
    client_app.run_blocking()
```

We have added a lot of logic, but it all boils down to we now connect our `CustomPrinterConnection` with the SimplyPrint `MyPrinterClient` class and store some extra information we can use on start up in `MyPrinterConfig`
