"""A complete vendor integration, headless, from the library alone.

This file is the 2.0 acceptance proof: with ONLY ``simplyprint_ws_client``
installed, a vendor ships one module containing a config, a printer client, a
declarative spec, and a guided add-printer flow -- then runs discovery, drives
the flow programmatically (no web layer), persists the printer through the one
identity seam, and runs the fleet::

    .venv/bin/python -m example.headless_vendor

The simulated device polls instead of pushing (the ``DevicePoller`` seam); a
push device would declare a ``WsDeviceLink``/``MqttDeviceLink`` in
``device_drivers()`` instead and override ``on_device_message``. Set
``SPWS_DEMO_SECONDS`` to auto-stop (the demo default is 5; ``0`` blocks forever
like a real vendor ``main()`` would).
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Optional

from simplyprint_ws_client import ClientSettings, PrinterConfig, PrinterStatus
from simplyprint_ws_client.device.discovery.spec import SubnetScanSpec
from simplyprint_ws_client.integration import DevicePoller, PrinterClient
from simplyprint_ws_client.integration.flow import FlowError, FlowState, run_flow
from simplyprint_ws_client.integration.flow.recipes import standard_add_printer_flow
from simplyprint_ws_client.integration.spec import PrinterSpec, ProductMetadata, lazy
from simplyprint_ws_client.core.host import Host
from simplyprint_ws_client.core.registry import SpecRegistry

# 1. The config: the persisted record of one printer.


class VendorConfig(PrinterConfig):
    host: Optional[str] = None
    serial: Optional[str] = None

    def stable_hardware_id(self) -> Optional[str]:
        # The hardware-match id discovery correlates on (never the slot id).
        return self.serial


# 2. The printer client: device facts -> PrinterState, via the polling seam.


class VendorPrinter(PrinterClient[VendorConfig]):
    def device_drivers(self):
        # A real vendor polls its device's HTTP API here; the simulation just
        # synthesizes readings, so every poll is a "sign of life".
        return (DevicePoller(self, interval=1.0, offline_after=10.0),)

    async def setup_device(self) -> None:  # called from your own init() if needed
        pass

    async def poll_device(self) -> None:
        self.printer.bed.temperature.actual = 20 + random.random()
        self.printer.tool0.temperature.actual = 20 + random.random()
        self.apply_status(PrinterStatus.OPERATIONAL)


# 3. The flow: the guided add-printer skeleton, composed.


async def _verify(state: FlowState, _answer) -> dict:
    host = str(state["host"])
    if not host:
        raise FlowError("No printer found at that address")
    return {"host": host, "serial": f"DEMO-{host}"}


def _make_config(state: FlowState) -> VendorConfig:
    config = VendorConfig.get_new()
    config.name = state.get("name") or None
    config.host = str(state["host"])
    config.serial = str(state.get("serial")) or None
    return config


def build_add_printer_flow():
    return standard_add_printer_flow(
        title="Add a Vendor printer",
        make_config=_make_config,
        verify=_verify,
        address_help="Your printer's address on your local network.",
        address_placeholder="192.168.1.42",
    )


# 4. The spec: the one declarative descriptor.


async def _probe(host: str):
    return None  # a real probe confirms (and identifies) one reachable host


class VendorSpec(PrinterSpec):
    KEY = "vendor"
    metadata = ProductMetadata(
        display_name="Vendor",
        image_url="/img/vendor.png",
        supported_transports=("http",),
        capabilities=(),
    )
    client = lazy("example.headless_vendor:VendorPrinter")
    config = lazy("example.headless_vendor:VendorConfig")

    @classmethod
    def subnet_spec(cls):
        # Declaring any discovery spec gives this type the DEFAULT discover():
        # scan the shared service under KEY -> neutral DiscoveredDevices.
        return SubnetScanSpec(brand=cls.KEY, probe=_probe, key=lambda r: r.host)

    @classmethod
    def add_printer_flow(cls):
        return build_add_printer_flow()


# 5. The headless main(): registry -> host -> discover -> flow -> add -> run.


async def onboard(host: Host) -> None:
    devices = await host.discover("vendor", timeout=1.0)
    print(f"discovered on the LAN: {devices!r}")

    flow = host.flow("vendor", "add-printer")

    async def answer(prompt):
        # A CLI would render prompt.step and read input; the demo scripts it.
        print(f"  flow prompt: {prompt.step.id}")
        return {}

    config = await run_flow(
        flow,
        on_prompt=answer,
        initial_state={"host": "192.168.1.42", "name": "Demo Vendor"},
    )
    client = host.add_printer("vendor", config)
    print(f"added printer {config.unique_id} -> {type(client).__name__}")


def main() -> None:
    registry = SpecRegistry.of(VendorSpec)
    # A real vendor persists configs: config_manager_t=ConfigManagerType.JSON.
    host = Host(registry, ClientSettings(name="vendor-demo"))

    host.start_discovery()
    host.start_services()
    asyncio.run(onboard(host))

    demo_seconds = float(os.environ.get("SPWS_DEMO_SECONDS", "5"))
    if demo_seconds > 0:
        host.app.run_detached()
        import time

        time.sleep(demo_seconds)
        host.stop()
        print("demo done")
    else:
        try:
            host.start(detach_fleet=False)  # blocks; Ctrl+C to stop
        finally:
            host.stop()


if __name__ == "__main__":
    # ``python -m example.headless_vendor`` imports this file as ``__main__``,
    # while the spec's lazy refs import it under its canonical name -- two
    # module objects, two VendorConfig classes. Delegate to the canonical one
    # so every constructed object shares one class identity.
    from example.headless_vendor import main as _main

    _main()
