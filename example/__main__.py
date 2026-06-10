from simplyprint_ws_client import (
    ClientApp,
    ClientSettings,
    ConfigManagerType,
    ConnectionMode,
)
from simplyprint_ws_client.common.asyncio.event_loop_runner import EventLoopBackend
from simplyprint_ws_client.common.cli.cli import ClientCli
from simplyprint_ws_client.common.logging import setup_logging

from .virtual_client import VirtualCamera, VirtualClient, VirtualConfig

if __name__ == "__main__":
    settings = ClientSettings(
        name="la_fair_printers",
        mode=ConnectionMode.SINGLE,
        event_loop_backend=EventLoopBackend.AUTO,
        client_factory=VirtualClient,
        config_factory=VirtualConfig,
        allow_setup=True,
        config_manager_t=ConfigManagerType.JSON,
        development=True,
        camera_workers=1,
        camera_protocols=[VirtualCamera],
    )

    setup_logging(settings)
    app = ClientApp(settings)
    cli = ClientCli(app)
    cli.start_client = lambda: app.run_blocking()
    cli(prog_name="python -m simplyprint_ws_client")
