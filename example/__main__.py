from simplyprint_ws_client import (
    ClientApp,
    ClientSettings,
    ConfigManagerType,
    ConnectionMode,
    IntegrationId,
    IntegrationCapability,
    IntegrationSpec,
    ProductMetadata,
)
from simplyprint_ws_client.common.asyncio.event_loop_runner import EventLoopBackend
from simplyprint_ws_client.common.logging import setup_logging
from simplyprint_ws_client.integration.discovery import DiscoveryService

from .virtual_client import VirtualCamera, VirtualClient, VirtualConfig

if __name__ == "__main__":
    settings = ClientSettings(
        integrations=(
            IntegrationSpec(
                id=IntegrationId("virtual"),
                client_factory=VirtualClient,
                config_factory=VirtualConfig,
                metadata=ProductMetadata(
                    display_name="Virtual",
                    image_url="/virtual.png",
                    supported_transports=(),
                    capabilities=(IntegrationCapability.CAMERA,),
                ),
            ),
        ),
        name="la_fair_printers",
        mode=ConnectionMode.SINGLE,
        event_loop_backend=EventLoopBackend.AUTO,
        allow_setup=True,
        config_manager_t=ConfigManagerType.JSON,
        development=True,
        camera_workers=1,
        camera_protocols=[VirtualCamera],
    )

    setup_logging(settings)
    app = ClientApp(
        settings,
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    app.run_blocking()
