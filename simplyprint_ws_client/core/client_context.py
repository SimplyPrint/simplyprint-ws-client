"""Explicit process-owned dependencies supplied to every runtime client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Mapping, Optional, Protocol

from simplyprint_ws_client.wire.pools import PoolRegistry

if TYPE_CHECKING:
    from simplyprint_ws_client.common.asyncio.event_loop_provider import (
        EventLoopProvider,
    )
    from simplyprint_ws_client.common.asyncio.offload import Offload
    from simplyprint_ws_client.integration.camera.pool import CameraPool
    from simplyprint_ws_client.integration.accounts import AccountProvider
    from simplyprint_ws_client.integration.discovery.service import DiscoveryService
    from simplyprint_ws_client.core.api.simplyprint_api import SimplyPrintApi
    from simplyprint_ws_client.integration.client import AppUpdater
    from simplyprint_ws_client.wire.transport import MqttTransport, WsTransport

__all__ = ["BackgroundService", "ClientContext"]


class BackgroundService(Protocol):
    """One host-owned service available to integration factories."""

    def stop(self) -> None: ...


@dataclass(frozen=True)
class ClientContext:
    """Dependencies owned by the app/host, never discovered through globals.

    ``background_service`` is already scoped to the integration whose factory
    receives this value. A factory never looks up its own id in a service map.
    """

    event_loop_provider: Optional["EventLoopProvider"] = None
    camera_pool: Optional["CameraPool"] = None
    offload: Optional["Offload"] = None
    discovery_service: Optional["DiscoveryService"] = None
    background_service: Optional[BackgroundService] = None
    account_provider: Optional["AccountProvider"] = None
    simplyprint_api: Optional["SimplyPrintApi"] = None
    app_updater: Optional["AppUpdater"] = None
    host_telemetry: Optional[Callable[[], Awaitable[Mapping[str, int | None]]]] = None
    mqtt_pools: "PoolRegistry[MqttTransport]" = field(default_factory=PoolRegistry)
    websocket_pools: "PoolRegistry[WsTransport]" = field(default_factory=PoolRegistry)
