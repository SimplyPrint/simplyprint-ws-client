import asyncio
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

from simplyprint_ws_client.core.client_context import ClientContext


def test_background_service_is_an_integration_scoped_value() -> None:
    service = object()
    context = ClientContext(background_service=service)

    assert context.background_service is service


def test_discovery_service_is_an_explicit_context_value() -> None:
    service = object()
    context = ClientContext(discovery_service=service)  # type: ignore[arg-type]

    assert context.discovery_service is service


def test_context_value_is_frozen() -> None:
    context = ClientContext()

    with pytest.raises(FrozenInstanceError):
        context.offload = object()  # type: ignore[misc]


@pytest.mark.asyncio
async def test_contexts_own_isolated_transport_registries() -> None:
    first = ClientContext()
    second = ClientContext()
    first_pool = Mock()
    second_pool = Mock()
    first_pool.stop.return_value = []
    second_pool.stop.return_value = []
    first_pool.provider.event_loop = asyncio.get_running_loop()
    second_pool.provider.event_loop = asyncio.get_running_loop()

    assert first.mqtt_pools.get(None, None, lambda: first_pool) is first_pool
    assert second.mqtt_pools.get(None, None, lambda: second_pool) is second_pool

    await first.mqtt_pools.close()

    first_pool.stop.assert_called_once_with()
    second_pool.stop.assert_not_called()


def test_scoped_context_shares_its_owners_transport_registries() -> None:
    owner = ClientContext()
    scoped = replace(owner, background_service=object())

    assert scoped.mqtt_pools is owner.mqtt_pools
    assert scoped.websocket_pools is owner.websocket_pools
