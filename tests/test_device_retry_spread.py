"""Device links that drop together must not retry together: the pace of a
printer's own link (decorrelated jitter) and of a failing poller."""

from random import Random
from statistics import pstdev

import pytest
import yarl

from simplyprint_ws_client.common.utils.backoff import DecorrelatedJitterBackoff
from simplyprint_ws_client.wire.async_mqtt import AsyncMqtt
from simplyprint_ws_client.wire.policy import (
    DEVICE_RETRY_BASE,
    DEVICE_RETRY_MAX,
    device_retry_policy,
)


def _links(n: int, seed: int = 7) -> list:
    rand = Random(seed)
    return [
        DecorrelatedJitterBackoff(DEVICE_RETRY_BASE, DEVICE_RETRY_MAX, Random(rand.random()))
        for _ in range(n)
    ]


def test_the_first_retry_is_quick_but_spread():
    firsts = [link.delay() for link in _links(250)]

    assert all(DEVICE_RETRY_BASE <= d <= 3 * DEVICE_RETRY_BASE for d in firsts)
    # 250 links dropped at once: their retries cover the whole first window,
    # not one instant (uniform over 1 s has a deviation of ~0.29 s).
    assert max(firsts) - min(firsts) > 0.9
    assert pstdev(firsts) > 0.2


def test_retries_grow_to_the_cap_and_never_below_the_base():
    link = DecorrelatedJitterBackoff(0.5, 5.0, Random(1))
    delays = [link.delay() for _ in range(200)]

    assert all(0.5 <= d <= 5.0 for d in delays)
    assert max(delays) > 4.0  # it does reach the long waits


def test_later_rounds_drift_apart():
    links = _links(250)
    rounds = [[link.delay() for link in links] for _ in range(4)]

    # Each round's retries are spread wider than the first's.
    assert pstdev(rounds[3]) > pstdev(rounds[0])


def test_a_connect_starts_the_pace_over():
    link = DecorrelatedJitterBackoff(0.5, 5.0, Random(3))
    for _ in range(10):
        link.delay()

    link.reset()

    assert 0.5 <= link.delay() <= 1.5


def test_the_device_policy_is_decorrelated_jitter():
    assert isinstance(device_retry_policy().backoff, DecorrelatedJitterBackoff)


@pytest.mark.asyncio
async def test_an_mqtt_link_retries_at_the_device_pace():
    transport = AsyncMqtt(yarl.URL("mqtts://printer.local"))

    assert isinstance(transport.policy.backoff, DecorrelatedJitterBackoff)


def test_a_failing_poller_waits_between_half_and_all_of_its_backoff():
    from simplyprint_ws_client.integration.drivers import DevicePoller

    poller = DevicePoller.__new__(DevicePoller)  # only the timing is used
    poller.failure_backoff = 10.0
    delays = [poller.failure_delay() for _ in range(500)]

    assert all(5.0 <= d <= 10.0 for d in delays)
    assert max(delays) - min(delays) > 4.0
