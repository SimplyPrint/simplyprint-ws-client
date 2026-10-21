"""Live add and scoped removal for the task scheduler, discovery and cameras.

Each subsystem gets one thing added while it is running and removed again by
closing the scope it was added under, leaving nothing behind.
"""

import asyncio
import concurrent.futures
import socket
import threading
import time
from datetime import timedelta
from typing import Optional

import pytest
from yarl import URL

from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.scope import Scope
from simplyprint_ws_client.core.status.registry import StatusRegistry
from simplyprint_ws_client.events import Event
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.camera.pool import CameraPool
from simplyprint_ws_client.integration.discovery.service import DiscoveryService
from simplyprint_ws_client.integration.discovery.spec import (
    MulticastSpec,
    NetworkServiceSpec,
    SubnetScanSpec,
)
from simplyprint_ws_client.integration.spec import (
    IntegrationId,
    IntegrationSpec,
    ProductMetadata,
)
from simplyprint_ws_client.integration.tasks import (
    ScopedTaskRegistry,
    SchedulerService,
    TaskRegistry,
    TaskSpec,
)

_METADATA = ProductMetadata(
    display_name="Live",
    image_url="/live.png",
    supported_transports=(),
    capabilities=(),
)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# SchedulerService.add
# ---------------------------------------------------------------------------


async def _noop(ctx) -> None:
    return None


def _spec(name: str, fn=_noop, interval: Optional[timedelta] = timedelta(hours=1)):
    return TaskSpec(name=name, fn=fn, interval=interval)


@pytest.fixture
def scheduler():
    service = SchedulerService(status=StatusRegistry(), registry=TaskRegistry())
    yield service
    service.shutdown()


def test_a_task_added_before_start_is_scheduled_at_start_and_removed(scheduler):
    scope = Scope("plugin")
    scheduler.add(_spec("plugin.poll"), scope)

    scheduler.start()
    assert "plugin.poll" in scheduler.job_ids()

    assert scope.close("unload").ok
    assert "plugin.poll" not in scheduler.job_ids()
    assert "plugin.poll" not in scheduler._registry.specs_by_name()


def test_a_task_added_while_running_goes_on_the_timer_and_comes_off(scheduler):
    scheduler.start()
    scope = Scope("plugin")
    scheduler.add(_spec("plugin.poll"), scope)

    assert _wait_for(lambda: "plugin.poll" in scheduler.job_ids())

    assert scope.close("unload").ok
    assert scheduler.job_ids() == ()
    with pytest.raises(KeyError):
        scheduler.fire("plugin.poll")


def test_closing_cancels_and_awaits_an_in_flight_run(scheduler):
    started = threading.Event()
    cancelled = threading.Event()

    async def wedged(ctx):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    scheduler.start()
    scope = Scope("plugin")
    scheduler.add(_spec("plugin.wedged", wedged, interval=None), scope)
    run = scheduler.fire("plugin.wedged", key="printer-1")
    assert started.wait(5)

    report = scope.close("unload", timeout=5)

    assert report.ok
    assert cancelled.is_set()
    assert scheduler._inflight == {}
    with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
        run.result(5)


def test_live_add_rejects_a_duplicate_name(scheduler):
    scheduler.add(_spec("dup"), Scope("a"))
    with pytest.raises(ValueError, match="duplicate task name"):
        scheduler.add(_spec("dup"), Scope("b"))


def test_a_scope_closed_before_start_leaves_nothing_to_schedule(scheduler):
    scope = Scope("plugin")
    scheduler.add(_spec("plugin.poll"), scope)
    scope.close("unload")

    scheduler.start()
    assert scheduler.job_ids() == ()


def test_scoped_task_registry_prefixes_names_once(scheduler):
    scope = Scope("alpha")
    registry = ScopedTaskRegistry(scheduler, scope, prefix="alpha")

    @registry.periodic(interval=timedelta(hours=1), name="sweep")
    async def sweep(ctx):
        return None

    registry.register(_spec("alpha.already"))

    names = set(scheduler._registry.specs_by_name())
    assert names == {"alpha.sweep", "alpha.already"}
    scope.close("unload")
    assert scheduler._registry.specs_by_name() == {}


# ---------------------------------------------------------------------------
# DiscoveryService.add
# ---------------------------------------------------------------------------


class _FoundEvent(Event): ...


def _mapper(request, addr):
    usn = request.headers.get("usn")
    return {"id": usn, "host": addr[0]} if usn else None


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _send_ssdp(port: int, usn: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(
            f"NOTIFY * HTTP/1.1\r\nUSN: {usn}\r\n\r\n".encode(), ("127.0.0.1", port)
        )
    finally:
        sock.close()


async def _probe(_host):
    return None


def _discovery_spec(key: str, port: int) -> IntegrationSpec:
    return IntegrationSpec(
        id=IntegrationId(key),
        client_factory=lambda config, *, context: None,
        config_factory=PrinterConfig,
        metadata=_METADATA,
        multicast=MulticastSpec(
            brand=key,
            group="239.255.255.250",
            port=port,
            event_type=_FoundEvent,
            mapper=_mapper,
            key=lambda record: record["id"],
        ),
        subnet=SubnetScanSpec(brand=key, probe=_probe, key=lambda record: record),
        network_services=(NetworkServiceSpec(id="http", transport="tcp", port=80),),
    )


def test_discovery_added_while_running_listens_and_is_removed():
    service = DiscoveryService()
    service.start()
    try:
        port = _free_udp_port()
        scope = Scope("live")
        service.add(_discovery_spec("live", port), scope)

        assert service.backend_keys() == ["live"]
        assert service.subnet_keys() == ["live"]
        assert service.network_service_keys() == ["live"]
        assert _wait_for(lambda: "live" in service._host._tasks)

        def received() -> bool:
            _send_ssdp(port, "uuid:live-1")
            return bool(service.snapshot("live"))

        assert _wait_for(received, timeout=8.0)
        task = service._host._tasks["live"]

        assert scope.close("unload", timeout=10).ok

        assert task.done()
        assert "live" not in service._host._tasks
        assert service._host.backend_keys() == []
        assert service.backend_keys() == []
        assert service.subnet_keys() == []
        assert service.network_service_keys() == []
        assert service.snapshot("live") == []
    finally:
        service.stop()


def test_discovery_added_before_start_runs_at_start_and_restarts_without_it():
    service = DiscoveryService()
    scope = Scope("early")
    service.add(_discovery_spec("early", _free_udp_port()), scope)
    service.start()
    try:
        assert _wait_for(lambda: "early" in service._host._tasks)
        assert scope.close("unload", timeout=10).ok
        assert service._host.backend_keys() == []
    finally:
        service.stop()

    service.start()  # a restart builds its host from what is still contributed
    try:
        assert service._host.backend_keys() == []
    finally:
        service.stop()


def test_discovery_add_rejects_a_taken_key():
    service = DiscoveryService()
    port = _free_udp_port()
    service.add(_discovery_spec("dup", port), Scope("a"))
    with pytest.raises(ValueError, match="duplicate"):
        service.add(_discovery_spec("dup", port), Scope("b"))


# ---------------------------------------------------------------------------
# CameraPool.add_protocols
# ---------------------------------------------------------------------------


class _LiveSnapshot(BaseCameraProtocol):
    is_async = True
    polling_mode = CameraProtocolPollingMode.ON_DEMAND

    @staticmethod
    def test(uri):
        return uri.scheme == "live"

    async def read(self):
        yield b"LIVE"


@pytest.mark.asyncio
async def test_camera_protocols_added_while_running_serve_and_are_removed():
    pool = CameraPool(event_loop_provider=EventLoopProvider(loop=asyncio.get_running_loop()))
    try:
        with pytest.raises(ValueError):
            pool.create(URL("live://cam"))

        scope = Scope("live")
        pool.add_protocols((_LiveSnapshot,), scope)
        handle = pool.create(URL("live://cam"))
        try:
            frame = await asyncio.wait_for(handle.receive_frame(), 2.0)
            assert frame == b"LIVE"
        finally:
            handle.stop()

        # close() blocks; run it off the loop like a real unload would.
        report = await asyncio.get_running_loop().run_in_executor(
            None, scope.close, "unload"
        )
        assert report.ok
        assert pool.all_protocols() == ()
        with pytest.raises(ValueError):
            pool.create(URL("live://cam"))
    finally:
        pool.stop()


def test_camera_protocol_removal_is_by_token_not_equality():
    pool = CameraPool()
    try:
        first, second = Scope("first"), Scope("second")
        pool.add_protocols((_LiveSnapshot,), first)
        pool.add_protocols((_LiveSnapshot,), second)
        pool.protocols.append(_LiveSnapshot)  # a fixed entry, never scoped

        first.close("unload")
        assert pool.all_protocols() == (_LiveSnapshot, _LiveSnapshot)

        second.close("unload")
        assert pool.all_protocols() == (_LiveSnapshot,)
    finally:
        pool.stop()
