"""Host.mount / ClientApp.mount_domain: one path for startup and live plugins.

A whole domain is mounted and closed while the host runs and leaves nothing
behind but its stored configs; a mount that raises midway cleans up fully; the
teardown order stops printers before their discovery and registry entries go;
and startup through mounting yields the same fleet as before.
"""

import asyncio
import threading
import time
from datetime import timedelta
from typing import Dict, List

import pytest

from simplyprint_ws_client import ClientApp, ClientSettings, PrinterConfig
from simplyprint_ws_client.core.client import Client
from simplyprint_ws_client.core.host import Host
from simplyprint_ws_client.core.registry import SpecRegistry
from simplyprint_ws_client.core.scope import Scope
from simplyprint_ws_client.core.status.registry import StatusRegistry
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.integration.discovery import DiscoveryService
from simplyprint_ws_client.integration.discovery.spec import (
    NetworkServiceSpec,
    SubnetScanSpec,
)
from simplyprint_ws_client.integration.spec import (
    IntegrationId,
    IntegrationSpec,
    ProductMetadata,
)
from simplyprint_ws_client.integration.tasks import SchedulerService, TaskRegistry

_METADATA = ProductMetadata(
    display_name="Mount",
    image_url="/mount.png",
    supported_transports=(),
    capabilities=(),
)

#: Ordered record of teardown-relevant edges, shared by the fakes below.
EVENTS: List[tuple] = []


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _Service:
    def __init__(self, key: str) -> None:
        self.key = key
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True
        EVENTS.append(("service-stop", self.key))


class _Provider:
    pass


class _Cam(BaseCameraProtocol):
    is_async = True
    polling_mode = CameraProtocolPollingMode.ON_DEMAND

    @staticmethod
    def test(uri):
        return uri.scheme == "mount"

    async def read(self):
        yield b"frame"


class _OfflineClient(Client[PrinterConfig]):
    """Never asks for a SimplyPrint connection; snapshots the host at teardown."""

    host = None  # set by each test

    def __init__(self, config, *, context):
        super().__init__(config, context=context)
        self._should_be_allocated = False
        self.domain = None

    async def teardown(self):
        host = _OfflineClient.host
        key = self.domain
        if host is None or key is None:
            return
        pool = host.app.camera_pool
        EVENTS.append(
            (
                "teardown",
                str(self.unique_id),
                {
                    "registry": host.registry.get(key) is not None,
                    "discovery": key in host.discovery.subnet_keys(),
                    "service": key in host.services and not host.services[key].stopped,
                    "cameras": pool is None or _Cam in pool.all_protocols(),
                    "provider": key in host.account_providers,
                },
            )
        )


def _factory(key: str, broken=()):
    def factory(config, *, context):
        if config.unique_id in broken:
            raise RuntimeError("driver exploded for %s" % config.unique_id)
        client = _OfflineClient(config, context=context)
        client.domain = key
        return client

    return factory


async def _probe(_host):
    return None


def _plain_spec(key: str, broken=()) -> IntegrationSpec:
    return IntegrationSpec(
        id=IntegrationId(key),
        client_factory=_factory(key, broken),
        config_factory=PrinterConfig,
        metadata=_METADATA,
    )


def _register_tasks(registry: TaskRegistry) -> None:
    @registry.periodic(interval=timedelta(hours=1), name="poll")
    async def poll(ctx):
        return None


def _full_spec(key: str = "plug", **overrides) -> IntegrationSpec:
    fields = dict(
        id=IntegrationId(key),
        client_factory=_factory(key),
        config_factory=PrinterConfig,
        metadata=_METADATA,
        account_provider_factory=lambda manager_t: _Provider(),
        background_service_factory=lambda provider: _Service(key),
        camera_protocols=lambda: (_Cam,),
        subnet=SubnetScanSpec(brand=key, probe=_probe, key=lambda record: record),
        network_services=(NetworkServiceSpec(id="http", transport="tcp", port=80),),
        register_tasks=_register_tasks,
    )
    fields.update(overrides)
    return IntegrationSpec(**fields)


def _config(unique_id: str, pk: int = 1) -> PrinterConfig:
    return PrinterConfig(id=pk, token="token-%s" % unique_id, unique_id=unique_id)


@pytest.fixture
def running():
    """A started host (fleet, services, discovery) with a task scheduler."""
    EVENTS.clear()
    tasks = SchedulerService(status=StatusRegistry(), registry=TaskRegistry())
    host = Host(
        SpecRegistry.of(_plain_spec("base")),
        ClientSettings(name="mount-test", camera_workers=1),
        task_scheduler=tasks,
    )
    _OfflineClient.host = host
    tasks.start()
    host.start(detach_fleet=True)
    assert _wait_for(lambda: host.app.scheduler.event_loop_is_running())
    yield host, tasks
    host.stop()
    tasks.shutdown()
    _OfflineClient.host = None


def _leftovers(host: Host, tasks: SchedulerService, key: str) -> Dict[str, object]:
    pool = host.app.camera_pool
    return {
        "registry": host.registry.get(key),
        "provider": host.account_providers.get(key),
        "service": host.services.get(key),
        "cameras": [p for p in pool.all_protocols() if p is _Cam] if pool else [],
        "subnet": key in host.discovery.subnet_keys(),
        "network_services": key in host.discovery.network_service_keys(),
        "jobs": [j for j in tasks.job_ids() if j.startswith(key + ".")],
        "task_specs": [n for n in tasks._registry.specs_by_name() if n.startswith(key)],
        "clients": [
            uid
            for uid, client in host.app.client_list.items()
            if getattr(client, "domain", None) == key
        ],
        "domain": key in host.app.mounted_domains(),
        "scope": host.scope_for(key),
    }


_NOTHING = {
    "registry": None,
    "provider": None,
    "service": None,
    "cameras": [],
    "subnet": False,
    "network_services": False,
    "jobs": [],
    "task_specs": [],
    "clients": [],
    "domain": False,
    "scope": None,
}


def test_a_whole_domain_mounts_live_and_closes_leaving_only_its_configs(running):
    host, tasks = running
    spec = _full_spec()
    scope = host.mount(spec, host.root)
    host.add_printer("plug", _config("p1", 1))
    host.add_printer("plug", _config("p2", 2))

    started = threading.Event()

    async def worker():
        started.set()
        await asyncio.sleep(3600)

    spawned = scope.spawn(worker(), name="plug.worker")
    assert started.wait(5)

    mounted = _leftovers(host, tasks, "plug")
    assert mounted["registry"] is spec
    assert isinstance(mounted["provider"], _Provider)
    assert isinstance(mounted["service"], _Service)
    assert mounted["cameras"] == [_Cam]
    assert mounted["subnet"] and mounted["network_services"]
    assert _wait_for(lambda: tasks.job_ids() == ("plug.poll",))
    assert sorted(mounted["clients"]) == ["p1", "p2"]
    assert host.app.context_for("plug").background_service is mounted["service"]

    service = mounted["service"]
    report = scope.close("unloaded by test", timeout=15)

    assert report.ok, report
    assert _leftovers(host, tasks, "plug") == _NOTHING
    assert spawned.done()
    assert service.stopped
    for unique_id in ("p1", "p2"):
        failure = host.app.failed_clients[unique_id]
        assert failure.stage == "plugin"
        assert failure.integration_id == "plug"
        assert failure.error == "unloaded by test"
    # The built-in domain is untouched.
    assert host.app.mounted_domains() == ("base",)
    assert host.registry.get("base") is not None

    # Configs are kept: mounting again brings the same printers back.
    again = host.mount(spec, host.root)
    assert _wait_for(
        lambda: sorted(_leftovers(host, tasks, "plug")["clients"]) == ["p1", "p2"]
    )
    assert "p1" not in host.app.failed_client_ids()
    assert again.close("done", timeout=15).ok


def test_teardown_stops_printers_before_their_entries_go(running):
    host, tasks = running
    scope = host.mount(_full_spec(), host.root)
    host.add_printer("plug", _config("p1"))
    EVENTS.clear()

    assert scope.close("unload", timeout=15).ok

    teardown = [event for event in EVENTS if event[0] == "teardown"]
    assert len(teardown) == 1
    _kind, unique_id, seen = teardown[0]
    assert unique_id == "p1"
    # Everything the printer could depend on was still there while it stopped.
    assert seen == {
        "registry": True,
        "discovery": True,
        "service": True,
        "cameras": True,
        "provider": True,
    }
    assert EVENTS.index(teardown[0]) < EVENTS.index(("service-stop", "plug"))


def test_a_mount_that_raises_midway_is_undone_fully(running):
    host, tasks = running
    reached = []

    def setup(scope: Scope) -> None:
        reached.append(scope)
        raise RuntimeError("setup exploded")

    with pytest.raises(RuntimeError, match="setup exploded"):
        host.mount(_full_spec(), host.root, setup=setup)

    assert len(reached) == 1 and reached[0].closed
    assert _leftovers(host, tasks, "plug") == _NOTHING
    assert any(event == ("service-stop", "plug") for event in EVENTS)
    # Nothing is left reserved: the same id mounts cleanly afterwards.
    scope = host.mount(_full_spec(), host.root)
    assert host.registry.get("plug") is not None
    assert scope.close("done", timeout=15).ok


def test_a_mount_whose_discovery_clashes_is_undone_before_discovery(running):
    host, tasks = running
    first = host.mount(_full_spec("one"), host.root)
    clash = _full_spec(
        "two",
        subnet=SubnetScanSpec(brand="one", probe=_probe, key=lambda record: record),
    )

    with pytest.raises(ValueError, match="duplicate subnet"):
        host.mount(clash, host.root)

    # "one" still owns its _Cam (every test spec shares the class): only the
    # failed mount's own registration may be gone.
    left = _leftovers(host, tasks, "two")
    assert left["cameras"] == [_Cam]
    assert {**left, "cameras": []} == _NOTHING
    assert host.discovery.subnet_keys().count("one") == 1
    assert first.close("done", timeout=15).ok


def test_mounting_an_id_twice_is_rejected(running):
    host, _tasks = running
    scope = host.mount(_full_spec(), host.root)
    with pytest.raises(ValueError, match="already mounted"):
        host.mount(_full_spec(), host.root)
    with pytest.raises(ValueError, match="already mounted"):
        host.mount(host.registry.get("base"), host.root)
    scope.close("done", timeout=15)


def test_tasks_attach_late_into_each_mounts_place():
    EVENTS.clear()
    host = Host(SpecRegistry.of(_full_spec("early")), ClientSettings(name="late-tasks"))
    tasks = SchedulerService(status=StatusRegistry(), registry=TaskRegistry())
    try:
        assert tasks._registry.specs_by_name() == {}
        host.attach_task_scheduler(tasks)
        assert set(tasks._registry.specs_by_name()) == {"early.poll"}
        with pytest.raises(ValueError):
            host.attach_task_scheduler(
                SchedulerService(status=StatusRegistry(), registry=TaskRegistry())
            )
    finally:
        host.stop()
    assert tasks._registry.specs_by_name() == {}


def test_services_built_at_start_services_belong_to_their_mount():
    EVENTS.clear()
    host = Host(
        SpecRegistry.of(_full_spec("one"), _full_spec("two")),
        ClientSettings(name="services"),
    )
    try:
        assert host.services == {}
        host.start_services()
        assert set(host.services) == {"one", "two"}
        late = host.mount(_full_spec("three"), host.root)
        assert isinstance(host.service_for("three"), _Service)  # built at mount
        late.close("unload")
        assert ("service-stop", "three") in EVENTS
        assert host.service_for("three") is None
    finally:
        host.stop()
    # Without a sink the host stops them, newest first, as before.
    stops = [event[1] for event in EVENTS if event[0] == "service-stop"]
    assert stops == ["three", "two", "one"]


def test_a_service_sink_owns_the_stop_at_host_stop():
    EVENTS.clear()
    sunk = []

    def sink(service):
        sunk.append(service)
        return service

    host = Host(
        SpecRegistry.of(_full_spec("one")),
        ClientSettings(name="sink"),
        service_sink=sink,
    )
    host.start_services()
    host.stop()

    assert [service.key for service in sunk] == ["one"]
    assert not sunk[0].stopped  # the sink's owner stops it, not the host
    assert host.services == {}


def test_host_stop_closes_the_root_scope_once():
    EVENTS.clear()
    host = Host(SpecRegistry.of(_full_spec("one")), ClientSettings(name="stop"))
    host.start_services()

    report = host.stop()

    assert report.ok
    assert host.root.closed
    assert host.registry.ids() == ()
    assert host.discovery.subnet_keys() == []
    assert host.account_providers == {}
    with pytest.raises(Exception):
        host.mount(_full_spec("late"), host.root)
    assert host.stop().ok  # a second stop is a no-op


# ---------------------------------------------------------------------------
# Startup through mounting is the same fleet as before
# ---------------------------------------------------------------------------


def _fleet(app: ClientApp):
    return sorted(
        (uid, client.domain, type(client).__name__)
        for uid, client in app.client_list.items()
    )


def _store(app: ClientApp) -> None:
    stored = {
        "alpha": [_config("a1", 1), _config("a2", 2), _config("broken", 3)],
        "beta": [_config("b1", 4)],
    }
    for key, configs in stored.items():
        for config in configs:
            app.config_managers[key].persist(config)


def test_startup_through_mounting_yields_the_same_fleet_as_a_plain_app():
    alpha = _plain_spec("alpha", broken={"broken"})
    beta = _plain_spec("beta")
    expected = [
        ("a1", "alpha", "_OfflineClient"),
        ("a2", "alpha", "_OfflineClient"),
        ("b1", "beta", "_OfflineClient"),
    ]

    host = Host(
        SpecRegistry.of(alpha, beta),
        ClientSettings(name="equivalence", camera_workers=None),
    )
    plain = ClientApp(
        ClientSettings(
            name="equivalence-plain", integrations=(alpha, beta), camera_workers=None
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    try:
        assert host.app.mounted_domains() == ("alpha", "beta")
        _store(host.app)
        _store(plain)

        host.start(detach_fleet=True)
        plain.run_detached()

        assert _wait_for(lambda: len(host.app.client_list) == 3)
        assert _wait_for(lambda: len(plain.client_list) == 3)
        assert _fleet(host.app) == _fleet(plain) == expected
        assert host.app.failed_client_ids() == plain.failed_client_ids() == ["broken"]
        assert host.app.failed_clients["broken"].stage == "build"
        assert plain.mounted_domains() == ("alpha", "beta")
        assert set(host.app.config_managers) == set(plain.config_managers)
    finally:
        host.stop()
        plain.stop()


def test_mount_domain_returns_the_printers_that_did_not_start():
    broken = _plain_spec("late", broken={"bad"})
    app = ClientApp(
        ClientSettings(
            name="late-domain", integrations=(_plain_spec("base"),), camera_workers=None
        ),
        discovery_service=DiscoveryService(),
        account_providers={},
    )
    try:
        app.run_detached()
        assert _wait_for(lambda: app.mounted_domains() == ("base",))
        scope = Scope("late")
        assert app.mount_domain(broken, scope) == []  # nothing stored yet
        app.add(_config("good", 1), integration_id="late")
        with pytest.raises(RuntimeError):
            app.add(_config("bad", 2), integration_id="late")
        assert scope.close("update", timeout=15).ok

        # A remount replays the kept configs and reports the one that fails.
        again = Scope("late-again")
        failures = app.mount_domain(broken, again)
        assert [failure.unique_id for failure in failures] == ["bad"]
        assert failures[0].stage == "build"
        assert _wait_for(lambda: "good" in app.client_list)
        assert again.close("done", timeout=15).ok
    finally:
        app.stop()
