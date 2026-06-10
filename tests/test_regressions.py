"""Regression tests for verified bug fixes (review pass 0).

Each test pins a bug that previously existed; see IMPROVEMENT_PLAN.md at the
workspace root for the inventory.
"""

import asyncio
import importlib
import os
import pkgutil
import socket
import time
import types
import uuid
from typing import List, Mapping, Optional

import psutil
import pytest
from pydantic import BaseModel

import simplyprint_ws_client
from simplyprint_ws_client import Client
from simplyprint_ws_client.events.emitter import Emitter  # noqa: F401 - resolved by get_type_hints
from simplyprint_ws_client.common.asyncio.continuous_task import ContinuousTask
from simplyprint_ws_client.common.asyncio.event_loop_runner import Runner
from simplyprint_ws_client.common.model.diff import SimpleUpdateModel, UpdatedField
from simplyprint_ws_client.common.model.reactive import ReactiveModel
from simplyprint_ws_client.core.files.file_manager import File, FileManager
from simplyprint_ws_client.core.files.file_backup import FileBackup
from simplyprint_ws_client.core.protocol.messages import (
    ConnectedMsg,
    ResolveNotificationDemandData,
    SetMaterialDataDemandData,
)
from simplyprint_ws_client.core.protocol.models import ServerMsgType
from simplyprint_ws_client.core.settings import ClientSettings
from simplyprint_ws_client.core.state import MaterialEntry


# --- every module must import -------------------------------------------------


def _walk_module_names():
    pkg = simplyprint_ws_client
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        yield info.name


@pytest.mark.parametrize("module_name", sorted(_walk_module_names()))
def test_module_imports(module_name):
    """Every library module must be importable (catches NameError-at-def bugs)."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # Heavy optional backends (paho-mqtt, aiomqtt, ...) may be absent in
        # the test environment; anything else is a real failure.
        if e.name and e.name.split(".")[0] not in simplyprint_ws_client.__name__:
            pytest.skip(f"optional dependency missing: {e.name}")
        raise


# --- Runner must not swallow exceptions ----------------------------------------


def test_runner_propagates_exceptions():
    with pytest.raises(RuntimeError, match="boom"):
        with Runner():
            raise RuntimeError("boom")


def test_runner_returns_result():
    async def main():
        return 42

    with Runner() as runner:
        assert runner.run(main()) == 42


def test_runner_explicit_debug_false_is_respected():
    async def main():
        return asyncio.get_running_loop().get_debug()

    runner = Runner(debug=False)
    with runner:
        assert runner.run(main(), debug=False) is False


# --- diff engine ----------------------------------------------------------------


class _Inner(SimpleUpdateModel, BaseModel):
    a: Optional[int] = None


class _Outer(SimpleUpdateModel, BaseModel):
    items: Optional[List[_Inner]] = None
    nums: Optional[List[int]] = None


def test_no_false_positive_diff_for_untouched_model_lists():
    """A partial update whose list elements carry nothing must not report the list."""
    left = _Outer(items=[_Inner(a=1), _Inner(a=2)])
    right = _Outer(items=[_Inner(), _Inner()])

    assert left.update_model(right) == {}
    assert [item.a for item in left.items] == [1, 2]


def test_model_list_diff_is_index_aligned():
    left = _Outer(items=[_Inner(a=1), _Inner(a=2)])
    right = _Outer(items=[_Inner(), _Inner(a=3)])

    changes = left.update_model(right)

    assert changes == {"items": [None, {"a": UpdatedField(2, 3)}]}
    assert left.items[1].a == 3


def test_scalar_list_element_changes_are_reported():
    left = _Outer(nums=[1, 2, 3])
    right = _Outer(nums=[1, 5, 3])

    changes = left.update_model(right)

    # Scalar elements follow the engine's "touched" semantics: every element
    # the update carried is reported; has_changed() distinguishes real changes.
    assert changes == {
        "nums": [UpdatedField(1, 1), UpdatedField(2, 5), UpdatedField(3, 3)]
    }
    assert left.nums == [1, 5, 3]
    nums_diff = changes["nums"]
    assert [f.has_changed() for f in nums_diff] == [False, True, False]


def test_updated_field_is_generic():
    field: UpdatedField[int] = UpdatedField(1, 2)
    assert field.has_changed()


# --- reactive annotation detection ----------------------------------------------


class _Leaf(ReactiveModel):
    x: int = 0


def test_mapping_fields_are_change_detected():
    detect = ReactiveModel.is_pydantic_change_detect_annotation
    assert detect(Mapping[str, _Leaf]) is True
    assert detect(List[_Leaf]) is True
    assert detect(Optional[_Leaf]) is True
    assert detect(Mapping[str, int]) is False


# --- client demand handlers ------------------------------------------------------


def test_set_material_data_applies_materials(client: Client):
    data = SetMaterialDataDemandData(
        materials=[
            MaterialEntry(nozzle=0, ext=0, type="PLA", color="Red", hex="#FF0000"),
            # Out-of-range entries must be skipped, not crash.
            MaterialEntry(nozzle=0, ext=99, type="PETG"),
            MaterialEntry(nozzle=42, ext=0, type="ABS"),
        ]
    )

    client._on_set_material_data(data)

    applied = client.printer.material(0, 0)
    assert applied is not None
    assert applied.type == "PLA"
    assert applied.color == "Red"
    assert applied.hex == "#FF0000"


@pytest.mark.asyncio
async def test_resolve_notification_for_unknown_event_is_ignored(client: Client):
    data = ResolveNotificationDemandData(event_id=uuid.uuid4(), action=None)

    # Must not raise even though the event id is unknown client-side.
    await client._on_resolve_notification(data)


@pytest.mark.asyncio
async def test_connected_msg_without_data_is_tolerated(client: Client):
    msg = ConnectedMsg(type=ServerMsgType.CONNECTED)
    assert msg.data is None

    previous_name = client.config.name

    await client._on_connected_data(msg)

    assert client.config.name == previous_name


# --- settings are real dataclass fields ------------------------------------------


def test_client_settings_tick_rate_is_a_field():
    settings = ClientSettings(tick_rate=2.5)
    assert settings.tick_rate == 2.5
    assert not hasattr(settings, "reconnect_timeout")


# --- file manager / backup --------------------------------------------------------


def test_get_files_to_remove_with_duplicate_entries():
    fm = FileManager(max_age=10)
    now = int(time.time())
    dup = File("dup.gcode", 100, last_modified=now - 100)
    fresh = File("fresh.gcode", 50, last_modified=now)
    files = [dup, fresh, dup]

    removed = list(fm.get_files_to_remove(files, 10_000, 250))

    assert removed.count(dup) == 2
    assert files == [fresh]


def test_backup_file_removes_all_expired_backups(tmp_path):
    import datetime

    target = tmp_path / "config.json"
    target.write_text("current")

    old_mtime = time.time() - 3600
    for i in range(3):
        backup = tmp_path / f"config.json.bak.{i}"
        backup.write_text(f"old-{i}")
        os.utime(backup, (old_mtime, old_mtime))

    FileBackup.backup_file(target, max_age=datetime.timedelta(seconds=60))

    backups = sorted(p.name for p in tmp_path.glob("config.json.bak.*"))
    assert backups == ["config.json.bak.0"]
    assert (tmp_path / "config.json.bak.0").read_text() == "current"


# --- host network MAC fallback ------------------------------------------------------


def test_no_mac_reported_when_no_interface_matches(monkeypatch):
    from simplyprint_ws_client.common.hardware import host_network

    class _FakeSocket:
        def __init__(self, *args, **kwargs): ...

        def settimeout(self, value): ...

        def connect(self, addr): ...

        def getsockname(self):
            return ("10.99.99.99", 0)

        def close(self): ...

    monkeypatch.setattr(host_network.socket, "socket", _FakeSocket)
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "eth0": [
                types.SimpleNamespace(
                    family=psutil.AF_LINK, address="aa:bb:cc:dd:ee:ff"
                ),
                types.SimpleNamespace(family=socket.AF_INET, address="192.168.1.2"),
            ]
        },
    )

    info = host_network.get_local_ip_and_mac()

    assert info.ip == "10.99.99.99"
    # No interface carries that ip, so no MAC may be guessed.
    assert info.mac is None


def test_mac_reported_for_matching_interface(monkeypatch):
    from simplyprint_ws_client.common.hardware import host_network

    class _FakeSocket:
        def __init__(self, *args, **kwargs): ...

        def settimeout(self, value): ...

        def connect(self, addr): ...

        def getsockname(self):
            return ("192.168.1.2", 0)

        def close(self): ...

    monkeypatch.setattr(host_network.socket, "socket", _FakeSocket)
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "wlan0": [
                types.SimpleNamespace(
                    family=psutil.AF_LINK, address="11:22:33:44:55:66"
                ),
                types.SimpleNamespace(family=socket.AF_INET, address="192.168.1.2"),
            ]
        },
    )

    info = host_network.get_local_ip_and_mac()

    assert info == ("192.168.1.2", "11:22:33:44:55:66")


# --- continuous task -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_of_cancelled_task_does_not_raise():
    task = ContinuousTask(lambda: asyncio.sleep(60), factory=asyncio.get_running_loop)
    task.schedule()
    task.cancel()

    # Let the cancellation be delivered so the task settles as cancelled.
    await asyncio.sleep(0)
    assert task.cancelled()

    task.discard()
    assert task.task is None


# --- predicate stack ------------------------------------------------------------


def test_predicate_tree_evaluates_all_matching_branches():
    from simplyprint_ws_client.events.event_bus_predicate_tree import (
        EventBusPredicateTree,
    )
    from simplyprint_ws_client.events.predicate import Eq, Gt, Sel

    tree = EventBusPredicateTree()
    # Two sibling chains that can both match the same input.
    a = tree.add("starts-positive", Sel(0) | Gt(0))
    b = tree.add("is-five", Sel(0) | Eq(5))

    matches = {tree.resources[rid] for rid in tree.evaluate(5)}

    # Previously only the first matching sibling was descended.
    assert matches == {"starts-positive", "is-five"}

    tree.remove_resource_id(a)
    matches = {tree.resources[rid] for rid in tree.evaluate(5)}
    assert matches == {"is-five"}

    tree.remove_resource_id(b)
    assert list(tree.evaluate(5)) == []
    assert tree.root.predicates == []


def test_pipe_or_does_not_mutate_the_original_chain():
    from simplyprint_ws_client.events.predicate import Eq, Gt, Sel

    base = Sel(0)
    first = base | Eq(1)
    second = base | Gt(10)

    # Previously building ``second`` silently extended ``base``/``first``.
    assert base.output is None
    assert first(1) is True
    assert first(11) is False
    assert second(11) is True
    assert second(1) is False


def test_reduce_equality_respects_lambda_constants():
    from simplyprint_ws_client.events.predicate import Reduce

    assert Reduce(lambda x: x > 5) != Reduce(lambda x: x > 99)
    assert Reduce(lambda x: x > 5) == Reduce(lambda x: x > 5)


def test_reduce_equality_respects_closures():
    from simplyprint_ws_client.events.predicate import Reduce

    def make(n):
        return lambda x: x > n

    assert Reduce(make(5)) != Reduce(make(99))


def test_forward_emitter_detected_with_string_annotations():
    from simplyprint_ws_client.events.event_bus_listeners import (
        EventBusListener,
        ListenerLifetimeForever,
    )

    # PEP 563-style string annotation, as produced by
    # ``from __future__ import annotations``. ``Emitter`` is importable from
    # this module's globals, exactly like a real handler module.
    def handler(event, emitter: "Emitter"): ...

    listener = EventBusListener(ListenerLifetimeForever(), 0, handler)

    assert listener.forward_emitter == "emitter"


# --- config persistence ----------------------------------------------------------


def test_corrupt_json_config_is_preserved_not_reset(tmp_path):
    from simplyprint_ws_client.core.config import PrinterConfig
    from simplyprint_ws_client.core.config.json import JsonConfigManager

    manager = JsonConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=tmp_path
    )
    corrupt_source = '{"definitely": "not a list"'
    (tmp_path / "printers.json").write_text(corrupt_source)

    manager.load()

    # The unreadable file is preserved for recovery, never silently discarded.
    assert (tmp_path / "printers.json.corrupt").read_text() == corrupt_source
    assert manager.get_all() == []


def test_json_config_flush_is_atomic(tmp_path):
    import json as json_module

    from simplyprint_ws_client.core.config import PrinterConfig
    from simplyprint_ws_client.core.config.json import JsonConfigManager

    manager = JsonConfigManager(
        name="printers", config_t=PrinterConfig, base_directory=tmp_path
    )
    config = PrinterConfig.get_new()
    config.id = 7
    manager.persist(config)
    manager.flush()

    data = json_module.loads((tmp_path / "printers.json").read_text())
    assert [entry["id"] for entry in data] == [7]
    # No temp file left behind.
    assert not (tmp_path / "printers.json.tmp").exists()
