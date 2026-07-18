"""Tests for the name-based, config-driven logging facility.

Routing is derived purely from the (plain, dotted) logger name -- no
``ClientName``. Per-printer records land in ``<uid>/<sub>.log``, system records in
the root-level app log; rules are configurable; the LogStore browses the tree.
"""

import gzip
import json
import logging
import zipfile

import pytest

from simplyprint_ws_client import ClientSettings
from simplyprint_ws_client.common.logging import (
    configure_logging,
    LoggingConfig,
    LogNotFound,
    LogStore,
    RoutingHandler,
    RoutingRule,
    printer_logger_name,
    PRINTER_ROOT,
    scope_of,
)
from simplyprint_ws_client.common.logging.naming import decode_uid, encode_uid


def _record(name, message="msg", level=logging.INFO):
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


@pytest.mark.parametrize(
    "uid",
    ["simple", "a.b.c", "with space", "uuid/with/slash", "weird%.~-_chars"],
)
def test_uid_encoding_is_a_reversible_opaque_segment(uid):
    encoded = encode_uid(uid)
    assert "." not in encoded  # the dot is our separator; uid must not contain it
    assert decode_uid(encoded) == uid


def test_scope_of_printer_and_system():
    printer = _record(printer_logger_name("a.b.c", "mqtt"))
    system = _record("supervisor")
    assert scope_of(printer) == ("a.b.c", "a.b.c")  # uid round-trips through the name
    assert scope_of(system) == ("system", None)


def test_json_formatter_tags_scope():
    from simplyprint_ws_client.common.logging import JsonLogFormatter

    formatter = JsonLogFormatter()
    printer = json.loads(
        formatter.format(_record(printer_logger_name("p7", "mqtt"), "hi"))
    )
    assert printer["scope"] == "p7"
    assert printer["unique_id"] == "p7"
    assert printer["message"] == "hi"

    system = json.loads(formatter.format(_record("supervisor", "boot")))
    assert system["scope"] == "system"
    assert "unique_id" not in system


def _drain(handler):
    for h in handler._handlers.values():
        h.flush()


def test_routing_per_printer_and_system(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record(printer_logger_name("p7", "mqtt"), "from mqtt"))
    handler.emit(_record(printer_logger_name("p7"), "from base"))
    handler.emit(_record("supervisor", "from system"))
    _drain(handler)

    assert (tmp_path / "p7" / "mqtt.log").read_text().strip().endswith("from mqtt")
    assert (tmp_path / "p7" / "main.log").read_text().strip().endswith("from base")
    assert (tmp_path / "system.log").read_text().strip().endswith("from system")
    # No per-integration file leaked at the root.
    assert {p.name for p in tmp_path.glob("*.log")} == {"system.log"}
    handler.close()


def test_routing_camera_and_worker_get_their_own_scope(tmp_path):
    # Shared (non-printer) camera/worker loggers route to their own scope dirs,
    # never into the system catch-all.
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record("camera.pool", "camera line", logging.WARNING))
    handler.emit(_record("worker.pool", "worker line", logging.WARNING))
    handler.emit(_record("supervisor", "system line", logging.INFO))
    _drain(handler)

    assert (
        (tmp_path / "camera" / "camera.log").read_text().strip().endswith("camera line")
    )
    assert (
        (tmp_path / "workers" / "workers.log")
        .read_text()
        .strip()
        .endswith("worker line")
    )
    system = (tmp_path / "system.log").read_text()
    assert "system line" in system
    assert "camera line" not in system
    assert "worker line" not in system
    # Only the system file sits at the log root; camera/worker live in subdirs.
    assert {p.name for p in tmp_path.glob("*.log")} == {"system.log"}
    handler.close()


def test_routing_policy_filters_files_by_scope_and_noise(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record("supervisor", "system debug", logging.DEBUG))
    handler.emit(_record("supervisor", "system info", logging.INFO))
    handler.emit(_record("websockets.client", "websocket info", logging.INFO))
    handler.emit(_record("websockets.client", "websocket warning", logging.WARNING))
    handler.emit(_record("paho.mqtt.client", "paho info", logging.INFO))
    handler.emit(
        _record(printer_logger_name("p7", "mqtt"), "mqtt debug", logging.DEBUG)
    )
    _drain(handler)

    system = (tmp_path / "system.log").read_text()
    printer = (tmp_path / "p7" / "mqtt.log").read_text()
    assert "system info" in system
    assert "websocket warning" in system
    assert "system debug" not in system
    assert "websocket info" not in system
    # paho (the MQTT stack) is clamped to warnings like the other noisy roots.
    assert "paho info" not in system
    assert "mqtt debug" in printer
    handler.close()


def test_routing_custom_rule(tmp_path):
    # The "powerful" bit: route 'discovery' to its own JSON file; everything else default.
    config = LoggingConfig(
        log_dir=tmp_path,
        routes=[
            RoutingRule(
                "discovery",
                "discovery",
                "json",
                resolver=lambda _name: ("system", "discovery"),
            ),
        ],
    )
    handler = RoutingHandler(config)
    handler.emit(_record("discovery", "found a printer"))
    handler.emit(_record("supervisor", "boot"))
    _drain(handler)

    discovery_line = (tmp_path / "discovery.log").read_text().strip()
    assert json.loads(discovery_line)["message"] == "found a printer"
    assert (tmp_path / "system.log").read_text().strip().endswith("boot")
    handler.close()


def test_routing_dotted_uid_round_trips_to_one_dir(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record(printer_logger_name("a.b.c", "mqtt"), "x"))
    _drain(handler)
    # The dotted uid is one directory, not nested a/b/c.
    assert (tmp_path / "a.b.c" / "mqtt.log").is_file()
    handler.close()


def test_configure_logging_uses_fixed_system_stem(tmp_path):
    # The system log is the fixed, brand-free ``system.log`` -- never derived
    # from ``ClientSettings.name`` (so it no longer becomes ``BambuClient.log``).
    facility = configure_logging(
        ClientSettings(name="BambuClient"), LoggingConfig(log_dir=tmp_path)
    )
    try:
        logging.getLogger("supervisor").warning("boot")
    finally:
        facility.stop()
        logging.basicConfig(handlers=[], force=True)

    assert facility.config.system_log_stem == "system"
    assert (tmp_path / "system.log").read_text().strip().endswith("boot")
    assert not (tmp_path / "BambuClient.log").exists()


def test_configure_logging_sets_effective_levels_for_performance(tmp_path):
    facility = configure_logging(
        ClientSettings(name="BambuClient"), LoggingConfig(log_dir=tmp_path)
    )
    try:
        assert logging.getLogger().getEffectiveLevel() == logging.INFO
        assert logging.getLogger(PRINTER_ROOT).getEffectiveLevel() == logging.DEBUG
        assert (
            logging.getLogger("websockets.client").getEffectiveLevel()
            == logging.WARNING
        )
        assert (
            logging.getLogger("httpcore.connection").getEffectiveLevel()
            == logging.WARNING
        )
    finally:
        facility.stop()
        logging.basicConfig(handlers=[], force=True)


def test_policy_rejects_spam_before_record_allocation(tmp_path):
    facility = configure_logging(
        ClientSettings(name="BambuClient"), LoggingConfig(log_dir=tmp_path)
    )
    printer_name = printer_logger_name("p7", "mqtt")
    created = []
    original_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        created.append(args[0])
        return original_factory(*args, **kwargs)

    try:
        logging.setLogRecordFactory(factory)
        logging.getLogger("supervisor").debug("system debug")
        logging.getLogger("websockets.client").info("websocket info")
        assert created == []

        logging.getLogger(printer_name).debug("mqtt debug")
        assert created == [printer_name]
    finally:
        logging.setLogRecordFactory(original_factory)
        facility.stop()
        logging.basicConfig(handlers=[], force=True)


def test_stream_policy_keeps_console_operational(capsys, tmp_path):
    facility = configure_logging(
        ClientSettings(name="BambuClient"), LoggingConfig(log_dir=tmp_path)
    )
    try:
        logging.getLogger("supervisor").info("system info")
        logging.getLogger("supervisor").debug("system debug")
        logging.getLogger("websockets.client").info("websocket info")
        logging.getLogger("websockets.client").warning("websocket warning")
        logging.getLogger(printer_logger_name("p7", "mqtt")).debug("mqtt debug")
        logging.getLogger(printer_logger_name("p7", "mqtt")).warning("mqtt warning")
    finally:
        facility.stop()
        logging.basicConfig(handlers=[], force=True)

    stderr = capsys.readouterr().err
    assert "system info" in stderr
    assert "websocket warning" in stderr
    assert "mqtt warning" in stderr
    assert "system debug" not in stderr
    assert "websocket info" not in stderr
    assert "mqtt debug" not in stderr


def test_development_policy_keeps_noisy_debug_suppressed(tmp_path):
    facility = configure_logging(
        ClientSettings(name="BambuClient", development=True),
        LoggingConfig(log_dir=tmp_path),
    )
    try:
        assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
        assert logging.getLogger(PRINTER_ROOT).getEffectiveLevel() == logging.DEBUG
        assert (
            logging.getLogger("websockets.client").getEffectiveLevel()
            == logging.WARNING
        )
        logging.getLogger("supervisor").debug("system debug")
        logging.getLogger("websockets.client").debug("websocket debug")
    finally:
        facility.stop()
        logging.basicConfig(handlers=[], force=True)

    system = (tmp_path / "system.log").read_text()
    assert "system debug" in system
    assert "websocket debug" not in system


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_logstore_enumeration_and_read(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    _write(tmp_path / "system.log", "system line\n")
    _write(tmp_path / "system" / "system.log", "ignored nested line\n")
    _write(tmp_path / "p7" / "main.log", "a\nb\nc\n")

    assert {s.scope for s in store.list_scopes()} == {"system", "p7"}
    assert [f.name for f in store.list_files("system")] == ["system.log"]
    assert store.read_text("system", "system.log") == "system line\n"
    assert [f.name for f in store.list_files("p7")] == ["main.log"]
    assert store.read_text("p7", "main.log", tail_lines=1) == "c\n"


def test_logstore_reads_gzipped_backup(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    backup = tmp_path / "p7" / "main.log.1"
    backup.parent.mkdir(parents=True)
    with gzip.GzipFile(backup, "wb", mtime=0) as handle:
        handle.write(b"compressed line\n")
    assert "compressed line" in store.read_text("p7", "main.log.1")


def test_read_tail_lines_returns_last_n_complete_lines(tmp_path):
    # A file several 64 KB blocks long, so the backward read seeks land mid-line
    # and the partial leading fragment must be dropped (not returned as a line).
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    lines = [f"line-{i:05d}-{'x' * 180}" for i in range(1000)]
    _write(tmp_path / "p7" / "main.log", "\n".join(lines) + "\n")

    text, truncated = store.read_tail_lines("p7", "main.log", 5)
    assert text.splitlines() == lines[-5:]
    # No fragment: the first returned line is a whole line, not a tail of an earlier one.
    assert text.splitlines()[0] == "line-00995-" + "x" * 180
    assert truncated is True  # the start of the file was cut off


def test_read_tail_lines_whole_file_when_fewer_lines(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    _write(tmp_path / "p7" / "main.log", "a\nb\nc\n")
    # Asking for more lines than exist returns the whole file, with no line dropped.
    text, truncated = store.read_tail_lines("p7", "main.log", 100)
    assert text.splitlines() == ["a", "b", "c"]
    assert truncated is False  # whole file -> line 0 is real, not a fragment


def test_read_tail_lines_gzip_fallback(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    backup = tmp_path / "p7" / "main.log.1"
    backup.parent.mkdir(parents=True)
    with gzip.GzipFile(backup, "wb", mtime=0) as handle:
        handle.write(b"one\ntwo\nthree\n")
    # A gzipped backup can't be seeked, so it falls back to the whole-file tail.
    text, truncated = store.read_tail_lines("p7", "main.log.1", 2)
    assert text.splitlines() == ["two", "three"]
    assert truncated is True  # more lines existed than were returned


def test_logstore_traversal_guard(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    _write(tmp_path / "p7" / "main.log", "x\n")
    with pytest.raises(LogNotFound):
        store.resolve_file("p7", "../system.log")


def test_logstore_prune_and_compress(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    _write(tmp_path / "system.log", "s\n")
    _write(tmp_path / "active" / "main.log", "a\n")
    _write(tmp_path / "stale" / "main.log", "b\n")
    _write(tmp_path / "active" / "main.log.1", "old\n")

    store.prune_unused_scopes(["active"])
    assert (tmp_path / "system.log").is_file()
    assert (tmp_path / "active").is_dir()
    assert not (tmp_path / "stale").exists()

    store.compress_rotated_files()
    with open(tmp_path / "active" / "main.log.1", "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"


def test_routing_clear_file_truncates_live_file(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record(printer_logger_name("p7"), "first line"))
    _drain(handler)
    path = tmp_path / "p7" / "main.log"
    assert path.read_text().strip().endswith("first line")

    # The active file is truncated in place (not unlinked) and stays present.
    assert handler.clear_file(str(path)) is True
    assert path.read_text() == ""

    # Logging keeps working and the next record starts a fresh file, with no
    # sparse gap from a stale write position.
    handler.emit(_record(printer_logger_name("p7"), "second line"))
    _drain(handler)
    text = path.read_text()
    assert "first line" not in text
    assert text.strip().endswith("second line")
    handler.close()


def test_routing_clear_file_unknown_path_returns_false(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    # A rotated backup no handler holds isn't owned -> caller should unlink it.
    assert handler.clear_file(str(tmp_path / "p7" / "main.log.1")) is False
    handler.close()


def test_logstore_delete_file_truncates_active_log(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record(printer_logger_name("p7"), "live line"))
    _drain(handler)
    path = tmp_path / "p7" / "main.log"

    store = LogStore(
        LoggingConfig(log_dir=tmp_path),
        on_scope_pruned=handler.close_scope,
        on_file_cleared=handler.clear_file,
    )
    store.delete_file("p7", "main.log")
    # Active log is truncated, not removed: unlinking an open handle fails on
    # Windows, and logging must be able to keep appending.
    assert path.is_file()
    assert path.read_text() == ""
    handler.close()


def test_logstore_delete_file_unlinks_inactive_backup(tmp_path):
    _write(tmp_path / "p7" / "main.log.1", "old\n")
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    store = LogStore(
        LoggingConfig(log_dir=tmp_path),
        on_file_cleared=handler.clear_file,
    )
    store.delete_file("p7", "main.log.1")
    assert not (tmp_path / "p7" / "main.log.1").exists()
    handler.close()


def test_logstore_delete_scope_closes_handlers_before_rmtree(tmp_path):
    handler = RoutingHandler(LoggingConfig(log_dir=tmp_path))
    handler.emit(_record(printer_logger_name("p7"), "live line"))
    _drain(handler)

    closed = []

    def on_pruned(scope):
        closed.append(scope)
        handler.close_scope(scope)

    store = LogStore(LoggingConfig(log_dir=tmp_path), on_scope_pruned=on_pruned)
    store.delete_scope("p7")
    assert closed == ["p7"]  # handles released before the directory is removed
    assert not (tmp_path / "p7").exists()
    handler.close()


def test_logstore_bundle_and_system_undeletable(tmp_path):
    store = LogStore(LoggingConfig(log_dir=tmp_path))
    _write(tmp_path / "system.log", "s\n")
    _write(tmp_path / "p7" / "main.log", "a\n")

    with zipfile.ZipFile(store.bundle_zip()) as archive:
        names = set(archive.namelist())
    assert {"system/system.log", "p7/main.log"} <= names

    with pytest.raises(ValueError):
        store.delete_scope("system")
