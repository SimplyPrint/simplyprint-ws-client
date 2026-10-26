"""The SQLite store: schema migrations, coalesced writes, the change feed,
crash safety, concurrency and recovery (docs/persistence-design.md)."""

import sqlite3
import threading
from pathlib import Path

import pytest

from simplyprint_ws_client.core.store import (
    SCHEMA,
    ChangeSet,
    SchemaStep,
    Store,
    StoreTooNew,
    plan,
)
from simplyprint_ws_client.core.store import store as store_module
from simplyprint_ws_client.core.store.schema import ARCHIVE_KEEP


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock):
    s = Store.open(tmp_path / "state.db", start=False, clock=clock)
    yield s
    s.close()


def _kv(value):
    return {"value": value}


def _raw(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


# -- schema ---------------------------------------------------------------------


def test_a_new_store_is_at_the_newest_schema(store):
    conn = _raw(store.path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(SCHEMA)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "meta", "migrations", "printers", "accounts", "settings", "kv",
        "legacy_files", "row_archive",
    } <= tables  # fmt: skip
    assert [r[0] for r in conn.execute("SELECT version FROM migrations")] == [1]
    conn.close()


def test_reopening_runs_no_step_twice(tmp_path):
    Store.open(tmp_path / "s.db", start=False).close()
    Store.open(tmp_path / "s.db", start=False).close()
    conn = _raw(tmp_path / "s.db")
    assert conn.execute("SELECT count(*) FROM migrations").fetchone()[0] == 1
    conn.close()


def test_steps_run_by_version_whatever_order_they_are_declared_in():
    one, two, three = (SchemaStep(v, ()) for v in (1, 2, 3))
    assert [s.version for s in plan(0, [three, one, two])] == [1, 2, 3]
    assert [s.version for s in plan(2, [three, one, two])] == [3]
    assert plan(3, [one, two, three]) == []


@pytest.mark.parametrize("versions", [(1, 3), (2,), (1, 1), (0, 1)])
def test_steps_must_be_numbered_without_gaps(versions):
    with pytest.raises(ValueError):
        plan(0, [SchemaStep(v, ()) for v in versions])


def test_a_database_from_a_newer_build_is_refused_not_rewritten(tmp_path):
    path = tmp_path / "s.db"
    Store.open(path, start=False).close()
    conn = _raw(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    with pytest.raises(StoreTooNew):
        Store.open(path, start=False)
    assert path.read_bytes() == before


def test_a_row_migration_is_pure_archives_the_old_row_and_bumps_user_version(tmp_path):
    path = tmp_path / "s.db"
    s = Store.open(path, start=False)
    with s.transaction() as txn:
        txn.put("kv", ("acme", "plugin", "k"), _kv({"old": 1}))
        txn.put("kv", ("acme", "plugin", "same"), _kv({"new": 2}))
    s.close()

    def rename(row):
        value = dict(row["value"])
        if "old" in value:
            value["new"] = value.pop("old")
        return {**row, "value": value}

    v2 = SchemaStep(2, ("CREATE TABLE extra (x)",), rows={"kv": rename})
    s = Store.open(path, start=False, migrations=SCHEMA + (v2,))
    try:
        assert s.get("kv", ("acme", "plugin", "k"))["value"] == {"new": 1}
        assert s.get("kv", ("acme", "plugin", "same"))["value"] == {"new": 2}
        conn = _raw(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        archived = conn.execute("SELECT tbl, key, data FROM row_archive").fetchall()
        # Only the row that changed is archived, as it was before.
        assert archived == [("kv", '["acme","plugin","k"]', '{"value":{"old":1}}')]
        conn.close()
    finally:
        s.close()


def test_a_failing_step_leaves_the_previous_version(tmp_path):
    path = tmp_path / "s.db"
    Store.open(path, start=False).close()

    def boom(row):
        raise RuntimeError("bad migration")

    bad = SchemaStep(2, ("CREATE TABLE extra (x)",), rows={"settings": boom})
    s = Store.open(path, start=False)
    with s.transaction() as txn:
        txn.put("settings", ("web", "port"), _kv(8080))
    s.close()
    with pytest.raises(RuntimeError):
        Store.open(path, start=False, migrations=SCHEMA + (bad,))
    conn = _raw(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert "extra" not in names
    conn.close()


def test_durability_sets_the_journal_and_sync_mode(tmp_path):
    durable = Store.open(tmp_path / "state.db", start=False)
    cache = Store.open(tmp_path / "cache.db", durability="cache", start=False)
    try:
        assert durable._wconn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert durable._wconn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert cache._wconn.execute("PRAGMA synchronous").fetchone()[0] == 0  # OFF
        assert cache.backups is None
    finally:
        durable.close()
        cache.close()


# -- transactions ---------------------------------------------------------------


def test_put_get_delete_and_the_revision(store):
    assert store.revision == 0
    with store.transaction() as txn:
        assert txn.put("printers", "u1", {"store": "p", "pk": 1, "sk": "t", "data": {"id": 1}})
    assert store.revision == 1
    row = store.get("printers", "u1")
    assert row == {
        "unique_id": "u1", "store": "p", "pk": 1, "sk": "t", "data": {"id": 1},
        "schema": 1, "state": "ok", "error": None, "rev": 1,
    }  # fmt: skip
    with store.transaction() as txn:
        assert txn.delete("printers", "u1")
        assert not txn.delete("printers", "u1")
    assert store.get("printers", "u1") is None
    assert store.revision == 2


def test_an_unchanged_put_writes_nothing_and_keeps_the_revision(store):
    row = {"store": "p", "data": {"id": 1}}
    with store.transaction() as txn:
        txn.put("printers", "u1", row)
    with store.transaction() as txn:
        assert not txn.put("printers", "u1", dict(row))
    assert store.revision == 1
    assert store.get("printers", "u1")["rev"] == 1


def test_one_transaction_is_one_revision(store):
    with store.transaction() as txn:
        for n in range(5):
            txn.put("kv", ("d", "plugin", str(n)), _kv(n))
    assert store.revision == 1
    assert {r["rev"] for r in store.rows("kv")} == {1}


def test_a_raising_block_rolls_back(store):
    with pytest.raises(KeyError):
        with store.transaction() as txn:
            txn.put("kv", ("d", "plugin", "k"), _kv(1))
            raise KeyError("x")
    assert store.rows("kv") == []
    assert store.revision == 0


def test_rows_filter_by_column(store):
    with store.transaction() as txn:
        txn.put("printers", "a", {"store": "one", "data": {}})
        txn.put("printers", "b", {"store": "two", "data": {}})
    assert [r["unique_id"] for r in store.rows("printers", store="two")] == ["b"]
    with pytest.raises(ValueError):
        store.rows("printers", nope=1)


def test_keys_must_match_the_table(store):
    with pytest.raises(ValueError):
        with store.transaction() as txn:
            txn.put("kv", "only-one-part", _kv(1))


def test_the_archive_keeps_the_last_versions_of_a_printer(store):
    for n in range(ARCHIVE_KEEP + 3):
        with store.transaction() as txn:
            txn.put("printers", "u1", {"store": "p", "data": {"n": n}})
    with store.transaction() as txn:
        txn.delete("printers", "u1")
    conn = _raw(store.path)
    kept = [r[0] for r in conn.execute("SELECT data FROM row_archive ORDER BY id")]
    conn.close()
    assert len(kept) == ARCHIVE_KEEP
    assert '"n":%d' % (ARCHIVE_KEEP + 2) in kept[-1]  # the version deleted last


def test_removing_a_printer_removes_its_scoped_plugin_data(store):
    with store.transaction() as txn:
        txn.put("printers", "u1", {"store": "p", "data": {}})
        txn.put("kv", ("acme", "printer:u1", "k"), _kv(1))
        txn.put("kv", ("acme", "printer:u2", "k"), _kv(2))
        txn.put("kv", ("acme", "plugin", "k"), _kv(3))
    seen = []
    store.subscribe("kv", seen.append)
    with store.transaction() as txn:
        txn.delete("printers", "u1")
    assert sorted(r["scope"] for r in store.rows("kv")) == ["plugin", "printer:u2"]
    assert seen[0].removed == (("acme", "printer:u1", "k"),)


# -- the coalescing writer ------------------------------------------------------


def test_a_mark_waits_out_the_debounce(store, clock):
    store.writer.mark("kv", ("d", "plugin", "k"), lambda: _kv(1))
    assert store.writer.next_deadline() == 101.0
    clock.t = 100.9
    assert not store.writer.is_due()
    clock.t = 101.0
    assert store.writer.is_due()


def test_repeated_marks_push_the_debounce_but_never_past_the_cap(store, clock):
    start = clock.t
    for _ in range(20):
        store.writer.mark("kv", ("d", "plugin", "k"), lambda: _kv(1))
        clock.t += 0.5
    # Marked every 0.5 s for 10 s: the cap made it due 5 s after the first.
    assert store.writer.next_deadline() == start + 5.0


def test_an_urgent_mark_is_due_at_once(store, clock):
    store.writer.mark("kv", ("d", "plugin", "a"), lambda: _kv(1))
    store.writer.mark("kv", ("d", "plugin", "b"), lambda: _kv(2), urgent=True)
    assert store.writer.is_due()


def test_a_batch_is_one_transaction_and_the_newest_produce_wins(store):
    store.writer.mark("kv", ("d", "plugin", "a"), lambda: _kv("old"))
    store.writer.mark("kv", ("d", "plugin", "a"), lambda: _kv("new"))
    store.writer.mark("kv", ("d", "plugin", "b"), lambda: _kv(2))
    assert store.writer.flush_now() == 2
    assert store.revision == 1
    assert store.get("kv", ("d", "plugin", "a"))["value"] == "new"
    assert not store.writer.dirty


def test_a_one_column_key_is_the_same_row_as_its_tuple(store):
    store.writer.mark("printers", "u1", lambda: {"store": "p", "data": {"n": 1}})
    store.writer.mark("printers", ("u1",), lambda: {"store": "p", "data": {"n": 2}})
    assert store.writer.pending() == [("printers", ("u1",))]


def test_an_unchanged_row_is_skipped_by_hash(store):
    value = {"store": "p", "data": {"id": 1, "token": "t"}}
    store.writer.mark("printers", "u1", lambda: value)
    store.writer.flush_now()
    store.writer.mark("printers", "u1", lambda: dict(value))
    assert store.writer.flush_now() == 0
    assert store.revision == 1


def test_produce_none_deletes(store):
    store.writer.mark("kv", ("d", "plugin", "a"), lambda: _kv(1), urgent=True)
    store.writer.flush_now()
    store.writer.mark("kv", ("d", "plugin", "a"), lambda: None, urgent=True)
    store.writer.flush_now()
    assert store.get("kv", ("d", "plugin", "a")) is None


def test_a_row_that_cannot_be_produced_keeps_its_stored_version(store):
    store.writer.mark("kv", ("d", "plugin", "a"), lambda: _kv(1))
    store.writer.flush_now()

    def broken():
        raise ValueError("cannot serialize")

    store.writer.mark("kv", ("d", "plugin", "a"), broken)
    store.writer.mark("kv", ("d", "plugin", "b"), lambda: _kv(object()))  # not JSON
    store.writer.mark("kv", ("d", "plugin", "c"), lambda: _kv(3))
    store.writer.flush_now()
    assert store.get("kv", ("d", "plugin", "a"))["value"] == 1
    assert store.get("kv", ("d", "plugin", "b")) is None
    assert store.get("kv", ("d", "plugin", "c"))["value"] == 3


def test_a_row_that_failed_to_produce_is_retried_with_backoff(store, clock):
    # The mark was taken for the batch; dropping it would lose the change
    # for good (nothing marks the row again until it changes again).
    key = ("d", "plugin", "a")
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("not yet")
        return _kv("made")

    store.writer.mark("kv", key, flaky, urgent=True)
    store.writer.flush_now()
    assert store.get("kv", key) is None
    assert store.writer.pending() == [("kv", key)]
    assert not store.writer.is_due()  # backing off, not spinning
    clock.t += 60.0
    assert store.writer.is_due()
    store.writer.flush_now()
    assert store.get("kv", key)["value"] == "made"
    assert not store.writer.dirty


def test_an_unknown_table_fails_at_mark_time(store):
    with pytest.raises(KeyError):
        store.writer.mark("nope", "k", lambda: {})


def test_the_store_thread_writes_an_urgent_mark(tmp_path):
    s = Store.open(tmp_path / "state.db", backups=tmp_path / "b")
    try:
        done = threading.Event()
        s.subscribe("kv", lambda change: done.set())
        s.writer.mark("kv", ("d", "plugin", "k"), lambda: _kv(1), urgent=True)
        assert done.wait(5)
        assert s.get("kv", ("d", "plugin", "k"))["value"] == 1
    finally:
        s.close()


def test_close_writes_what_is_pending(tmp_path):
    s = Store.open(tmp_path / "state.db", backups=tmp_path / "b")
    s.writer.mark("kv", ("d", "plugin", "k"), lambda: _kv("last"))
    s.close()
    again = Store.open(tmp_path / "state.db", start=False)
    try:
        assert again.get("kv", ("d", "plugin", "k"))["value"] == "last"
    finally:
        again.close()


def test_a_held_writer_writes_pending_first_then_nothing_until_released(tmp_path):
    # A restore holds the writer from before its commit until the running
    # code follows; a mark queued meanwhile produces after the hold, so it
    # reads memory as it is then, never as it was before the restore.
    s = Store.open(tmp_path / "state.db", backups=tmp_path / "b")
    key = ("d", "plugin", "k")
    memory = {"value": "before"}
    try:
        s.writer.mark("kv", key, lambda: _kv(memory["value"]))
        with s.writes_held():
            assert s.get("kv", key)["value"] == "before"  # pending went first
            s.writer.mark("kv", key, lambda: _kv(memory["value"]), urgent=True)
            waited = threading.Event()

            def needs_it_on_disk():
                s.writer.flush_now()
                waited.set()

            threading.Thread(target=needs_it_on_disk, daemon=True).start()
            # Neither the store's thread nor flush_now writes during the hold.
            assert not waited.wait(0.3)
            # A transaction (the restore's own write) still goes through.
            with s.transaction() as txn:
                txn.put("kv", key, _kv("restored"))
            memory["value"] = "restored"  # the running code followed
            assert s.get("kv", key)["value"] == "restored"
        assert waited.wait(5)
        assert s.get("kv", key)["value"] == "restored"
    finally:
        s.close()


# -- crash safety ---------------------------------------------------------------


class _WriterKilled(BaseException):
    """Stands in for the writer dying mid-batch (a signal, an interpreter
    teardown): not an Exception, so nothing on the way catches it."""


def test_a_writer_killed_mid_batch_commits_none_of_it(tmp_path):
    path = tmp_path / "state.db"
    s = Store.open(path, start=False)
    with s.transaction() as txn:
        txn.put("kv", ("d", "plugin", "kept"), _kv("before"))

    def killed():
        raise _WriterKilled()

    s.writer.mark("kv", ("d", "plugin", "kept"), lambda: _kv("torn"))
    s.writer.mark("kv", ("d", "plugin", "dies"), killed)
    s.writer.mark("kv", ("d", "plugin", "after"), lambda: _kv("never"))
    with pytest.raises(_WriterKilled):
        s.writer.flush_now()

    # Nothing of the batch is visible, to this store or to a new process.
    assert s.get("kv", ("d", "plugin", "kept"))["value"] == "before"
    other = sqlite3.connect(str(path))
    assert other.execute("SELECT count(*) FROM kv").fetchone()[0] == 1
    other.close()
    assert s.revision == 1
    # The batch went back to the queue: nothing pending is lost.
    assert len(s.writer.pending()) == 3
    s.writer.mark("kv", ("d", "plugin", "dies"), lambda: _kv("fine now"))
    s.writer.flush_now()
    assert s.get("kv", ("d", "plugin", "kept"))["value"] == "torn"
    s.close()


class _CommitFails:
    """The writer connection, except that COMMIT fails like a torn fsync."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, *args):
        if sql == "COMMIT":
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_failed_commit_rolls_back_and_requeues(tmp_path):
    path = tmp_path / "state.db"
    s = Store.open(path, start=False)
    real = s._wconn
    s._wconn = _CommitFails(real)
    seen = []
    s.subscribe("kv", seen.append)
    s.writer.mark("kv", ("d", "plugin", "k"), lambda: _kv(1))
    with pytest.raises(sqlite3.OperationalError):
        s.writer.flush_now()
    assert not real.in_transaction
    assert seen == []  # nothing published for what was not committed
    assert s.revision == 0
    s._wconn = real
    assert s.writer.flush_now() == 1
    assert [c.added for c in seen] == [(("d", "plugin", "k"),)]
    s.close()


def test_an_uncommitted_transaction_from_a_dead_process_is_not_visible(tmp_path):
    path = tmp_path / "state.db"
    Store.open(path, start=False).close()
    dead = sqlite3.connect(str(path), isolation_level=None)
    dead.execute("BEGIN IMMEDIATE")
    dead.execute(
        "INSERT INTO kv (domain, scope, key, value, rev) VALUES ('d','plugin','k','1',1)"
    )
    dead.close()  # the process died before COMMIT
    s = Store.open(path, start=False)
    try:
        assert s.rows("kv") == []
        assert s.recovery is None
    finally:
        s.close()


# -- concurrency ----------------------------------------------------------------


def test_writes_from_threads_while_reading(tmp_path):
    s = Store.open(tmp_path / "state.db", backups=tmp_path / "b")
    writers, rounds = 6, 40
    errors = []
    stop = threading.Event()

    def write(n):
        try:
            for i in range(rounds):
                value = {"n": i, "double": i * 2}
                if i % 2:
                    s.writer.mark("kv", ("d", "plugin", str(n)), lambda v=value: _kv(v))
                else:
                    with s.transaction() as txn:
                        txn.put("kv", ("d", "plugin", str(n)), _kv(value))
        except Exception as exc:  # noqa: BLE001 - any failure fails the test
            errors.append(exc)

    def read():
        try:
            while not stop.is_set():
                for row in s.rows("kv"):
                    # A reader never sees half a row.
                    assert row["value"]["double"] == row["value"]["n"] * 2
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    threads = [threading.Thread(target=write, args=(n,)) for n in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s.writer.flush_now()
    stop.set()
    reader.join()
    try:
        assert errors == []
        rows = s.rows("kv")
        assert len(rows) == writers
        assert {r["value"]["n"] for r in rows} == {rounds - 1}
    finally:
        s.close()


# -- the change feed ------------------------------------------------------------


def test_local_commits_are_published_with_their_revision(store):
    seen = []
    unsubscribe = store.subscribe("printers", seen.append)
    with store.transaction() as txn:
        txn.put("printers", "u1", {"store": "p", "data": {"n": 1}})
    with store.transaction() as txn:
        txn.put("printers", "u1", {"store": "p", "data": {"n": 2}})
        txn.put("printers", "u2", {"store": "p", "data": {}})
    with store.transaction() as txn:
        txn.delete("printers", "u2")
    assert seen == [
        ChangeSet("printers", ("u1",), (), (), "local", 1),
        ChangeSet("printers", ("u2",), ("u1",), (), "local", 2),
        ChangeSet("printers", (), (), ("u2",), "local", 3),
    ]
    unsubscribe()
    with store.transaction() as txn:
        txn.put("printers", "u3", {"store": "p", "data": {}})
    assert len(seen) == 3


def test_an_unchanged_commit_publishes_nothing(store):
    seen = []
    store.subscribe("kv", seen.append)
    with store.transaction() as txn:
        txn.put("kv", ("d", "plugin", "k"), _kv(1))
    with store.transaction() as txn:
        txn.put("kv", ("d", "plugin", "k"), _kv(1))
    assert len(seen) == 1


def test_a_failing_subscriber_does_not_stop_the_others(store):
    seen = []

    def bad(change):
        raise RuntimeError("subscriber bug")

    store.subscribe("kv", bad)
    store.subscribe("kv", seen.append)
    with store.transaction() as txn:
        txn.put("kv", ("d", "plugin", "k"), _kv(1))
    assert len(seen) == 1


def test_another_process_writes_reach_the_feed_once(tmp_path):
    path = tmp_path / "state.db"
    service = Store.open(path, start=False)
    cli = Store.open(path, start=False)
    try:
        seen = []
        service.subscribe("printers", seen.append)
        assert service.poll_external() == []

        with cli.transaction() as txn:
            txn.put("printers", "u1", {"store": "p", "data": {"name": "edited"}})
        changes = service.poll_external()
        assert changes == [ChangeSet("printers", ("u1",), (), (), "external", 1)]
        assert seen == changes
        assert service.revision == 1
        # Polled once; nothing new until someone writes again.
        assert service.poll_external() == []

        # The service's own write is known by its revision, not re-reported...
        with service.transaction() as txn:
            txn.put("printers", "u2", {"store": "p", "data": {}})
        assert service.poll_external() == []
        # ...while the other process hears about it as external.
        assert cli.poll_external()[0].added == ("u2",)

        with cli.transaction() as txn:
            txn.put("printers", "u1", {"store": "p", "data": {"name": "again"}})
            txn.delete("printers", "u2")
        changes = service.poll_external()
        assert changes[0].changed == ("u1",) and changes[0].removed == ("u2",)
    finally:
        cli.close()
        service.close()


def test_an_external_change_invalidates_the_unchanged_write_check(tmp_path):
    path = tmp_path / "state.db"
    service = Store.open(path, start=False)
    cli = Store.open(path, start=False)
    try:
        # kv: no archive, so the check relies on the cached hash alone.
        key = ("d", "plugin", "k")
        with service.transaction() as txn:
            txn.put("kv", key, _kv("mine"))
        with cli.transaction() as txn:
            txn.put("kv", key, _kv("theirs"))
        service.poll_external()
        # Writing our old value again is a real change now, not a skip.
        with service.transaction() as txn:
            assert txn.put("kv", key, _kv("mine"))
        assert service.get("kv", key)["value"] == "mine"
    finally:
        cli.close()
        service.close()


def test_an_external_change_is_not_skipped_before_the_poll_sees_it(tmp_path):
    # The poll runs every 2 s; a write in between must not trust the hash
    # cached before another process changed the row.
    path = tmp_path / "state.db"
    service = Store.open(path, start=False)
    cli = Store.open(path, start=False)
    try:
        key = ("d", "plugin", "k")
        with service.transaction() as txn:
            txn.put("kv", key, _kv("mine"))
        with cli.transaction() as txn:
            txn.put("kv", key, _kv("theirs"))
        # No poll_external() here.
        service.writer.mark("kv", key, lambda: _kv("mine"))
        service.writer.flush_now()
        assert service.get("kv", key)["value"] == "mine"
    finally:
        cli.close()
        service.close()


# -- snapshots, export, import ----------------------------------------------------


def test_the_daily_snapshot_keeps_five(tmp_path):
    s = Store.open(tmp_path / "state.db", start=False, backups=tmp_path / "backups")
    try:
        day = 24 * 3600.0
        base = 1_800_000_000.0
        for n in range(7):
            s.maintain(now=base + n * day)
        names = sorted(p.name for p in (tmp_path / "backups").glob("*.db"))
        assert len(names) == 5
        assert names == sorted(names)
        # The same day again takes no second snapshot.
        s.maintain(now=base + 6 * day + 60)
        assert len(list((tmp_path / "backups").glob("*.db"))) == 5
        assert s.fault is None
    finally:
        s.close()


def test_a_snapshot_is_a_complete_database(store, tmp_path):
    with store.transaction() as txn:
        txn.put("kv", ("d", "plugin", "k"), _kv(1))
    target = store.snapshot_to(tmp_path / "copy.db")
    copy = Store.open(target, start=False)
    try:
        assert copy.get("kv", ("d", "plugin", "k"))["value"] == 1
    finally:
        copy.close()


def test_export_and_import_round_trip_without_secrets(tmp_path):
    a = Store.open(tmp_path / "a.db", start=False)
    b = Store.open(tmp_path / "b.db", start=False)
    try:
        with a.transaction() as txn:
            txn.put("printers", "u1", {"store": "p", "pk": 1, "sk": "t", "data": {"x": 1}})
            txn.put("accounts", ("cloud", "42"), {"secondary": "eu", "data": {"id": 42}})
            txn.put("settings", ("web", "port"), _kv(8080))
            txn.put("settings", ("host_secrets", "secret_key"), _kv("s3cret"))
        exported = a.export_json()
        assert all(
            r["namespace"] != "host_secrets" for r in exported["tables"]["settings"]
        )
        with b.transaction() as txn:
            txn.put("printers", "stale", {"store": "p", "data": {}})
            txn.put("settings", ("host_secrets", "secret_key"), _kv("b's own"))
        b.import_json(exported, replace=True)
        assert b.get("printers", "stale") is None
        assert b.get("printers", "u1")["data"] == {"x": 1}
        assert b.get("accounts", ("cloud", "42"))["secondary"] == "eu"
        assert b.get("settings", ("host_secrets", "secret_key"))["value"] == "b's own"
        assert b.import_json(exported, replace=True) == 0  # idempotent
        with pytest.raises(ValueError):
            b.import_json({"format": 99}, replace=False)
    finally:
        a.close()
        b.close()


def test_shared_returns_one_store_per_file(tmp_path):
    first = Store.shared(tmp_path / "state.db")
    try:
        assert Store.shared(tmp_path / "." / "state.db") is first
        with pytest.raises(ValueError):
            Store.shared(tmp_path / "state.db", durability="cache")
    finally:
        first.close()
    reopened = Store.shared(tmp_path / "state.db")
    assert reopened is not first
    reopened.close()


# -- recovery -------------------------------------------------------------------


def _filled(path: Path, backups: Path) -> None:
    s = Store.open(path, start=False, backups=backups)
    with s.transaction() as txn:
        txn.put("printers", "old", {"store": "p", "data": {"in": "snapshot"}})
    s.snapshot_to(backups / "state-2026-10-01.db")
    with s.transaction() as txn:
        txn.put("printers", "new", {"store": "p", "data": {"in": "db only"}})
    s.close()


def test_a_garbage_file_is_moved_aside_and_the_snapshot_fills_in(tmp_path):
    path, backups = tmp_path / "state.db", tmp_path / "backups"
    _filled(path, backups)
    garbage = b"this is not a database" * 300
    path.write_bytes(garbage)

    s = Store.open(path, start=False, backups=backups)
    try:
        report = s.recovery
        assert report is not None
        assert report.moved_to.read_bytes() == garbage  # nothing deleted
        assert report.snapshot == backups / "state-2026-10-01.db"
        assert report.from_snapshot["printers"] == 1
        assert [r["unique_id"] for r in s.rows("printers")] == ["old"]
        # New writes get revisions past every recovered row's.
        with s.transaction() as txn:
            txn.put("printers", "after", {"store": "p", "data": {}})
        assert s.get("printers", "after")["rev"] > s.get("printers", "old")["rev"]
    finally:
        s.close()


def test_readable_rows_are_kept_before_the_snapshot_is_used(tmp_path, monkeypatch):
    path, backups = tmp_path / "state.db", tmp_path / "backups"
    _filled(path, backups)
    real = Store._check
    calls = []

    def failing_once(conn):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.DatabaseError("quick_check: *** in database main ***")
        real(conn)

    monkeypatch.setattr(Store, "_check", staticmethod(failing_once))
    s = Store.open(path, start=False, backups=backups)
    try:
        report = s.recovery
        assert report.from_damaged["printers"] == 2
        assert report.from_snapshot.get("printers", 0) == 0  # nothing was missing
        assert {r["unique_id"] for r in s.rows("printers")} == {"old", "new"}
        assert "quick_check" in report.reason
    finally:
        s.close()


def test_a_table_read_completely_is_not_topped_up_from_the_snapshot(
    tmp_path, monkeypatch
):
    # A printer deleted after the last snapshot must stay deleted: only a
    # table the damaged file could not fully give is filled in.
    path, backups = tmp_path / "state.db", tmp_path / "backups"
    _filled(path, backups)
    s = Store.open(path, start=False, backups=backups)
    with s.transaction() as txn:
        txn.delete("printers", "old")
    s.close()
    real = Store._check
    calls = []

    def failing_once(conn):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.DatabaseError("quick_check: *** in database main ***")
        real(conn)

    monkeypatch.setattr(Store, "_check", staticmethod(failing_once))
    s = Store.open(path, start=False, backups=backups)
    try:
        assert [r["unique_id"] for r in s.rows("printers")] == ["new"]
        assert s.recovery.unreadable_tables == []
        assert s.recovery.from_snapshot.get("printers", 0) == 0
    finally:
        s.close()


def test_a_locked_database_is_not_treated_as_damaged(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    Store.open(path, start=False).close()

    def locked(conn):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Store, "_check", staticmethod(locked))
    with pytest.raises(sqlite3.OperationalError):
        Store.open(path, start=False)
    assert path.exists()
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_a_damaged_cache_starts_over_without_snapshots(tmp_path):
    path = tmp_path / "cache.db"
    Store.open(path, durability="cache", start=False).close()
    path.write_bytes(b"not sqlite at all " * 256)
    s = Store.open(path, durability="cache", start=False)
    try:
        assert s.recovery is not None and s.recovery.snapshot is None
        assert s.rows("kv") == []
    finally:
        s.close()


def test_the_atexit_hook_closes_open_stores(tmp_path):
    s = Store.open(tmp_path / "state.db", start=False)
    s.writer.mark("kv", ("d", "plugin", "k"), lambda: _kv(1))
    store_module._close_open_stores()
    assert s.closed
    again = Store.open(tmp_path / "state.db", start=False)
    try:
        assert again.get("kv", ("d", "plugin", "k"))["value"] == 1
    finally:
        again.close()
