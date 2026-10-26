"""StoreConfigManager: what the store adds over the file managers."""

from typing import ClassVar

import pytest

from simplyprint_ws_client.core.config import Config, PrinterConfig, StoreConfigManager
from simplyprint_ws_client.core.store import Store


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

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


def _manager(store, tmp_path, config_t=PrinterConfig, name="printers"):
    return StoreConfigManager(
        name=name, config_t=config_t, base_directory=str(tmp_path), store=store
    )


def _put_printer(store, unique_id, data, *, schema=1, name="printers"):
    with store.transaction() as txn:
        txn.put(
            "printers",
            unique_id,
            {"store": name, "pk": data.get("id"), "sk": data.get("token"),
             "data": data, "schema": schema},
        )  # fmt: skip


def test_unknown_fields_survive_a_rewrite(store, tmp_path):
    _put_printer(
        store,
        "u1",
        {"id": 1, "token": "t", "unique_id": "u1", "future_field": {"x": 1}},
    )
    manager = _manager(store, tmp_path)
    (config,) = manager.get_all()
    assert not hasattr(config, "future_field")

    config.name = "Renamed"
    manager.flush(config)

    data = store.get("printers", "u1")["data"]
    assert data["future_field"] == {"x": 1}
    assert data["name"] == "Renamed"


def test_a_field_change_waits_out_the_debounce(store, tmp_path, clock):
    manager = _manager(store, tmp_path)
    config = PrinterConfig(id=1, token="t", unique_id="u1", name="before")
    manager.persist(config)
    manager.flush()  # an add: on disk at once
    assert store.get("printers", "u1")["data"]["name"] == "before"

    config.name = "after"
    manager.flush()
    assert store.get("printers", "u1")["data"]["name"] == "before"
    assert not store.writer.is_due()
    clock.t += 1.0
    assert store.writer.is_due()
    store.writer.flush_now()
    assert store.get("printers", "u1")["data"]["name"] == "after"


def test_a_load_sees_this_process_pending_changes(store, tmp_path):
    manager = _manager(store, tmp_path)
    config = PrinterConfig(id=1, token="t", unique_id="u1", name="before")
    manager.persist(config)
    manager.flush()
    config.name = "pending"
    manager.flush()

    (reloaded,) = _manager(store, tmp_path).get_all()
    assert reloaded.name == "pending"


def test_a_config_without_unique_id_gets_one_and_keeps_it(store, tmp_path):
    manager = _manager(store, tmp_path)
    config = PrinterConfig(id=5, token="t")
    manager.persist(config)
    manager.flush()
    assigned = config.unique_id
    assert assigned
    config.name = "x"
    manager.flush()
    store.writer.flush_now()
    assert [r["unique_id"] for r in store.rows("printers")] == [assigned]


def test_an_unreadable_row_is_kept_and_marked_never_deleted(store, tmp_path):
    raw = {"id": "not a number", "token": "t", "unique_id": "bad"}
    _put_printer(store, "bad", raw)
    manager = _manager(store, tmp_path)
    assert manager.get_all() == []
    (entry,) = manager.unreadable
    assert entry.label == "bad" and entry.raw == raw

    manager.flush()  # holds nothing: must not delete what it could not read
    store.writer.flush_now()
    row = store.get("printers", "bad")
    assert row["data"] == raw
    assert row["state"] == "unreadable" and row["error"]


class _V2(PrinterConfig):
    config_version: ClassVar[int] = 2

    @classmethod
    def upgrade_config(cls, data, from_version):
        assert from_version == 1
        data = dict(data)
        data["name"] = data.pop("old_name", None)
        return data


def test_an_older_row_is_upgraded_and_the_old_one_archived(store, tmp_path):
    _put_printer(store, "u1", {"id": 1, "token": "t", "unique_id": "u1", "old_name": "N"})
    manager = _manager(store, tmp_path, config_t=_V2)
    (config,) = manager.get_all()
    assert config.name == "N"
    store.writer.flush_now()  # the load marked the upgraded row
    row = store.get("printers", "u1")
    assert row["schema"] == 2
    assert row["data"]["name"] == "N" and "old_name" not in row["data"]
    conn = store._rconn
    archived = conn.execute("SELECT schema FROM row_archive WHERE tbl='printers'").fetchall()
    assert archived == [(1,)]


def test_a_row_from_a_newer_build_is_never_rewritten(store, tmp_path):
    _put_printer(store, "u1", {"id": 1, "token": "t", "unique_id": "u1"}, schema=3)
    before = store.get("printers", "u1")
    manager = _manager(store, tmp_path, config_t=_V2)
    assert manager.get_all() == []
    assert "newer" in manager.unreadable[0].error
    manager.flush()
    store.writer.flush_now()
    assert store.get("printers", "u1") == before


def test_namespaces_do_not_touch_each_other(store, tmp_path):
    a = _manager(store, tmp_path, name="a")
    b = _manager(store, tmp_path, name="b")
    a.persist(PrinterConfig(id=1, token="t", unique_id="ua"))
    b.persist(PrinterConfig(id=2, token="t", unique_id="ub"))
    a.flush()
    b.flush()
    a.clear()
    a.flush()  # a holds nothing now: its row goes, b's stays
    assert [r["unique_id"] for r in store.rows("printers")] == ["ub"]


def test_removing_a_printer_commits_at_once(store, tmp_path):
    manager = _manager(store, tmp_path)
    config = PrinterConfig(id=1, token="t", unique_id="u1")
    manager.persist(config)
    manager.flush(config)
    manager.remove(config)
    manager.flush(config)
    assert store.get("printers", "u1") is None
    assert not store.writer.dirty


@pytest.mark.asyncio
async def test_on_the_event_loop_registration_never_waits_for_the_write_lock(
    tmp_path,
):
    # The daily VACUUM INTO or the weekly integrity_check holds the write
    # lock for seconds; adding, removing or loading on the loop must not wait
    # for it (nor for an fsync). The writer thread commits right after.
    import asyncio
    import threading
    import time

    s = Store.open(tmp_path / "state.db")
    try:
        manager = _manager(s, tmp_path)
        config = PrinterConfig(id=1, token="t", unique_id="u1")
        release = threading.Event()
        held = threading.Event()

        def maintenance():
            with s._wlock:
                held.set()
                release.wait(5)

        threading.Thread(target=maintenance, daemon=True).start()
        assert held.wait(5)
        started = time.monotonic()
        manager.persist(config)
        manager.flush(config)  # app.add
        manager.load()  # an account kind following the store
        manager.remove(config)
        manager.flush(config)  # app.remove
        assert time.monotonic() - started < 1.0
        release.set()
        # Registration is an urgent mark: the writer thread commits it.
        manager.persist(config)
        manager.flush(config)
        for _ in range(500):
            if s.get("printers", "u1") is not None:
                break
            await asyncio.sleep(0.01)
        assert s.get("printers", "u1")["data"]["id"] == 1
    finally:
        s.close()


def test_after_its_store_closed_a_manager_never_opens_another(tmp_path, caplog):
    # Persistence.close() unbinds and closes state.db. A late flush (a
    # flusher, a plugin teardown) must fail loudly, not open a second
    # database in the config directory that nothing ever reads.
    s = Store.open(tmp_path / "data" / "state.db", start=False)
    StoreConfigManager.bind(s)
    try:
        manager = StoreConfigManager(
            name="acme", config_t=PrinterConfig, base_directory=str(tmp_path / "config")
        )
        config = PrinterConfig(id=1, token="t", unique_id="u1")
        manager.persist(config)
        manager.flush(config)
        StoreConfigManager.bind(None)
        s.close()

        config.name = "late"
        manager.flush()  # a field change: only a mark, dropped loudly
        with pytest.raises(RuntimeError):
            manager.flush(config)  # registration: must be on disk, cannot be
        assert not (tmp_path / "config" / "state.db").exists()
        assert any("closed" in r.getMessage() for r in caplog.records)
    finally:
        StoreConfigManager.bind(None)


class _Record(Config):
    """An account record, as the client's HostedAccountStore keeps them."""

    store_table = "accounts"

    def __init__(self, data):
        self.data = dict(data)

    @property
    def account_id(self) -> str:
        return str(self.data["id"])

    @property
    def pk(self):
        return self.data["id"]

    @property
    def sk(self) -> str:
        return self.data.get("region") or ""

    def is_empty(self) -> bool:
        return False

    def as_dict(self) -> dict:
        return dict(self.data)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or "id" not in data:
            raise ValueError("an account record needs its 'id' field")
        return cls(data)


def test_accounts_go_to_their_own_table_as_the_record_is(store, tmp_path):
    manager = _manager(store, tmp_path, config_t=_Record, name="CloudAccounts")
    record = _Record({"id": 7, "region": "eu", "token": "secret", "extra": 1})
    manager.persist(record)
    manager.flush(record)
    row = store.get("accounts", ("CloudAccounts", "7"))
    assert row["secondary"] == "eu" and row["data"]["extra"] == 1
    assert store.rows("printers") == []

    record.data = {"id": 7, "region": "eu", "token": "rotated"}
    manager.flush(record)
    # The provider dropped "extra": an account record is not overlaid.
    assert store.get("accounts", ("CloudAccounts", "7"))["data"] == record.data

    (reloaded,) = _manager(store, tmp_path, config_t=_Record, name="CloudAccounts").get_all()
    assert reloaded.data == record.data

    manager.remove(record)
    manager.flush()
    assert store.rows("accounts") == []


def test_an_unreadable_account_is_kept(store, tmp_path):
    with store.transaction() as txn:
        txn.put("accounts", ("CloudAccounts", "x"), {"secondary": None, "data": {"no": "id"}})
    manager = _manager(store, tmp_path, config_t=_Record, name="CloudAccounts")
    assert manager.get_all() == []
    assert manager.unreadable[0].label == "key:CloudAccounts/x"
    manager.flush()
    store.writer.flush_now()
    assert store.get("accounts", ("CloudAccounts", "x")) is not None


def test_delete_storage_removes_only_this_namespace(store, tmp_path):
    a = _manager(store, tmp_path, name="a")
    a.persist(PrinterConfig(id=1, token="t", unique_id="ua"))
    a.flush()
    _put_printer(store, "ub", {"id": 2, "token": "t"}, name="b")
    a.delete_storage()
    assert [r["unique_id"] for r in store.rows("printers")] == ["ub"]
