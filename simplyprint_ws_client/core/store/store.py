"""One SQLite database in WAL mode: printers, accounts, settings, plugin data.

See docs/persistence-design.md (client repo) for the why. In short: one fsync
per batch instead of one per file, one transaction across kinds of data, a
consistent snapshot for export, and concurrent readers.

Threads and connections, the whole discipline:

* **One writer connection**, used only under ``_wlock``. The store's own
  thread uses it for coalesced batches, the external-change poll and
  maintenance. :meth:`Store.transaction` and :meth:`Writer.flush_now` use it
  from the calling thread, for writes that must be on disk before the caller
  returns (adding a printer, an import). Those block for an fsync, so the
  event loop never calls them; it only calls :meth:`Writer.mark`, which takes
  a short in-memory lock.
* **One reader connection**, used only under ``_rlock``. In WAL mode a reader
  never waits for the writer, so reads stay short even mid-batch.

Every committed change bumps the global ``meta.rev`` once; the changed rows
carry that revision. Subscribers get a :class:`ChangeSet` per table after each
commit (``origin="local"``) and after the poller finds another process's
writes (``origin="external"``): ``PRAGMA data_version`` on the writer
connection changes only for other connections' commits, and rows whose
revision this store already knows are skipped.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import logging
import os
import sqlite3
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from typing import Literal

from simplyprint_ws_client.core.store.recovery import (
    RecoveryReport,
    move_aside,
    recover,
    snapshots_in,
)
from simplyprint_ws_client.core.store.schema import (
    SCHEMA,
    TABLES,
    SchemaStep,
    Table,
    apply_steps,
    archive_row,
    decode_values,
    dump_json,
    encode_values,
)
from simplyprint_ws_client.core.store.writer import (
    CAP,
    DEBOUNCE,
    SKIP,
    Pending,
    Writer,
)

__all__ = [
    "Store",
    "Txn",
    "ChangeSet",
    "Durability",
    "Key",
    "SECRET_NAMESPACES",
    "EXPORT_FORMAT",
]

_logger = logging.getLogger("store")

Durability = Literal["durable", "cache"]
#: A row key: the key column's value, or a tuple of them for compound keys.
Key = Union[str, Tuple[Any, ...]]

#: Settings namespaces that never leave the machine (not in exports, kept on
#: a replacing import): the web app's signing secret and the like.
SECRET_NAMESPACES = frozenset({"host_secrets"})
#: The shape of :meth:`Store.export_json`.
EXPORT_FORMAT = 1
#: Snapshots kept by the daily ``VACUUM INTO``.
SNAPSHOTS_KEPT = 5
SNAPSHOT_EVERY = 24 * 3600.0
INTEGRITY_EVERY = 7 * 24 * 3600.0
POLL_EVERY = 2.0


@dataclass(frozen=True)
class ChangeSet:
    """The rows of one table that one commit (or one poll) changed. Keys are
    plain values for one-column keys (a printer's ``unique_id``), tuples
    otherwise."""

    table: str
    added: Tuple[Key, ...]
    changed: Tuple[Key, ...]
    removed: Tuple[Key, ...]
    #: ``"local"``: this store committed it. ``"external"``: another process
    #: (the CLI with the service down, a restore tool) did.
    origin: str
    rev: int

    def __bool__(self) -> bool:
        return bool(self.added or self.changed or self.removed)


def _digest(table: Table, encoded: Sequence[Any]) -> bytes:
    # The unchanged-write check: a 16-byte hash of the stored column values.
    del table
    return hashlib.blake2b(dump_json(list(encoded)).encode(), digest_size=16).digest()


class Txn:
    """One write transaction (``BEGIN IMMEDIATE``). Use it through
    :meth:`Store.transaction`; it commits when the block exits."""

    def __init__(self, store: "Store", conn: sqlite3.Connection) -> None:
        self._store = store
        self._conn = conn
        # Read inside the transaction: another process may have moved it on.
        self._base_rev = int(
            conn.execute("SELECT value FROM meta WHERE key = 'rev'").fetchone()[0]
        )
        self.rev = self._base_rev + 1
        self._now = store._wall()
        self._hashes: Dict[Tuple[str, Tuple[Any, ...]], Optional[bytes]] = {}
        #: table -> key -> "added" | "changed" | "removed"
        self._changes: Dict[str, Dict[Tuple[Any, ...], str]] = {}

    @property
    def changed(self) -> int:
        """Rows this transaction has changed so far."""
        return sum(len(rows) for rows in self._changes.values())

    def get(self, table: str, key: Key) -> Optional[Dict[str, Any]]:
        spec = TABLES[table]
        normal = self._store.key_of(table, key)
        raw = self._select(spec, normal)
        if raw is None:
            return None
        return _row_dict(spec, normal, raw[:-1], raw[-1])

    def put(self, table: str, key: Key, row: Mapping[str, Any]) -> bool:
        """Insert or replace one row; ``row`` holds the value columns (missing
        ones take the table's defaults). Returns whether anything changed:
        a row equal to what is stored is not written at all."""
        spec = TABLES[table]
        normal = self._store.key_of(table, key)
        encoded = encode_values(spec, row)
        values = tuple(encoded[c] for c in spec.columns)
        digest = _digest(spec, values)
        rid = (table, normal)
        old_raw = None
        known = self._known_hash(rid)
        if known is None or spec.archive:
            old_raw = self._select(spec, normal)
            known = None if old_raw is None else _digest(spec, old_raw[:-1])
        if known == digest:
            self._hashes[rid] = digest
            return False
        if old_raw is not None and spec.archive:
            archive_row(
                self._conn, spec, normal, decode_values(spec, old_raw[:-1]), self._now
            )
        columns = spec.key + spec.columns + ("rev",)
        self._conn.execute(
            "INSERT OR REPLACE INTO %s (%s) VALUES (%s)"
            % (spec.name, ", ".join(columns), ", ".join("?" for _ in columns)),
            normal + values + (self.rev,),
        )
        self._hashes[rid] = digest
        existed = old_raw is not None or self._store._known_rev(table, normal) is not None
        self._note(table, normal, "changed" if existed else "added")
        return True

    def delete(self, table: str, key: Key) -> bool:
        """Delete one row; returns whether it existed. Deleting a printer
        also deletes its ``printer:<unique_id>`` plugin data, in this same
        transaction (docs/persistence-design.md, "Plugin storage")."""
        spec = TABLES[table]
        normal = self._store.key_of(table, key)
        old_raw = self._select(spec, normal)
        if old_raw is None:
            return False
        if spec.archive:
            archive_row(
                self._conn, spec, normal, decode_values(spec, old_raw[:-1]), self._now
            )
        self._conn.execute(
            "DELETE FROM %s WHERE %s" % (spec.name, _where(spec.key)), normal
        )
        self._hashes[(table, normal)] = None
        self._note(table, normal, "removed")
        if table == "printers":
            scope = "printer:%s" % normal[0]
            for kv_key in self._conn.execute(
                "SELECT domain, scope, key FROM kv WHERE scope = ?", (scope,)
            ).fetchall():
                self.delete("kv", tuple(kv_key))
        return True

    def put_legacy_file(
        self, name: str, sha256: str, data: bytes, *, classified: bool
    ) -> bool:
        """Record one imported legacy file verbatim (``legacy_files``)."""
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO legacy_files"
            " (name, sha256, bytes, imported_at, classified) VALUES (?, ?, ?, ?, ?)",
            (name, sha256, sqlite3.Binary(data), self._now, int(classified)),
        )
        return cursor.rowcount > 0

    def mark_legacy_classified(self, name: str, sha256: str) -> None:
        self._conn.execute(
            "UPDATE legacy_files SET classified = 1 WHERE name = ? AND sha256 = ?",
            (name, sha256),
        )

    # -- internals -----------------------------------------------------------

    def _known_hash(self, rid: Tuple[str, Tuple[Any, ...]]) -> Optional[bytes]:
        if rid in self._hashes:
            return self._hashes[rid]
        return self._store._hashes.get(rid)

    def _select(self, spec: Table, key: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        return self._conn.execute(
            "SELECT %s, rev FROM %s WHERE %s"
            % (", ".join(spec.columns), spec.name, _where(spec.key)),
            key,
        ).fetchone()

    def _note(self, table: str, key: Tuple[Any, ...], what: str) -> None:
        rows = self._changes.setdefault(table, {})
        before = rows.get(key)
        if before == "added" and what == "removed":
            del rows[key]  # added and removed in one transaction: no change
        elif before == "added" and what == "changed":
            pass
        elif before == "removed" and what == "added":
            rows[key] = "changed"
        else:
            rows[key] = what


class Store:
    """One database file. Open with :meth:`open` (or :meth:`shared`)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("use Store.open(path, ...)")

    # -- opening -------------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        *,
        durability: Durability = "durable",
        migrations: Sequence[SchemaStep] = SCHEMA,
        app_version: Optional[str] = None,
        start: bool = True,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        debounce: float = DEBOUNCE,
        cap: float = CAP,
        poll_every: float = POLL_EVERY,
        backups: Optional[Union[str, Path]] = None,
    ) -> "Store":
        """Open (creating, migrating, or recovering) the database at ``path``.

        ``durability="durable"`` (``state.db``) commits with
        ``synchronous=FULL`` and keeps daily snapshots in ``backups``
        (default ``<dir>/backups``); ``"cache"`` (``cache.db``) uses
        ``synchronous=OFF``: it may lose its last writes in a power cut, and
        everything in it can be rebuilt. ``start=False`` runs no thread: the
        owner (a test) drives :meth:`Writer.flush_now`, :meth:`poll_external`
        and :meth:`maintain` itself.

        A damaged file is recovered (see :mod:`.recovery`) and the report
        kept in :attr:`recovery`. A file from a newer build raises
        :class:`~.schema.StoreTooNew`.
        """
        if durability not in ("durable", "cache"):
            raise ValueError("durability must be 'durable' or 'cache'")
        self = object.__new__(cls)
        self.path = Path(path)
        self.durability = durability
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backups = (
            Path(backups)
            if backups is not None
            else (self.path.parent / "backups" if durability == "durable" else None)
        )
        self._clock = clock
        self._wall = wall
        self._poll_every = poll_every
        self.recovery: Optional[RecoveryReport] = None
        #: A problem found after opening (a failed weekly integrity check),
        #: for the host to show; ``None`` while healthy.
        self.fault: Optional[str] = None

        conn = None
        try:
            conn = self._connect()
            self._check(conn)
        except sqlite3.DatabaseError as error:
            if conn is not None:
                conn.close()
            # Only damage is recovered: SQLITE_CORRUPT and SQLITE_NOTADB come
            # as the base DatabaseError. A locked, unopenable or full database
            # (OperationalError and the other subclasses) is healthy, and
            # moving it aside would lose it.
            if type(error) is not sqlite3.DatabaseError:
                raise
            damaged = move_aside(self.path, wall)
            conn = self._connect()
            apply_steps(conn, migrations, app_version=app_version)
            self.recovery = recover(
                self.path,
                conn,
                damaged,
                reason="%s: %s" % (type(error).__name__, error),
                snapshots=snapshots_in(self.backups, self.path.stem)
                if durability == "durable"
                else (),
                steps=migrations,
            )
        try:
            apply_steps(conn, migrations, app_version=app_version)
        except BaseException:
            conn.close()
            raise
        self._wconn = conn
        self._rconn = self._connect(reader=True)
        self._wlock = threading.Lock()
        self._rlock = threading.Lock()
        #: How many :meth:`writes_held` blocks are open; while any is, no
        #: batch is written. ``_resumed`` wakes the batches that wait for it.
        self._held = 0
        self._resumed = threading.Condition(self._wlock)
        self._closed = False
        #: Last-written hashes, for the unchanged-write check.
        self._hashes: Dict[Tuple[str, Tuple[Any, ...]], Optional[bytes]] = {}
        #: table -> key -> rev: what this store knows is in the database.
        self._known: Dict[str, Dict[Tuple[Any, ...], int]] = {}
        self._subscribers: Dict[str, List[Callable[[ChangeSet], None]]] = {}
        self._sub_lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self.writer = Writer(
            debounce=debounce, cap=cap, clock=clock, wake=self._wake.set
        )
        self.writer._flush = self._flush_pending
        self.writer._key_of = self.key_of
        with self._wlock:
            self.revision = self._read_rev(conn)
            self._data_version = conn.execute("PRAGMA data_version").fetchone()[0]
            #: ``data_version`` the hash cache was last checked against
            #: (apart from the poll's, which publishes what changed).
            self._hash_version = self._data_version
            for table in TABLES.values():
                self._known[table.name] = self._scan_revs(conn, table)
        self._next_poll = clock() + poll_every
        self._next_maintenance = clock()
        self._thread: Optional[threading.Thread] = None
        _OPEN.add(self)
        if start:
            self._thread = threading.Thread(
                target=self._run, name="sp-store-%s" % self.path.stem, daemon=True
            )
            self._thread.start()
        return self

    @classmethod
    def shared(
        cls, path: Union[str, Path], *, durability: Durability = "durable"
    ) -> "Store":
        """The one open store for ``path`` in this process, opened on first
        use. Several config managers (one per brand, one per account kind)
        share a file; they must share its writer too, or their writes would
        contend for the database lock."""
        resolved = Path(path).resolve()
        with _SHARED_LOCK:
            store = _SHARED.get(resolved)
            if store is None or store._closed:
                store = cls.open(resolved, durability=durability)
                _SHARED[resolved] = store
            elif store.durability != durability:
                raise ValueError("%s is already open as %s" % (resolved, store.durability))
            return store

    def _connect(self, *, reader: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False, timeout=5.0
        )
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            if not reader:
                mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).lower() != "wal":
                    # WAL needs shared memory on a local filesystem; on a
                    # network mount SQLite keeps a rollback journal. Slower,
                    # still correct.
                    _logger.warning(
                        "%s: WAL unavailable, using journal_mode=%s", self.path, mode
                    )
                conn.execute(
                    "PRAGMA synchronous = %s"
                    % ("FULL" if self.durability == "durable" else "OFF")
                )
                conn.execute("PRAGMA journal_size_limit = 4194304")
                conn.execute("PRAGMA wal_autocheckpoint = 1000")
            else:
                conn.execute("PRAGMA query_only = 1")
            conn.execute("PRAGMA cache_size = -512")
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    def _check(conn: sqlite3.Connection) -> None:
        result = [row[0] for row in conn.execute("PRAGMA quick_check").fetchall()]
        if result != ["ok"]:
            raise sqlite3.DatabaseError("quick_check: %s" % "; ".join(map(str, result)))

    # -- keys ----------------------------------------------------------------

    @staticmethod
    def key_of(table: str, key: Key) -> Tuple[Any, ...]:
        """``key`` as the tuple of ``table``'s key columns."""
        spec = TABLES[table]
        normal = key if isinstance(key, tuple) else (key,)
        if len(normal) != len(spec.key):
            raise ValueError(
                "%s keys are %s, got %r" % (table, ", ".join(spec.key), key)
            )
        return normal

    @staticmethod
    def _public_key(table: str, key: Tuple[Any, ...]) -> Key:
        return key[0] if len(TABLES[table].key) == 1 else key

    # -- writing -------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[Txn]:
        """A write transaction, committed when the block exits (rolled back
        if it raises). Blocks for the commit's fsync: not for the event loop.
        Subscribers hear about it after the commit."""
        with self._wlock:
            txn = self._begin_locked()
            try:
                yield txn
                published = self._commit_locked(txn)
            except BaseException:
                self._rollback_locked()
                raise
        self._publish(published)

    def _begin_locked(self) -> Txn:
        if self._closed:
            raise RuntimeError("store %s is closed" % self.path)
        self._wconn.execute("BEGIN IMMEDIATE")
        try:
            # Another process may have written since the last poll (every
            # 2 s): the cached hashes would then call a real change
            # "unchanged" and skip it. ``data_version`` changes only for
            # other connections' commits, and we hold the write lock now, so
            # one check here keeps the cache true for this transaction.
            version = self._wconn.execute("PRAGMA data_version").fetchone()[0]
            if version != self._hash_version:
                self._hashes.clear()
                self._hash_version = version
            return Txn(self, self._wconn)
        except BaseException:
            self._rollback_locked()
            raise

    def _rollback_locked(self) -> None:
        # After a failed COMMIT SQLite may or may not have ended the
        # transaction; only roll back one that is still open.
        if self._wconn.in_transaction:
            self._wconn.execute("ROLLBACK")

    def _commit_locked(self, txn: Txn) -> List[ChangeSet]:
        if not txn._changes:
            self._wconn.execute("COMMIT")
            self._hashes.update(txn._hashes)
            return []
        self._wconn.execute(
            "UPDATE meta SET value = ? WHERE key = 'rev'", (str(txn.rev),)
        )
        self._wconn.execute("COMMIT")
        # Committed: only now does the store's memory follow.
        self._hashes.update(txn._hashes)
        self.revision = txn.rev
        published = []
        for table, rows in txn._changes.items():
            known = self._known.setdefault(table, {})
            for key, what in rows.items():
                if what == "removed":
                    known.pop(key, None)
                else:
                    known[key] = txn.rev
            published.append(self._changeset(table, rows, "local", txn.rev))
        return published

    @contextlib.contextmanager
    def writes_held(self) -> Iterator[None]:
        """Hold the coalescing writer for the block: what is pending is
        written first, then marks are only queued, and a
        :meth:`Writer.flush_now` from another thread waits for the block to
        end. :meth:`transaction` still works inside it.

        Why: a restore replaces rows that running code holds in memory. A
        mark's ``produce`` reads memory when the batch is written, so a batch
        written after the restore commits but before the running code took
        the restored rows would put the old values back. The restore holds
        the writer from before its commit until memory follows; the marks
        queued meanwhile then produce the restored values."""
        published: List[ChangeSet] = []
        with self._wlock:
            if not self._held:
                published = self._write_pending_locked()[1]
            self._held += 1
        self._publish(published)
        try:
            yield
        finally:
            with self._wlock:
                self._held -= 1
                self._resumed.notify_all()
            self._wake.set()

    def _flush_pending(self) -> int:
        """Write every pending mark as one transaction (the writer's batch)."""
        with self._wlock:
            # A held writer (a restore in progress) writes nothing; a caller
            # that needs its rows on disk waits for the hold to end.
            while self._held and not self._stopping:
                self._resumed.wait()
            txn, published = self._write_pending_locked()
        self._publish(published)
        return 0 if txn is None else txn.changed

    def _write_pending_locked(self) -> Tuple[Optional[Txn], List[ChangeSet]]:
        if self._closed:
            raise RuntimeError("store %s is closed" % self.path)
        batch = self.writer.take()
        if not batch:
            return None, []
        try:
            txn = self._begin_locked()
            try:
                failed = self._write_batch(txn, batch)
                published = self._commit_locked(txn)
            except BaseException:
                self._rollback_locked()
                raise
        except BaseException:
            # Not committed: the rows go back to the queue (a newer mark
            # wins), so a failed or interrupted batch loses nothing.
            self.writer.restore(batch)
            raise
        self.writer.retry(failed)
        return txn, published

    @staticmethod
    def _write_batch(
        txn: Txn, batch: Dict[Tuple[str, Tuple[Any, ...]], Pending]
    ) -> Dict[Tuple[str, Tuple[Any, ...]], Pending]:
        """Write the batch's rows; returns the ones whose produce raised."""
        failed: Dict[Tuple[str, Tuple[Any, ...]], Pending] = {}
        for (table, key), pending in batch.items():
            try:
                row = pending.produce()
                if row is SKIP:
                    continue
                if row is not None:
                    encode_values(TABLES[table], row)  # serializes, or raises
            except Exception:
                # One row that cannot be produced keeps its stored version;
                # the rest of the batch is still written. Its mark goes back
                # with a backoff (Writer.retry): the change is not lost.
                _logger.exception(
                    "store: could not produce %s %r; keeping its stored row,"
                    " retrying later",
                    table,
                    key,
                )
                failed[(table, key)] = pending
                continue
            if row is None:
                txn.delete(table, key)
            else:
                txn.put(table, key, row)
        return failed

    # -- reading -------------------------------------------------------------

    def get(self, table: str, key: Key) -> Optional[Dict[str, Any]]:
        """One row (key columns, value columns and ``rev``), or ``None``.
        Reads what is committed: call :meth:`Writer.flush_now` first to see
        this process's pending marks."""
        spec = TABLES[table]
        normal = self.key_of(table, key)
        with self._rlock:
            raw = self._rconn.execute(
                "SELECT %s, rev FROM %s WHERE %s"
                % (", ".join(spec.columns), spec.name, _where(spec.key)),
                normal,
            ).fetchone()
        return None if raw is None else _row_dict(spec, normal, raw[:-1], raw[-1])

    def rows(self, table: str, **where: Any) -> List[Dict[str, Any]]:
        """Every row of ``table`` whose columns equal ``where``, in key order."""
        spec = TABLES[table]
        names = spec.key + spec.columns
        unknown = set(where) - set(names)
        if unknown:
            raise ValueError("%s has no column %s" % (table, ", ".join(sorted(unknown))))
        sql = "SELECT %s, rev FROM %s" % (", ".join(names), spec.name)
        if where:
            sql += " WHERE " + _where(tuple(where))
        sql += " ORDER BY " + ", ".join(spec.key)
        with self._rlock:
            raw_rows = self._rconn.execute(sql, tuple(where.values())).fetchall()
        n = len(spec.key)
        return [
            _row_dict(spec, tuple(raw[:n]), raw[n:-1], raw[-1]) for raw in raw_rows
        ]

    def legacy_files(self) -> List[Dict[str, Any]]:
        """Every recorded legacy file, without its bytes."""
        with self._rlock:
            rows = self._rconn.execute(
                "SELECT name, sha256, imported_at, classified FROM legacy_files"
                " ORDER BY name, imported_at"
            ).fetchall()
        return [
            {"name": n, "sha256": s, "imported_at": at, "classified": bool(c)}
            for n, s, at, c in rows
        ]

    def legacy_bytes(self, name: str, sha256: str) -> Optional[bytes]:
        with self._rlock:
            row = self._rconn.execute(
                "SELECT bytes FROM legacy_files WHERE name = ? AND sha256 = ?",
                (name, sha256),
            ).fetchone()
        return None if row is None else bytes(row[0])

    # -- the change feed -----------------------------------------------------

    def subscribe(
        self, table: str, fn: Callable[[ChangeSet], None]
    ) -> Callable[[], None]:
        """Call ``fn`` with each non-empty :class:`ChangeSet` of ``table``.
        Returns the unsubscribe function.

        ``fn`` runs on whichever thread committed (the store's thread, or a
        caller of :meth:`transaction`), so it must be quick and thread-safe:
        a host hands the change to its event loop
        (``loop.call_soon_threadsafe``). An exception is logged and the other
        subscribers still run."""
        TABLES[table]  # unknown tables fail here, not at the first commit
        with self._sub_lock:
            self._subscribers.setdefault(table, []).append(fn)

        def unsubscribe() -> None:
            with self._sub_lock:
                fns = self._subscribers.get(table, [])
                if fn in fns:
                    fns.remove(fn)

        return unsubscribe

    def _publish(self, changes: Sequence[ChangeSet]) -> None:
        for change in changes:
            if not change:
                continue
            with self._sub_lock:
                fns = list(self._subscribers.get(change.table, ()))
            for fn in fns:
                try:
                    fn(change)
                except Exception:
                    _logger.exception("store: a %s subscriber failed", change.table)

    def poll_external(self) -> List[ChangeSet]:
        """Find rows another process wrote since the last poll and publish
        them (``origin="external"``). Cheap when nothing changed: one
        ``PRAGMA data_version``. The store's thread calls it every 2 s."""
        with self._wlock:
            if self._closed:
                return []
            version = self._wconn.execute("PRAGMA data_version").fetchone()[0]
            if version == self._data_version:
                return []
            self._data_version = version
            rev = self._read_rev(self._wconn)
            published = []
            for table in TABLES.values():
                now = self._scan_revs(self._wconn, table)
                before = self._known.get(table.name, {})
                rows: Dict[Tuple[Any, ...], str] = {}
                for key, row_rev in now.items():
                    if key not in before:
                        rows[key] = "added"
                    elif row_rev != before[key]:
                        rows[key] = "changed"
                for key in before:
                    if key not in now:
                        rows[key] = "removed"
                for key in rows:
                    # Whatever we hashed for it is no longer what is stored.
                    self._hashes.pop((table.name, key), None)
                self._known[table.name] = now
                if rows:
                    published.append(self._changeset(table.name, rows, "external", rev))
            self.revision = max(self.revision, rev)
        self._publish(published)
        return published

    def _changeset(
        self, table: str, rows: Mapping[Tuple[Any, ...], str], origin: str, rev: int
    ) -> ChangeSet:
        def keys(what: str) -> Tuple[Key, ...]:
            return tuple(
                self._public_key(table, k) for k, w in sorted(rows.items()) if w == what
            )

        return ChangeSet(
            table=table,
            added=keys("added"),
            changed=keys("changed"),
            removed=keys("removed"),
            origin=origin,
            rev=rev,
        )

    def _known_rev(self, table: str, key: Tuple[Any, ...]) -> Optional[int]:
        return self._known.get(table, {}).get(key)

    @staticmethod
    def _scan_revs(
        conn: sqlite3.Connection, table: Table
    ) -> Dict[Tuple[Any, ...], int]:
        n = len(table.key)
        return {
            tuple(row[:n]): row[n]
            for row in conn.execute(
                "SELECT %s, rev FROM %s" % (", ".join(table.key), table.name)
            )
        }

    @staticmethod
    def _read_rev(conn: sqlite3.Connection) -> int:
        return int(conn.execute("SELECT value FROM meta WHERE key = 'rev'").fetchone()[0])

    # -- snapshots, export, import ---------------------------------------------

    def snapshot_to(self, path: Union[str, Path]) -> Path:
        """A consistent copy of the whole database at ``path`` (``VACUUM
        INTO``), written to a temporary name first so ``path`` is either the
        old file or a complete new one."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(".%s.%d.tmp" % (target.name, os.getpid()))
        if temporary.exists():
            temporary.unlink()
        with self._wlock:
            try:
                self._wconn.execute("VACUUM INTO ?", (str(temporary),))
            except sqlite3.OperationalError:
                # SQLite older than 3.27 has no VACUUM INTO; the backup API
                # gives the same consistent copy.
                destination = sqlite3.connect(str(temporary))
                try:
                    self._wconn.backup(destination)
                finally:
                    destination.close()
        os.replace(temporary, target)
        return target

    def maintain(self, now: Optional[float] = None) -> None:
        """The daily snapshot (keeping :data:`SNAPSHOTS_KEPT`) and the weekly
        ``integrity_check``. Durable stores only; the store's thread calls it."""
        if self.durability != "durable" or self.backups is None:
            return
        wall = self._wall() if now is None else now
        day = datetime.fromtimestamp(wall, tz=timezone.utc).strftime("%Y-%m-%d")
        target = self.backups / ("%s-%s.db" % (self.path.stem, day))
        if not target.exists():
            self.snapshot_to(target)
            for old in snapshots_in(self.backups, self.path.stem)[SNAPSHOTS_KEPT:]:
                old.unlink()
        checked = self._meta("integrity_checked_at")
        if checked is None or wall - float(checked) >= INTEGRITY_EVERY:
            with self._wlock:
                result = [
                    row[0]
                    for row in self._wconn.execute("PRAGMA integrity_check").fetchall()
                ]
                self._wconn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    ("integrity_checked_at", str(wall)),
                )
            if result != ["ok"]:
                # Recovery runs on the next open; until then the host shows it.
                self.fault = "integrity_check: %s" % "; ".join(map(str, result))
                _logger.error("%s: %s", self.path, self.fault)

    def _meta(self, key: str) -> Optional[str]:
        with self._rlock:
            row = self._rconn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else row[0]

    def export_json(self) -> Dict[str, Any]:
        """Every row as plain JSON-ready data, per table, without secrets
        (:data:`SECRET_NAMESPACES`). The bundle's per-namespace export."""
        tables: Dict[str, List[Dict[str, Any]]] = {}
        for name in TABLES:
            rows = []
            for row in self.rows(name):
                if name == "settings" and row["namespace"] in SECRET_NAMESPACES:
                    continue
                row.pop("rev", None)
                rows.append(row)
            tables[name] = rows
        return {"format": EXPORT_FORMAT, "rev": self.revision, "tables": tables}

    def import_json(self, data: Mapping[str, Any], *, replace: bool) -> int:
        """Write an :meth:`export_json` value in one transaction; returns the
        rows changed. ``replace`` also deletes rows of the exported tables
        that the export does not hold (a restore); secrets stay either way.
        Tables the export does not name are left alone."""
        if data.get("format") != EXPORT_FORMAT:
            raise ValueError("unknown export format %r" % data.get("format"))
        tables = data.get("tables")
        if not isinstance(tables, Mapping):
            raise ValueError("export has no tables")
        with self.transaction() as txn:
            for name, rows in tables.items():
                if name not in TABLES:
                    continue
                spec = TABLES[name]
                wanted = set()
                for row in rows:
                    key = tuple(row[c] for c in spec.key)
                    wanted.add(key)
                    txn.put(name, key, {c: row.get(c) for c in spec.columns})
                if replace:
                    for row in self._scan_revs(self._wconn, spec):
                        if row in wanted:
                            continue
                        if name == "settings" and row[0] in SECRET_NAMESPACES:
                            continue
                        txn.delete(name, row)
        return txn.changed

    # -- the store's thread ----------------------------------------------------

    def _run(self) -> None:
        while not self._stopping:
            now = self._clock()
            deadlines = [self._next_poll, self._next_maintenance]
            # A held writer has no deadline: the hold's end wakes the thread.
            pending = None if self._held else self.writer.next_deadline()
            if pending is not None:
                deadlines.append(pending)
            self._wake.wait(max(0.0, min(deadlines) - now))
            self._wake.clear()
            if self._stopping:
                break
            # Each duty on its own: a failing poll must not stop writes.
            try:
                # Never wait for a hold here: the poll and maintenance go on.
                if not self._held and self.writer.is_due():
                    self._flush_pending()
            except Exception:
                _logger.exception("store %s: write failed; will retry", self.path)
                # Do not spin on a failure that repeats (disk full): wait
                # out one debounce before the next attempt.
                self._wake.wait(self.writer.debounce)
            now = self._clock()
            if now >= self._next_poll:
                self._next_poll = now + self._poll_every
                try:
                    self.poll_external()
                except Exception:
                    _logger.exception("store %s: change poll failed", self.path)
            if now >= self._next_maintenance:
                self._next_maintenance = now + 3600.0
                try:
                    self.maintain()
                except Exception:
                    _logger.exception("store %s: maintenance failed", self.path)

    def close(self) -> None:
        """Write what is pending, checkpoint the WAL, close. Idempotent."""
        if self._closed:
            return
        self._stopping = True
        self._wake.set()
        with self._wlock:
            # Batches waiting for a hold go through: the process is stopping.
            self._resumed.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=30)
        try:
            self._flush_pending()
        except Exception:
            _logger.exception("store %s: final write failed", self.path)
        with self._wlock:
            try:
                self._wconn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                _logger.warning("store %s: checkpoint failed", self.path, exc_info=True)
            self._closed = True
            self.writer.closed = True
            self._wconn.close()
        with self._rlock:
            self._rconn.close()
        _OPEN.discard(self)

    @property
    def closed(self) -> bool:
        return self._closed


def _where(columns: Sequence[str]) -> str:
    return " AND ".join("%s = ?" % c for c in columns)


def _row_dict(
    spec: Table, key: Tuple[Any, ...], raw_values: Sequence[Any], rev: int
) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(zip(spec.key, key))
    row.update(decode_values(spec, raw_values))
    row["rev"] = rev
    return row


#: Stores opened by :meth:`Store.shared`, by resolved path.
_SHARED: Dict[Path, Store] = {}
_SHARED_LOCK = threading.Lock()
#: Every open store, so a normal interpreter exit still writes pending marks.
_OPEN: "weakref.WeakSet[Store]" = weakref.WeakSet()


@atexit.register
def _close_open_stores() -> None:
    # The store's thread is a daemon (it must never keep a stopping process
    # alive), so exit would drop up to CAP seconds of marks without this.
    for store in list(_OPEN):
        try:
            store.close()
        except Exception:  # pragma: no cover - exit path, best effort
            _logger.exception("store %s: close at exit failed", store.path)
