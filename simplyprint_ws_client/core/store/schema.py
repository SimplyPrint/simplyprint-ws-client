"""The store's tables and its schema migrations.

The schema version is SQLite's own ``PRAGMA user_version``: it lives in the
database header, so it changes in the same transaction as the tables it
describes and can never disagree with them.

A migration is a :class:`SchemaStep`: DDL statements plus optional row
transforms. Both are plain data and pure ``dict -> dict`` functions, so a step
is tested against fixtures without a database; :func:`plan` (also pure) decides
which steps run and in what order. :func:`apply_steps` is the only part that
touches SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "Table",
    "TABLES",
    "SchemaStep",
    "SCHEMA",
    "StoreTooNew",
    "plan",
    "apply_steps",
    "dump_json",
    "ARCHIVE_KEEP",
]

#: Versions of a row kept in ``row_archive`` (the newest first).
ARCHIVE_KEEP = 3

#: A row transform: one row's value columns in, the migrated columns out.
RowTransform = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class Table:
    """A table the store writes rows into, described as data so one generic
    put/delete/read path serves them all."""

    name: str
    #: The primary-key columns, in order.
    key: Tuple[str, ...]
    #: The value columns (everything but the key and ``rev``).
    columns: Tuple[str, ...]
    #: Value columns stored as JSON text and handed out as Python values.
    json_columns: FrozenSet[str]
    #: Keep the previous versions of a changed or deleted row in
    #: ``row_archive``. Only for the rows a user would miss (printers,
    #: accounts); chatty plugin data would just churn the archive.
    archive: bool = False
    #: Defaults for value columns a put leaves out.
    defaults: Mapping[str, Any] = field(default_factory=dict)


TABLES: Dict[str, Table] = {
    table.name: table
    for table in (
        Table(
            "printers",
            key=("unique_id",),
            columns=("store", "pk", "sk", "data", "schema", "state", "error"),
            json_columns=frozenset({"data"}),
            archive=True,
            defaults={"schema": 1, "state": "ok"},
        ),
        Table(
            "accounts",
            key=("store", "account_id"),
            columns=("secondary", "data"),
            json_columns=frozenset({"data"}),
            archive=True,
        ),
        Table(
            "settings",
            key=("namespace", "key"),
            columns=("value",),
            json_columns=frozenset({"value"}),
        ),
        Table(
            "kv",
            key=("domain", "scope", "key"),
            columns=("value",),
            json_columns=frozenset({"value"}),
        ),
    )
}


@dataclass(frozen=True)
class SchemaStep:
    """One schema version: the DDL that makes it, and how existing rows change.

    ``rows`` maps a table name to a pure transform of each row's value columns
    (JSON columns already decoded). The previous row goes to ``row_archive``
    before it is replaced, whatever the table.
    """

    version: int
    statements: Tuple[str, ...]
    rows: Mapping[str, RowTransform] = field(default_factory=dict)


#: Version 1: every table the design names (docs/persistence-design.md, 3).
_V1 = SchemaStep(
    1,
    (
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "CREATE TABLE migrations ("
        " version INTEGER PRIMARY KEY, applied_at REAL NOT NULL, app_version TEXT)",
        "CREATE TABLE printers ("
        " unique_id TEXT PRIMARY KEY, store TEXT NOT NULL, pk, sk,"
        " data TEXT NOT NULL, schema INTEGER NOT NULL DEFAULT 1,"
        " rev INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'ok', error TEXT)",
        # (pk, sk) is how the old stores addressed a printer. It changes when a
        # printer is keyed, so it is only an index here, never the identity.
        "CREATE INDEX printers_store_key ON printers (store, pk, sk)",
        "CREATE TABLE accounts ("
        " store TEXT NOT NULL, account_id TEXT NOT NULL, secondary TEXT,"
        " data TEXT NOT NULL, rev INTEGER NOT NULL,"
        " PRIMARY KEY (store, account_id))",
        "CREATE TABLE settings ("
        " namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
        " rev INTEGER NOT NULL, PRIMARY KEY (namespace, key))",
        "CREATE TABLE kv ("
        " domain TEXT NOT NULL, scope TEXT NOT NULL, key TEXT NOT NULL,"
        " value TEXT NOT NULL, rev INTEGER NOT NULL,"
        " PRIMARY KEY (domain, scope, key))",
        "CREATE INDEX kv_scope ON kv (scope)",
        "CREATE TABLE legacy_files ("
        " name TEXT NOT NULL, sha256 TEXT NOT NULL, bytes BLOB NOT NULL,"
        " imported_at REAL NOT NULL, classified INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (name, sha256))",
        "CREATE TABLE row_archive ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, tbl TEXT NOT NULL, key TEXT NOT NULL,"
        " schema INTEGER, data TEXT NOT NULL, at REAL NOT NULL)",
        "CREATE INDEX row_archive_key ON row_archive (tbl, key)",
        "INSERT INTO meta (key, value) VALUES ('rev', '0')",
    ),
)

#: The schema this build knows, oldest first.
SCHEMA: Tuple[SchemaStep, ...] = (_V1,)


class StoreTooNew(Exception):
    """The database was written by a newer build (its ``user_version`` is past
    every step this build knows). Writing to it could corrupt what that build
    stored, so the store refuses to open it instead."""


def plan(current: int, steps: Sequence[SchemaStep]) -> List[SchemaStep]:
    """The steps that take a database at ``current`` to the newest version.

    Pure. Steps may be declared in any order; they run by version. Versions
    must be 1, 2, 3, ... without gaps or repeats, because ``user_version`` is
    the only record of what ran: a gap would be skipped forever.
    """
    ordered = sorted(steps, key=lambda step: step.version)
    versions = [step.version for step in ordered]
    if versions != list(range(1, len(ordered) + 1)):
        raise ValueError("schema steps must be numbered 1..n, got %r" % versions)
    if current > len(ordered):
        raise StoreTooNew(
            "store schema %d is newer than this build's %d" % (current, len(ordered))
        )
    return ordered[current:]


def dump_json(value: Any) -> str:
    """The one JSON encoding the store writes: compact and key-sorted, so equal
    values give equal text (the unchanged-write check compares hashes of it)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def apply_steps(
    conn: sqlite3.Connection,
    steps: Sequence[SchemaStep],
    *,
    app_version: Optional[str] = None,
) -> int:
    """Bring ``conn`` (in autocommit mode) to the newest schema; returns the
    version it ended at.

    Each step is its own ``BEGIN IMMEDIATE`` transaction, ``user_version``
    included, so a crash between steps leaves a database at a known version
    that the next open continues from.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for step in plan(current, steps):
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in step.statements:
                conn.execute(statement)
            for table_name, transform in step.rows.items():
                _transform_rows(conn, TABLES[table_name], transform)
            conn.execute(
                "INSERT INTO migrations (version, applied_at, app_version)"
                " VALUES (?, ?, ?)",
                (step.version, time.time(), app_version),
            )
            # PRAGMA takes no parameters; the version is an int we control.
            conn.execute("PRAGMA user_version = %d" % int(step.version))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        current = step.version
    return current


def _transform_rows(
    conn: sqlite3.Connection, table: Table, transform: RowTransform
) -> None:
    columns = table.key + table.columns
    rows = conn.execute(
        "SELECT %s FROM %s" % (", ".join(columns), table.name)
    ).fetchall()
    now = time.time()
    for row in rows:
        key = tuple(row[: len(table.key)])
        values = decode_values(table, row[len(table.key) :])
        migrated = transform(dict(values))
        if migrated == values:
            continue
        archive_row(conn, table, key, values, now)
        encoded = encode_values(table, migrated)
        conn.execute(
            "UPDATE %s SET %s WHERE %s"
            % (
                table.name,
                ", ".join("%s = ?" % c for c in table.columns),
                " AND ".join("%s = ?" % c for c in table.key),
            ),
            tuple(encoded[c] for c in table.columns) + key,
        )


def decode_values(table: Table, raw: Sequence[Any]) -> Dict[str, Any]:
    """Value columns as stored -> Python values (JSON columns decoded).

    A JSON column whose text does not parse is handed out as the text itself:
    the row is kept and its reader decides (a printer becomes unreadable)."""
    values: Dict[str, Any] = {}
    for column, value in zip(table.columns, raw):
        if column in table.json_columns and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        values[column] = value
    return values


def encode_values(table: Table, values: Mapping[str, Any]) -> Dict[str, Any]:
    """Python values -> what is stored, filling the table's defaults."""
    encoded: Dict[str, Any] = {}
    for column in table.columns:
        value = values.get(column, table.defaults.get(column))
        if column in table.json_columns:
            value = dump_json(value)
        encoded[column] = value
    return encoded


def archive_row(
    conn: sqlite3.Connection,
    table: Table,
    key: Tuple[Any, ...],
    values: Mapping[str, Any],
    at: float,
) -> None:
    """Keep ``values`` (the row before a change) in ``row_archive``, trimmed to
    the last :data:`ARCHIVE_KEEP` versions of that row."""
    key_text = dump_json(list(key))
    conn.execute(
        "INSERT INTO row_archive (tbl, key, schema, data, at) VALUES (?, ?, ?, ?, ?)",
        (table.name, key_text, values.get("schema"), dump_json(dict(values)), at),
    )
    conn.execute(
        "DELETE FROM row_archive WHERE tbl = ? AND key = ? AND id NOT IN"
        " (SELECT id FROM row_archive WHERE tbl = ? AND key = ?"
        " ORDER BY id DESC LIMIT ?)",
        (table.name, key_text, table.name, key_text, ARCHIVE_KEEP),
    )
