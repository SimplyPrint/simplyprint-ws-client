"""Opening a damaged database without losing what can still be read.

Never silent and never destructive (docs/persistence-design.md, 3,
"Corruption"):

1. the damaged database and its ``-wal`` / ``-shm`` files are moved aside to
   ``<name>.corrupt-<ts>`` (nothing is deleted);
2. a fresh database gets every row that can still be read from the damaged
   one, table by table, row by row;
3. the tables that could not be read to the end (all of them, when the file
   does not open) are filled from the newest snapshot (the daily ``VACUUM
   INTO``); a table read completely is not, or rows deleted since the
   snapshot would come back;
4. the :class:`RecoveryReport` says what came from where. The store keeps it
   as ``Store.recovery`` for the host to raise a persistent fault, and logs it.

The stdlib has no ``.recover``, so step 2 is best effort: a table whose pages
are gone yields the rows read before the damage, then the snapshot fills in.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import quote

from simplyprint_ws_client.core.store.schema import (
    TABLES,
    RowTransform,
    SchemaStep,
    Table,
    decode_values,
    encode_values,
)

__all__ = ["RecoveryReport", "recover", "move_aside", "snapshots_in", "RECOVERED"]

_logger = logging.getLogger("store")

#: Tables copied into the fresh database, in this order. ``row_archive`` is
#: history and is left in the moved-aside file.
RECOVERED = ("printers", "accounts", "settings", "kv", "legacy_files")


@dataclass(frozen=True)
class RecoveryReport:
    """What a recovery did; the host shows it as a persistent fault."""

    #: Why the database was judged damaged.
    reason: str
    #: Where the damaged database now is (``-wal`` / ``-shm`` next to it).
    moved_to: Path
    #: Rows copied from the damaged database, per table.
    from_damaged: Dict[str, int] = field(default_factory=dict)
    #: Rows filled from :attr:`snapshot`, per table.
    from_snapshot: Dict[str, int] = field(default_factory=dict)
    #: The snapshot used, if any.
    snapshot: Optional[Path] = None
    #: Tables the damaged database could not (fully) be read from.
    unreadable_tables: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            "store damaged (%s): moved to %s; rows kept %s; from snapshot %s %s;"
            " tables not fully readable: %s"
            % (
                self.reason,
                self.moved_to,
                self.from_damaged,
                self.snapshot,
                self.from_snapshot,
                self.unreadable_tables or "none",
            )
        )


def move_aside(path: Path, wall: Callable[[], float] = time.time) -> Path:
    """Move ``path`` and its ``-wal`` / ``-shm`` files to
    ``<name>.corrupt-<ts>``; returns the new database path."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(wall()))
    target = path.with_name("%s.corrupt-%s" % (path.name, stamp))
    n = 1
    while target.exists():
        n += 1
        target = path.with_name("%s.corrupt-%s-%d" % (path.name, stamp, n))
    for suffix in ("", "-wal", "-shm"):
        source = Path(str(path) + suffix)
        if source.exists():
            os.replace(source, Path(str(target) + suffix))
    return target


def snapshots_in(directory: Optional[Path], stem: str) -> List[Path]:
    """Snapshots of database ``stem`` in ``directory``, newest first. Names
    carry a sortable UTC date (``<stem>-YYYY-MM-DD.db``)."""
    if directory is None or not directory.is_dir():
        return []
    return sorted(directory.glob("%s-*.db" % stem), reverse=True)


def recover(
    path: Path,
    fresh: sqlite3.Connection,
    damaged: Path,
    *,
    reason: str,
    snapshots: Sequence[Path] = (),
    steps: Sequence[SchemaStep] = (),
) -> RecoveryReport:
    """Fill ``fresh`` (already at the current schema, ``steps``) from
    ``damaged``, then from the first snapshot that opens. Rows already
    present are never overwritten: the damaged file's rows are newer than
    any snapshot's."""
    report = RecoveryReport(reason=reason, moved_to=damaged)
    if not _copy_from(
        fresh, damaged, report.from_damaged, report.unreadable_tables, steps=steps
    ):
        report.unreadable_tables[:] = list(RECOVERED)
    # Only the tables the damaged file could not fully give are filled from
    # the snapshot. A table read to the end is complete: topping it up would
    # bring back every row deleted since the snapshot (a removed printer).
    missing = [t for t in RECOVERED if t in report.unreadable_tables]
    for snapshot in snapshots if missing else ():
        counts: Dict[str, int] = {}
        failed: List[str] = []
        if _copy_from(fresh, snapshot, counts, failed, tables=missing, steps=steps):
            report = RecoveryReport(
                reason=report.reason,
                moved_to=report.moved_to,
                from_damaged=report.from_damaged,
                from_snapshot=counts,
                snapshot=snapshot,
                unreadable_tables=report.unreadable_tables,
            )
            break
    _reset_rev(fresh)
    _logger.error("%s: %s", path, report.describe())
    return report


def _copy_from(
    fresh: sqlite3.Connection,
    source: Path,
    counts: Dict[str, int],
    failed: List[str],
    *,
    tables: Sequence[str] = RECOVERED,
    steps: Sequence[SchemaStep] = (),
) -> bool:
    """Copy every readable row of ``tables`` of ``source`` into ``fresh``
    (``INSERT OR IGNORE``). Returns whether ``source`` opened at all.

    ``fresh`` is already at the newest schema, so its migrations' row
    transforms ran over no rows: a row from a ``source`` at an older schema
    gets the transforms of every step past that schema here, in order, as
    :func:`~.schema.apply_steps` would have given it."""
    try:
        conn = sqlite3.connect(
            "file:%s?mode=ro" % quote(source.as_posix()),
            uri=True,
            isolation_level=None,
        )
    except sqlite3.Error:
        return False
    try:
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.Error:
            return False
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        except sqlite3.Error:
            version = 0
        transforms = _transforms_after(version, steps)
        for table in tables:
            columns = _columns(fresh, table)
            counts[table] = 0
            fns = transforms.get(table, [])
            spec = TABLES.get(table)
            try:
                cursor = conn.execute("SELECT * FROM %s" % table)
                names = [d[0] for d in cursor.description]
                # A transform may fill value columns the old file lacks.
                extra = [c for c in spec.columns if c not in names] if fns and spec else []
                shared = [n for n in names + extra if n in columns]
                insert = "INSERT OR IGNORE INTO %s (%s) VALUES (%s)" % (
                    table,
                    ", ".join(shared),
                    ", ".join("?" for _ in shared),
                )
                fresh.execute("BEGIN IMMEDIATE")
                try:
                    # Row by row: a read error part-way keeps the rows before it.
                    while True:
                        try:
                            row = cursor.fetchone()
                        except sqlite3.DatabaseError:
                            failed.append(table)
                            break
                        if row is None:
                            break
                        values = dict(zip(names, row))
                        if fns and spec is not None:
                            try:
                                values.update(_migrated(spec, values, fns))
                            except Exception:
                                # Never silent: the row stays in the
                                # moved-aside file, and the log says which.
                                _logger.exception(
                                    "recovery: %s row %r does not migrate; left in %s",
                                    table,
                                    [values.get(k) for k in spec.key],
                                    source,
                                )
                                continue
                        try:
                            added = fresh.execute(
                                insert, tuple(values.get(n) for n in shared)
                            ).rowcount
                        except sqlite3.IntegrityError:
                            # A damaged row (a NULL where the schema needs a
                            # value): skip it, keep the rest of the table.
                            continue
                        counts[table] += max(added, 0)
                    fresh.execute("COMMIT")
                except BaseException:
                    fresh.execute("ROLLBACK")
                    raise
            except sqlite3.DatabaseError:
                failed.append(table)
        return True
    finally:
        conn.close()


def _transforms_after(
    version: int, steps: Sequence[SchemaStep]
) -> Dict[str, List[RowTransform]]:
    """Per table, the row transforms of the steps past ``version``, in order."""
    found: Dict[str, List[RowTransform]] = {}
    for step in sorted(steps, key=lambda s: s.version):
        if step.version > version:
            for table, transform in step.rows.items():
                found.setdefault(table, []).append(transform)
    return found


def _migrated(
    spec: Table, values: Dict[str, Any], fns: Sequence[RowTransform]
) -> Dict[str, Any]:
    """``values`` (one raw row) with its value columns run through ``fns``,
    encoded back as stored."""
    decoded = decode_values(
        spec, [values.get(c, spec.defaults.get(c)) for c in spec.columns]
    )
    for fn in fns:
        decoded = fn(dict(decoded))
    return encode_values(spec, decoded)


def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)]


def _reset_rev(fresh: sqlite3.Connection) -> None:
    # The global revision must stay ahead of every copied row's, or a later
    # write could reuse a revision a reader has already seen.
    top = 0
    for table in TABLES:
        value = fresh.execute("SELECT max(rev) FROM %s" % table).fetchone()[0]
        top = max(top, value or 0)
    fresh.execute(
        "UPDATE meta SET value = ? WHERE key = 'rev'", (str(top + 1),)
    )
