"""Opening a damaged database without losing what can still be read.

Never silent and never destructive (docs/persistence-design.md, 3,
"Corruption"):

1. the damaged database and its ``-wal`` / ``-shm`` files are moved aside to
   ``<name>.corrupt-<ts>`` (nothing is deleted);
2. a fresh database gets every row that can still be read from the damaged
   one, table by table, row by row;
3. rows still missing are filled from the newest snapshot (the daily
   ``VACUUM INTO``);
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
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import quote

from simplyprint_ws_client.core.store.schema import TABLES

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
) -> RecoveryReport:
    """Fill ``fresh`` (already at the current schema) from ``damaged``, then
    from the first snapshot that opens. Rows already present are never
    overwritten: the damaged file's rows are newer than any snapshot's."""
    report = RecoveryReport(reason=reason, moved_to=damaged)
    _copy_from(fresh, damaged, report.from_damaged, report.unreadable_tables)
    for snapshot in snapshots:
        counts: Dict[str, int] = {}
        failed: List[str] = []
        if _copy_from(fresh, snapshot, counts, failed):
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
) -> bool:
    """Copy every readable row of ``source`` into ``fresh`` (``INSERT OR
    IGNORE``). Returns whether ``source`` opened at all."""
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
        for table in RECOVERED:
            columns = _columns(fresh, table)
            counts[table] = 0
            try:
                cursor = conn.execute("SELECT * FROM %s" % table)
                names = [d[0] for d in cursor.description]
                shared = [n for n in names if n in columns]
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
                        try:
                            added = fresh.execute(
                                insert, tuple(values[n] for n in shared)
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
