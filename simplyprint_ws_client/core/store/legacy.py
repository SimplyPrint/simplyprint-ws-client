"""Importing the files the store replaces (docs/persistence-design.md, 4).

Three parts, kept apart so the conversion is tested without a disk or a
database:

* :meth:`LegacyImporter.scan` reads the config directory (the only file IO):
  every ``*.json``, ``*.db`` and ``*.json.corrupt``, and ``uploads/*.json``.
* :meth:`LegacyImporter.convert` turns one file into rows. **Pure**: bytes
  in, rows out, the same rows every time (a ``unique_id`` it has to mint is
  derived from the entry, not random).
* :meth:`LegacyImporter.run` writes, in one transaction, every file not yet
  recorded (by name and sha256) into ``legacy_files`` verbatim, plus the rows
  of those it could classify. A second run finds every file recorded and
  imports nothing. When an old build (after an OTA rollback) changes a file,
  its sha256 changes and the next run imports it again, laid over the stored
  rows so fields the old build does not know survive.

Classification is by name: a pinned printer storage name, an account kind's
``store_name``, the web settings file, an upload record. Anything else stays
in ``legacy_files`` unclassified until a plugin that names it mounts
(:meth:`LegacyImporter.import_unclassified`), so the printers of a plugin that
failed to load are not lost. ``*.json.corrupt`` files are only kept.

The original files are never changed or deleted here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union
from urllib.parse import quote

from simplyprint_ws_client.core.store.store import Store

__all__ = [
    "LegacyImporter",
    "LegacyFile",
    "LegacyImportError",
    "AccountFormat",
    "Put",
    "Rows",
    "ImportReport",
    "UPLOADS_DOMAIN",
    "SETTINGS_SECRETS",
]

#: The kv domain upload records go to, in ``cache.db``.
UPLOADS_DOMAIN = "host.uploads"
#: Settings that go to the never-exported ``host_secrets`` namespace.
SETTINGS_SECRETS = frozenset({"secret_key"})
#: Names that are the store itself, never a legacy file.
_STORE_FILES = frozenset({"state.db", "cache.db"})
#: The namespace minted ``unique_id`` values derive from (fixed forever: the
#: same entry must get the same id on every import).
_LEGACY_NAMESPACE = uuid.UUID("5b0f6c1e-3d2a-4e8b-9c7f-1a2b3c4d5e6f")


class LegacyImportError(Exception):
    """A file that must be readable is not (the web settings): the import
    fails loudly, as loading them does today, and writes nothing."""


@dataclass(frozen=True)
class AccountFormat:
    """Where an account kind's records keep their id (``AccountKind.key``)
    and, optionally, their secondary key (``secondary_key``)."""

    key: str
    secondary: Optional[str] = None


@dataclass(frozen=True)
class LegacyFile:
    """One file as found: its name relative to the config directory (POSIX
    separators), its bytes and their sha256."""

    name: str
    data: bytes
    sha256: str
    #: An old SQLite config's ``(pk, sk, data)`` rows, read at scan time
    #: (``None`` for other files, or when it could not be read).
    sqlite_rows: Optional[Tuple[Tuple[Any, Any, Any], ...]] = None
    #: Why the file could not be read at scan time.
    error: Optional[str] = None

    @classmethod
    def of(cls, name: str, data: bytes, **kwargs: Any) -> "LegacyFile":
        return cls(name, data, hashlib.sha256(data).hexdigest(), **kwargs)


@dataclass(frozen=True)
class Put:
    """One row to write: ``Txn.put(table, key, row)``."""

    table: str
    key: Union[str, Tuple[Any, ...]]
    row: Mapping[str, Any]


@dataclass(frozen=True)
class Rows:
    """What one file converts to."""

    #: Rows for ``state.db``.
    state: Tuple[Put, ...] = ()
    #: Rows for ``cache.db``.
    cache: Tuple[Put, ...] = ()
    #: Whether the file was recognised (and converted, as far as it could be).
    classified: bool = False
    errors: Tuple[str, ...] = ()


@dataclass
class ImportReport:
    #: Files newly recorded in ``legacy_files``.
    recorded: List[str] = field(default_factory=list)
    #: Of those (or of the unclassified ones), the ones converted to rows.
    converted: List[str] = field(default_factory=list)
    #: Recorded but not recognised: kept until a plugin names them.
    unclassified: List[str] = field(default_factory=list)
    rows: int = 0
    errors: List[str] = field(default_factory=list)


class LegacyImporter:
    """Imports one config directory. ``printer_stores`` are the pinned
    storage names (``<storage name>.json`` / ``.db``); ``account_stores`` maps
    each loaded account kind's ``store_name`` to its record format."""

    def __init__(
        self,
        config_dir: Union[str, Path],
        *,
        printer_stores: Iterable[str] = (),
        account_stores: Optional[Mapping[str, AccountFormat]] = None,
        settings_file: str = "settings.json",
        settings_namespace: str = "web",
        uploads_dir: str = "uploads",
    ) -> None:
        self.config_dir = Path(config_dir)
        self.printer_stores: Set[str] = set(printer_stores)
        self.account_stores: Dict[str, AccountFormat] = dict(account_stores or {})
        self.settings_file = settings_file
        self.settings_namespace = settings_namespace
        self.uploads_dir = uploads_dir

    # -- reading the directory ---------------------------------------------------

    def scan(self) -> List[LegacyFile]:
        """Every legacy file in the config directory, sorted by name."""
        found: List[LegacyFile] = []
        if not self.config_dir.is_dir():
            return found
        for path in sorted(self.config_dir.iterdir()):
            name = path.name
            if not path.is_file() or name in _STORE_FILES:
                continue
            if name.endswith((".json", ".json.corrupt", ".db")):
                found.append(self._read(path, name))
        uploads = self.config_dir / self.uploads_dir
        if uploads.is_dir():
            for path in sorted(uploads.glob("*.json")):
                if path.is_file():
                    found.append(self._read(path, "%s/%s" % (self.uploads_dir, path.name)))
        return found

    @staticmethod
    def _read(path: Path, name: str) -> LegacyFile:
        data = path.read_bytes()
        if not name.endswith(".db"):
            return LegacyFile.of(name, data)
        rows, error = _sqlite_rows(path)
        return LegacyFile.of(name, data, sqlite_rows=rows, error=error)

    # -- converting (pure) ----------------------------------------------------------

    def convert(self, file: LegacyFile) -> Rows:
        """The rows ``file`` stands for. Pure; raises
        :class:`LegacyImportError` only for unreadable web settings."""
        name = file.name
        if name == self.settings_file:
            return Rows(state=tuple(self._settings(file)), classified=True)
        if name.startswith(self.uploads_dir + "/"):
            return self._upload(file)
        if name.endswith(".json.corrupt"):
            return Rows()
        stem, suffix = _split(name)
        if stem in self.printer_stores:
            entries, errors = self._entries(file, suffix)
            if entries is None:
                return Rows(errors=tuple(errors))
            return Rows(
                state=tuple(printer_rows(stem, entries)),
                classified=True,
                errors=tuple(errors),
            )
        if stem in self.account_stores and suffix == ".json":
            entries, errors = self._entries(file, suffix)
            if entries is None:
                return Rows(errors=tuple(errors))
            puts, more = account_rows(stem, self.account_stores[stem], entries)
            return Rows(state=tuple(puts), classified=True, errors=tuple(errors + more))
        return Rows()

    def _entries(
        self, file: LegacyFile, suffix: str
    ) -> Tuple[Optional[List[Any]], List[str]]:
        """A config file's entries, as the old manager would have read them;
        ``None`` (with the error) when the file cannot be read at all."""
        if suffix == ".db":
            if file.sqlite_rows is None:
                return None, ["%s: %s" % (file.name, file.error or "not an SQLite config")]
            entries = []
            for _pk, _sk, text in file.sqlite_rows:
                try:
                    entries.append(json.loads(text))
                except (TypeError, ValueError):
                    entries.append(text)  # kept as an unreadable entry
            return entries, []
        try:
            value = json.loads(file.data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            return None, ["%s: not valid JSON (%s)" % (file.name, error)]
        # Valid JSON but not the registry's list: the whole value is one
        # entry, as JsonConfigManager reads it.
        return (value if isinstance(value, list) else [value]), []

    def _settings(self, file: LegacyFile) -> List[Put]:
        try:
            value = json.loads(file.data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise LegacyImportError("%s is unreadable: %s" % (file.name, error))
        if not isinstance(value, dict):
            raise LegacyImportError("%s is not a settings object" % file.name)
        return [
            Put(
                "settings",
                (
                    "host_secrets" if key in SETTINGS_SECRETS else self.settings_namespace,
                    key,
                ),
                {"value": item},
            )
            for key, item in sorted(value.items())
        ]

    def _upload(self, file: LegacyFile) -> Rows:
        # Upload records are advisory: an unreadable one is simply not
        # carried over (the next upload writes a fresh record).
        stem = file.name[len(self.uploads_dir) + 1 :][: -len(".json")]
        try:
            value = json.loads(file.data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            return Rows(classified=True, errors=("%s: %s" % (file.name, error),))
        return Rows(
            cache=(Put("kv", (UPLOADS_DOMAIN, "printer:%s" % stem, "record"), {"value": value}),),
            classified=True,
        )

    # -- deciding what is new (pure) -------------------------------------------------

    @staticmethod
    def pending(
        files: Iterable[LegacyFile], recorded: Iterable[Tuple[str, str]]
    ) -> List[LegacyFile]:
        """The files whose ``(name, sha256)`` is not recorded yet."""
        seen = set(recorded)
        return [f for f in files if (f.name, f.sha256) not in seen]

    # -- writing ----------------------------------------------------------------------

    def run(self, state: Store, cache: Optional[Store] = None) -> ImportReport:
        """Scan, convert and write every new file. Everything is converted
        before anything is written, so unreadable settings write nothing."""
        recorded = [(f["name"], f["sha256"]) for f in state.legacy_files()]
        todo = [(f, self.convert(f)) for f in self.pending(self.scan(), recorded)]
        report = ImportReport()
        with state.transaction() as txn:
            for file, rows in todo:
                txn.put_legacy_file(
                    file.name, file.sha256, file.data, classified=rows.classified
                )
                report.recorded.append(file.name)
                report.rows += _apply(txn, rows.state)
                self._account(report, file, rows)
        self._write_cache(cache, [rows for _, rows in todo], report)
        return report

    def import_unclassified(
        self, state: Store, cache: Optional[Store] = None
    ) -> ImportReport:
        """Convert recorded files that were not recognised when recorded but
        are now (a plugin naming them has mounted). The newest version of
        each file is used."""
        newest: Dict[str, Dict[str, Any]] = {}
        for entry in state.legacy_files():
            if entry["name"].endswith(".json.corrupt"):
                continue
            current = newest.get(entry["name"])
            if current is None or entry["imported_at"] >= current["imported_at"]:
                newest[entry["name"]] = entry
        todo = []
        for entry in newest.values():
            if entry["classified"]:
                continue
            data = state.legacy_bytes(entry["name"], entry["sha256"])
            if data is None:
                continue
            file = _from_bytes(entry["name"], data)
            rows = self.convert(file)
            if rows.classified:
                todo.append((file, rows))
        report = ImportReport()
        if not todo:
            return report
        with state.transaction() as txn:
            for file, rows in todo:
                report.rows += _apply(txn, rows.state)
                txn.mark_legacy_classified(file.name, file.sha256)
                self._account(report, file, rows)
        self._write_cache(cache, [rows for _, rows in todo], report)
        return report

    @staticmethod
    def _account(report: ImportReport, file: LegacyFile, rows: Rows) -> None:
        (report.converted if rows.classified else report.unclassified).append(file.name)
        report.errors.extend(rows.errors)

    @staticmethod
    def _write_cache(
        cache: Optional[Store], converted: List[Rows], report: ImportReport
    ) -> None:
        puts = [put for rows in converted for put in rows.cache]
        if cache is None or not puts:
            return
        with cache.transaction() as txn:
            report.rows += _apply(txn, puts)


# -- pure conversions -----------------------------------------------------------------


def printer_rows(store_name: str, entries: List[Any]) -> List[Put]:
    """Printer rows for one storage name's entries.

    An entry without a ``unique_id`` gets one derived from its storage name
    and id/token (or its position, for an entry with neither), written into
    its data: re-importing the same file gives the same id, so a rollback
    re-import never duplicates a printer. An entry that is not an object is
    kept as an unreadable row."""
    puts: List[Put] = []
    used: Set[str] = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            data = dict(entry)
            unique_id = data.get("unique_id")
            if not isinstance(unique_id, str) or not unique_id:
                unique_id = _minted_id(store_name, data, index, used)
                data["unique_id"] = unique_id
            row = {
                "store": store_name,
                "pk": data.get("id"),
                "sk": data.get("token"),
                "data": data,
                "schema": 1,
                "state": "ok",
                "error": None,
            }
        else:
            unique_id = _minted_id(store_name, None, index, used)
            row = {
                "store": store_name,
                "pk": None,
                "sk": None,
                "data": entry,
                "schema": 1,
                "state": "unreadable",
                "error": "entry is not a JSON object",
            }
        used.add(unique_id)
        puts.append(Put("printers", unique_id, row))
    return puts


def account_rows(
    store_name: str, fmt: AccountFormat, entries: List[Any]
) -> Tuple[List[Put], List[str]]:
    """Account rows for one account kind's records; a record without its id
    cannot be keyed and is reported (it stays in ``legacy_files``)."""
    puts: List[Put] = []
    errors: List[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get(fmt.key) is None:
            errors.append(
                "%s.json: record %d has no %r; kept in legacy_files only"
                % (store_name, index, fmt.key)
            )
            continue
        secondary = entry.get(fmt.secondary) if fmt.secondary else None
        puts.append(
            Put(
                "accounts",
                (store_name, str(entry[fmt.key])),
                {
                    "secondary": None if secondary is None else str(secondary),
                    "data": dict(entry),
                },
            )
        )
    return puts, errors


def _minted_id(
    store_name: str, data: Optional[Dict[str, Any]], index: int, used: Set[str]
) -> str:
    if data is not None and (data.get("id") or data.get("token")):
        seed = "%s\n%s\n%s" % (store_name, data.get("id"), data.get("token"))
    else:
        seed = "%s\n#%d" % (store_name, index)
    minted = str(uuid.uuid5(_LEGACY_NAMESPACE, seed))
    if minted in used:
        # Two entries with the same id and token: tell them apart by position.
        minted = str(uuid.uuid5(_LEGACY_NAMESPACE, "%s\n#%d" % (seed, index)))
    return minted


def _split(name: str) -> Tuple[str, str]:
    for suffix in (".json", ".db"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


def _apply(txn: Any, puts: Iterable[Put]) -> int:
    """Write ``puts``. A printer or account already stored keeps the fields
    the file does not have (the file came from an older build)."""
    changed = 0
    for put in puts:
        row = dict(put.row)
        if put.table in ("printers", "accounts") and isinstance(row.get("data"), dict):
            stored = txn.get(put.table, put.key)
            if stored is not None and isinstance(stored.get("data"), dict):
                row["data"] = {**stored["data"], **row["data"]}
        changed += int(txn.put(put.table, put.key, row))
    return changed


def _sqlite_rows(
    path: Path,
) -> Tuple[Optional[Tuple[Tuple[Any, Any, Any], ...]], Optional[str]]:
    """The ``printers(pk, sk, data)`` rows of an old SQLiteConfigManager file,
    opened read-only so the file is not touched."""
    try:
        conn = sqlite3.connect(
            "file:%s?mode=ro" % quote(path.resolve().as_posix()), uri=True
        )
    except sqlite3.Error as error:
        return None, str(error)
    try:
        rows = conn.execute("SELECT pk, sk, data FROM printers").fetchall()
        return tuple(tuple(r) for r in rows), None
    except sqlite3.Error as error:
        return None, "%s: %s" % (type(error).__name__, error)
    finally:
        conn.close()


def _from_bytes(name: str, data: bytes) -> LegacyFile:
    """A recorded file back as a :class:`LegacyFile` (an SQLite one is read
    through a private temporary copy)."""
    if not name.endswith(".db"):
        return LegacyFile.of(name, data)
    fd, temporary = tempfile.mkstemp(suffix=".db")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        rows, error = _sqlite_rows(Path(temporary))
    finally:
        os.unlink(temporary)
    return LegacyFile.of(name, data, sqlite_rows=rows, error=error)
