"""A config manager over the SQLite store (``ConfigManagerType.STORE``).

Same interface as the JSON and SQLite managers, three differences that are
the point of the move (docs/persistence-design.md):

* **Rows are keyed by ``unique_id``**, the slot's identity, which never
  changes. ``(pk, sk)`` (id and token) changes when a printer is keyed, so it
  is only an indexed column. A config that has no ``unique_id`` yet gets one
  when it is first written, assigned on the config itself.
* **Unknown fields survive.** Each row keeps the raw dict it was read as, and
  the dumped model is laid on top of it, so a field this build does not know
  (written by a newer build, before an OTA rollback) is written back as it
  was. Only an explicit migration drops a field.
* **Writes are coalesced.** :meth:`flush` marks rows in the store's writer
  instead of rewriting a file. A change to fields waits out the writer's
  debounce (1 s, at most 5 s); adding or removing a printer, and
  ``flush(config)``, which the app calls on add and remove, commit before
  returning, except on an event loop: there they are urgent marks the
  store's thread commits at once, because the loop never waits for the
  write lock or an fsync.

Config versions: a config type may declare ``config_version`` (an int) and
``upgrade_config(data, from_version) -> dict`` (the SDK's ``DriverConfig``
does, from its ``@migration`` steps). A row stored at an older version is
upgraded when it is read and written back at the new one (the old row is
archived). A row stored at a *newer* version than the type knows is reported
unreadable and never rewritten.

Which table: printers by default; a config type with ``store_table =
"accounts"`` (the client's account records) is stored in ``accounts``,
keyed by ``(storage name, account id)``, as the provider's record exactly.

All managers on one file share one :class:`~simplyprint_ws_client.core.store.Store`
(see :meth:`StoreConfigManager.bind` and ``Store.shared``).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from simplyprint_ws_client.core.config.config import Config, PrinterConfig
from simplyprint_ws_client.core.config.manager import ConfigManager, UnreadableConfig
from simplyprint_ws_client.core.store import SKIP, Key, Store

__all__ = ["StoreConfigManager", "STORE_FILE"]

_logger = logging.getLogger("config.store")

#: The durable store's file name, in the manager's base directory unless a
#: store is bound or passed.
STORE_FILE = "state.db"


class _Rows:
    """How one kind of config maps onto one store table."""

    table = "printers"
    #: Lay the dumped model over the stored dict (keep unknown fields).
    overlay = True
    #: Rows carry a config version (``schema`` column).
    versioned = True

    def key(self, manager: "StoreConfigManager", config: Config) -> Key:
        unique_id = getattr(config, "unique_id", None)
        if not unique_id:
            # The slot reference is assigned once, at the first write, and
            # kept on the config: from now on it is this printer's row.
            unique_id = str(uuid.uuid4())
            config.unique_id = unique_id  # type: ignore[attr-defined]
        return str(unique_id)

    def row(
        self, manager: "StoreConfigManager", config: Config, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "store": manager.name,
            "pk": data.get("id"),
            "sk": data.get("token"),
            "data": data,
            "schema": manager.config_version,
            "state": "ok",
            "error": None,
        }

    def stored(self, manager: "StoreConfigManager") -> List[Tuple[Key, Dict[str, Any]]]:
        return [
            (row["unique_id"], row)
            for row in manager.store.rows(self.table, store=manager.name)
        ]

    def adopt(self, config: Config, key: Key) -> None:
        # The row's key is the identity; a config read without (or with a
        # hand-edited) unique_id takes it, so its next write is the same row.
        if getattr(config, "unique_id", None) != key:
            config.unique_id = key  # type: ignore[attr-defined]

    def unreadable_row(
        self, manager: "StoreConfigManager", row: Dict[str, Any], error: str
    ) -> Optional[Dict[str, Any]]:
        # The data stays exactly as stored; only the state says why it is not
        # in use (for the CLI and the UI).
        return {**_values(row), "state": "unreadable", "error": error}


class _AccountRows(_Rows):
    table = "accounts"
    # An account record is the provider's JSON as it is: a field the provider
    # drops is gone.
    overlay = False
    versioned = False

    def key(self, manager: "StoreConfigManager", config: Config) -> Key:
        account_id = getattr(config, "account_id", None)
        return (manager.name, str(account_id if account_id is not None else config.pk))

    def row(
        self, manager: "StoreConfigManager", config: Config, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        secondary = config.sk
        return {"secondary": secondary or None, "data": data}

    def stored(self, manager: "StoreConfigManager") -> List[Tuple[Key, Dict[str, Any]]]:
        return [
            ((row["store"], row["account_id"]), row)
            for row in manager.store.rows(self.table, store=manager.name)
        ]

    def adopt(self, config: Config, key: Key) -> None:
        pass

    def unreadable_row(
        self, manager: "StoreConfigManager", row: Dict[str, Any], error: str
    ) -> Optional[Dict[str, Any]]:
        return None  # accounts have no state column: leave the row alone


def _values(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in ("unique_id", "rev")}


class StoreConfigManager(ConfigManager):
    """``ConfigManager`` whose storage name is a namespace in the store."""

    #: The store every manager uses unless one is passed; the host binds its
    #: ``state.db`` once at start (:meth:`bind`).
    _bound: ClassVar[Optional[Store]] = None

    def __init__(
        self,
        name: str = "printers",
        config_t: Any = None,
        base_directory: Optional[str] = None,
        *,
        store: Optional[Store] = None,
    ) -> None:
        config_t = PrinterConfig if config_t is None else config_t
        self._explicit_store = store
        self._store: Optional[Store] = None
        # Guards the maps below. Never held while waiting for the store's
        # write lock: the writer thread calls _produce, which takes it.
        self._lock = threading.RLock()
        #: The dict each config was read as or last written as.
        self._raw: Dict[Config, Dict[str, Any]] = {}
        #: The key each config was last stored under.
        self._row_key: Dict[Config, Key] = {}
        #: Keys of this namespace's rows this manager knows are stored.
        self._stored: Set[Key] = set()
        self._rows: _Rows = (
            _AccountRows()
            if getattr(config_t, "store_table", "printers") == "accounts"
            else _Rows()
        )
        super().__init__(name, config_t, base_directory)

    @classmethod
    def bind(cls, store: Optional[Store]) -> None:
        """Make ``store`` the one every manager without an explicit store
        uses (``None`` unbinds). The client binds its ``state.db`` before
        building the app, so ``ConfigManagerType.STORE`` needs no path."""
        cls._bound = store

    @property
    def store(self) -> Store:
        if self._store is None or (
            self._store.closed and self._explicit_store is None
        ):
            self._store = (
                self._explicit_store
                or StoreConfigManager._bound
                or Store.shared(self.base_directory / STORE_FILE)
            )
        return self._store

    @property
    def config_version(self) -> int:
        version = getattr(self.config_t, "config_version", 1)
        return version if isinstance(version, int) and version >= 1 else 1

    # -- ConfigManager -----------------------------------------------------------

    def load(self):
        store = self.store
        # This process's own pending marks first, so a load reads them. Not
        # on the event loop: flush_now waits for the write lock (a snapshot
        # may hold it for seconds) and an fsync. A load there (an account
        # kind following the store after clear()) reads what is committed;
        # its own forgotten marks produce nothing (SKIP) anyway.
        if not _on_event_loop():
            store.writer.flush_now()
        rows = self._rows.stored(self)
        upgraded: List[Tuple[Key, Config]] = []
        unreadable_rows: List[Tuple[Key, Dict[str, Any]]] = []
        with self._lock:
            self.unreadable = []
            for key, row in rows:
                self._stored.add(key)
                config, error = self._read_row(key, row)
                if config is None:
                    if error is not None and row.get("error") != error:
                        unreadable_rows.append((key, self._rows.unreadable_row(self, row, error)))
                    continue
                if self._rows.versioned and (row.get("schema") or 1) < self.config_version:
                    upgraded.append((key, config))
        for key, values in unreadable_rows:
            if values is not None:
                store.writer.mark(self._rows.table, key, lambda v=values: v)
        for key, config in upgraded:
            store.writer.mark(
                self._rows.table, key, lambda c=config, k=key: self._produce(c, k)
            )

    def _read_row(
        self, key: Key, row: Dict[str, Any]
    ) -> Tuple[Optional[Config], Optional[str]]:
        """One row -> a config, or ``(None, error)`` with the row kept as an
        unreadable entry. Caller holds ``_lock``."""
        raw = row.get("data")
        schema = row.get("schema") or 1
        if self._rows.versioned and schema > self.config_version:
            # Written by a newer build: never rewritten, not even its state.
            self._unreadable(
                key,
                raw,
                "config schema %d is newer than this build's %d"
                % (schema, self.config_version),
            )
            return None, None
        data = raw
        if self._rows.versioned and schema < self.config_version:
            upgrade = getattr(self.config_t, "upgrade_config", None)
            if upgrade is not None:
                try:
                    data = upgrade(dict(raw), schema)
                except Exception as e:
                    error = "upgrade from config schema %d failed: %s: %s" % (
                        schema,
                        type(e).__name__,
                        e,
                    )
                    self._unreadable(key, raw, error)
                    return None, error
        try:
            config = self.config_t.from_dict(data)
        except Exception as e:
            # One bad row must not stop the others from loading; it stays in
            # the store, untouched, for a build or a fix that can read it.
            error = "%s: %s" % (type(e).__name__, e)
            self._unreadable(key, raw, error)
            return None, error
        self.persist(config)
        self._rows.adopt(config, key)
        self._raw[config] = dict(data) if isinstance(data, dict) else {}
        self._row_key[config] = key
        return config, None

    def _unreadable(self, key: Key, raw: Any, error: str) -> None:
        unique_id = key if isinstance(key, str) else None
        entry = UnreadableConfig(
            raw=raw,
            error=error,
            unique_id=unique_id,
            key=None if unique_id is not None else tuple(key),  # type: ignore[arg-type]
        )
        if not any(e.unique_id == unique_id and e.key == entry.key for e in self.unreadable):
            self.unreadable.append(entry)
        _logger.error(
            "%s: printer config %s is unreadable (kept in the store): %s",
            self.name,
            entry.label,
            error,
        )

    def _unreadable_keys(self) -> Set[Key]:
        keys: Set[Key] = set()
        for entry in self.unreadable:
            if entry.unique_id is not None:
                keys.add(entry.unique_id)
            elif entry.key is not None:
                keys.add(tuple(entry.key))
        return keys

    def flush(self, config: Optional[Config] = None):
        store = self.store
        table = self._rows.table
        with self._lock:
            desired: Dict[Key, Config] = {}
            rekeyed: Set[Key] = set()
            for held in self.get_all():
                try:
                    # Blank configs (a pending slot) are not stored, as before.
                    if held.is_empty():
                        continue
                    key = self._rows.key(self, held)
                except Exception:
                    _logger.exception(
                        "%s: could not key printer config %s; keeping its row",
                        self.name,
                        getattr(held, "unique_id", None),
                    )
                    previous = self._row_key.get(held)
                    if previous is not None:
                        desired[previous] = held
                    continue
                previous = self._row_key.get(held)
                if previous is not None and previous != key:
                    rekeyed.add(previous)
                self._row_key[held] = key
                desired[key] = held
            # Unreadable rows are not "detached": they stay untouched, so the
            # printer comes back once a build (or a fix) can read them.
            removed = (self._stored | rekeyed) - set(desired) - self._unreadable_keys()
            added = set(desired) - self._stored
            registration = bool(config is not None or added or removed)
            for key, held in desired.items():
                store.writer.mark(
                    table,
                    key,
                    lambda c=held, k=key: self._produce(c, k),
                    urgent=registration,
                )
            for key in removed:
                store.writer.mark(table, key, _delete, urgent=True)
            self._stored = (self._stored - removed) | set(desired)
        if registration and not _on_event_loop():
            # Registration (add, remove, an explicit config) must be on disk
            # before the caller moves on; field changes ride the debounce.
            # On the event loop it is an urgent mark instead: flush_now waits
            # for the write lock and an fsync, and the loop never waits. The
            # store's thread commits it at once, in mark order with every
            # other write of this process.
            store.writer.flush_now()

    def _produce(self, config: Config, key: Key) -> Optional[Dict[str, Any]]:
        """The row for ``config`` now; called by the store's writer."""
        dumped = config.as_dict()
        with self._lock:
            if self._row_key.get(config) != key:
                # Re-keyed or forgotten (clear) since the mark: a newer mark,
                # if any, says what happens to the row; this one says nothing.
                return SKIP
            if self._rows.overlay:
                data = dict(self._raw.get(config) or {})
                data.update(dumped)
            else:
                data = dict(dumped)
            self._raw[config] = data
            return self._rows.row(self, config, data)

    def clear(self):
        super().clear()
        with self._lock:
            # The rows stay (``_stored``): a flush now deletes them, exactly
            # as the JSON manager writes an empty list.
            self._raw = {}
            self._row_key = {}

    def delete_storage(self):
        store = self.store
        store.writer.flush_now()
        keys = {key for key, _ in self._rows.stored(self)}
        with self._lock:
            keys |= self._stored | set(self._row_key.values())
            self._stored = set()
            self._row_key = {}
            self._raw = {}
        for key in keys:
            self.store.writer.mark(self._rows.table, key, _delete, urgent=True)
        store.writer.flush_now()

    def backup_storage(self, *args, **kwargs):
        store = self.store
        store.writer.flush_now()
        directory = store.backups or store.path.parent / "backups"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        return store.snapshot_to(directory / ("manual-%s-%s.db" % (store.path.stem, stamp)))


def _delete() -> None:
    return None


def _on_event_loop() -> bool:
    """Whether the caller runs on an event loop (the app's, or any other):
    there nothing may wait for the store's write lock or an fsync."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
