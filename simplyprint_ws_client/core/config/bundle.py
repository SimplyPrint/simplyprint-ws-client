"""Configuration export/import as one portable bundle.

A :class:`ConfigBundle` packages an explicit, integration-supplied list of on-disk
config files (printer registries, app settings, accounts, ...) into a single ZIP
with a typed manifest, and restores them back atomically.

It is **brand-free**: it never names a brand, a ``ClientApp`` or any settings
model. The caller injects the exact files as :class:`BundleSource` value objects,
so the library packs/unpacks bytes without knowing what they mean. Restore maps
each manifest entry back onto a current source by ``(logical_name, kind)``, so the
absolute paths can differ between the machine that exported and the one importing.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from simplyprint_ws_client.core.files.atomic import atomic_write_bytes

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BundleSource",
    "BundleEntry",
    "BundleManifest",
    "RestoreReport",
    "ConfigBundle",
    "BundleError",
    "BundleCorrupt",
    "BundleVersionError",
]

#: Bundle schema version. Bump on a breaking layout change; ``inspect`` refuses a
#: bundle whose ``format_version`` is newer than this build understands.
BUNDLE_FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"


class BundleError(Exception):
    """A bundle could not be read, or is not a valid config bundle."""


class BundleCorrupt(BundleError):
    """A bundle's contents do not match its manifest (missing member / bad hash)."""


class BundleVersionError(BundleError):
    """A bundle's format version is newer than this build understands."""


@dataclass(frozen=True)
class BundleSource:
    """One archivable file the integration wants in the bundle.

    ``logical_name`` is the store's name (e.g. ``"settings"`` or a registry name);
    ``kind`` is an opaque label (``"config_store"`` / ``"settings"`` /
    ``"accounts"`` / ``"raw"``); ``path`` is the live file on disk. All brand-free:
    the library never inspects these beyond reading/writing the file. ``(logical_name,
    kind)`` must be unique within one bundle -- it is the key restore maps back on.
    """

    logical_name: str
    kind: str
    path: Path


class BundleEntry(BaseModel):
    """One file recorded in the manifest."""

    archive_name: str
    kind: str
    logical_name: str
    size: int
    sha256: str


class BundleManifest(BaseModel):
    """The bundle's table of contents + provenance."""

    format_version: int = BUNDLE_FORMAT_VERSION
    created_at: Optional[datetime] = None
    #: Advisory provenance, supplied by the integration; never gates restore.
    app_version: Optional[str] = None
    library_version: Optional[str] = None
    entries: List[BundleEntry] = Field(default_factory=list)


@dataclass(frozen=True)
class RestoreReport:
    """What a restore did: which logical stores were written vs. skipped (no
    matching source on this machine)."""

    applied: List[str]
    skipped: List[str]


class ConfigBundle:
    """Owns export/inspect/restore of a configuration bundle over a fixed set of
    :class:`BundleSource` files."""

    def __init__(
        self,
        sources: Iterable[BundleSource],
        *,
        app_version: Optional[str] = None,
        library_version: Optional[str] = None,
    ) -> None:
        self._sources: List[BundleSource] = list(sources)
        self._app_version = app_version
        self._library_version = library_version

    @staticmethod
    def _archive_name(source: BundleSource) -> str:
        # Group by kind so members read clearly when unzipped; the real target is
        # resolved on restore by (logical_name, kind), not this path.
        return f"{source.kind}/{source.path.name}"

    def export_to(
        self, buffer: BinaryIO, *, created_at: Optional[datetime] = None
    ) -> BundleManifest:
        """Write every existing source into ``buffer`` as a ZIP and return the
        manifest. A source whose file is absent is simply omitted."""
        entries: List[BundleEntry] = []
        with zipfile.ZipFile(
            buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for source in self._sources:
                if not source.path.exists():
                    continue
                data = source.path.read_bytes()
                archive_name = self._archive_name(source)
                archive.writestr(archive_name, data)
                entries.append(
                    BundleEntry(
                        archive_name=archive_name,
                        kind=source.kind,
                        logical_name=source.logical_name,
                        size=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
            manifest = BundleManifest(
                format_version=BUNDLE_FORMAT_VERSION,
                created_at=created_at or datetime.now(timezone.utc),
                app_version=self._app_version,
                library_version=self._library_version,
                entries=entries,
            )
            # Manifest last, so it can carry every member's hash.
            archive.writestr(MANIFEST_NAME, manifest.model_dump_json(indent=2))
        return manifest

    def export_bytes(self, *, created_at: Optional[datetime] = None) -> io.BytesIO:
        """Convenience: a rewound :class:`io.BytesIO` of the exported ZIP."""
        buffer = io.BytesIO()
        self.export_to(buffer, created_at=created_at)
        buffer.seek(0)
        return buffer

    @staticmethod
    def inspect(buffer: BinaryIO) -> BundleManifest:
        """Validate a bundle and return its manifest, without touching disk.

        Raises :class:`BundleError` (or a subclass) on a bad archive, a missing or
        invalid manifest, an unsupported ``format_version``, or any member that is
        missing or whose bytes do not match the recorded size/hash.
        """
        buffer.seek(0)
        try:
            with zipfile.ZipFile(buffer, "r") as archive:
                try:
                    raw = archive.read(MANIFEST_NAME)
                except KeyError as exc:
                    raise BundleError("bundle has no manifest.json") from exc

                try:
                    manifest = BundleManifest.model_validate_json(raw)
                except Exception as exc:
                    raise BundleError(f"invalid manifest: {exc}") from exc

                if manifest.format_version > BUNDLE_FORMAT_VERSION:
                    raise BundleVersionError(
                        f"bundle format v{manifest.format_version} is newer than "
                        f"the supported v{BUNDLE_FORMAT_VERSION}"
                    )

                names = set(archive.namelist())
                for entry in manifest.entries:
                    if entry.archive_name not in names:
                        raise BundleCorrupt(
                            f"missing bundle member: {entry.archive_name}"
                        )
                    data = archive.read(entry.archive_name)
                    if (
                        len(data) != entry.size
                        or hashlib.sha256(data).hexdigest() != entry.sha256
                    ):
                        raise BundleCorrupt(
                            f"checksum mismatch for {entry.archive_name}"
                        )
                return manifest
        except zipfile.BadZipFile as exc:
            raise BundleError(f"not a valid bundle archive: {exc}") from exc

    def restore_from(self, buffer: BinaryIO) -> RestoreReport:
        """Validate the bundle, then atomically write each member onto the
        matching current source. Validation happens first, so a corrupt bundle
        leaves every target untouched; each write is atomic (tmp + os.replace).

        A member with no matching source on this machine (by ``logical_name`` +
        ``kind``) is skipped and reported, never written to a guessed path.
        """
        manifest = self.inspect(buffer)  # raises on any problem -> nothing written
        by_key: Dict[Tuple[str, str], BundleSource] = {
            (source.logical_name, source.kind): source for source in self._sources
        }

        applied: List[str] = []
        skipped: List[str] = []
        buffer.seek(0)
        with zipfile.ZipFile(buffer, "r") as archive:
            for entry in manifest.entries:
                source = by_key.get((entry.logical_name, entry.kind))
                if source is None:
                    skipped.append(entry.logical_name)
                    continue
                atomic_write_bytes(source.path, archive.read(entry.archive_name))
                applied.append(entry.logical_name)

        return RestoreReport(applied=applied, skipped=skipped)
