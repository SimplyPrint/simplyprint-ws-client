"""Persistent, TTL-bounded store of LAN discovery results.

The on-demand discover endpoints recompute results per request, so the
add-printer UI lost everything it had found whenever the page reloaded or the
user navigated away -- and a fresh subnet scan had to start from scratch. This
store keeps the most recent results in memory with a per-entry TTL: the UI can
repaint instantly after a reload and poll for new devices while an expensive
scan runs in the background.

Lives one level up from ``core/`` alongside the rest of the discovery subsystem
because only some brands discover over the LAN. It stays brand-free -- the scan
fan-out and the brand list are *injected*, so this class never imports a brand.
That is the same bar the ``_DISCOVERY_MODULES`` guard in
``tests/test_architecture.py`` holds the other shared discovery files to.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from simplyprint_ws_client.integration.discovery.device import (
    DiscoveredDevice,
    JsonValue,
)
from simplyprint_ws_client.common.utils.expiring_dict import ExpiringDict

DiscoveryExtra = Dict[str, JsonValue]

#: How long a discovered device lingers in the store after it was last seen. A
#: device that keeps answering refreshes its own TTL, so only ones that have
#: actually gone quiet fall out.
_DEFAULT_TTL = 300

#: ``async (client_type, timeout) -> list[DiscoveredDevice] | None`` -- the
#: existing neutral discovery fan-out, injected so this module imports no brand.
ScanFn = Callable[[str, float], Awaitable[Optional[List[DiscoveredDevice]]]]

#: ``() -> [client_type, ...]`` -- the brands worth scanning, injected likewise.
BrandsFn = Callable[[], List[str]]


def _normalise_host(value: Optional[str]) -> str:
    """A host string normalised for identity comparison (same rule the web
    layer's reconciler applies): case, scheme and trailing slashes are
    cosmetic, so ``HTTP://Printer.local/`` and ``printer.local`` are one box."""
    if value is None:
        return ""
    value = str(value).strip().lower()
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.rstrip("/")


@dataclass(frozen=True)
class DiscoveryResult:
    """A discovered device tagged with the brand whose scan found it.

    The neutral :class:`DiscoveredDevice` facts plus the ``type`` (client type)
    so one merged list can carry devices from every brand without this store
    ever naming one.
    """

    type: str
    host: str
    name: Optional[str] = None
    serial: Optional[str] = None
    extra: DiscoveryExtra = field(default_factory=dict)


class DiscoveryResultsStore:
    """Owns the discovery-results lifecycle: trigger -> collect -> retain -> serve.

    A single injected instance the web layer reads from. :meth:`scan` runs the
    injected fan-out and folds the devices found into a TTL dict; :meth:`current`
    returns the live (non-expired) snapshot, optionally filtered to one brand.
    """

    def __init__(
        self, scan_fn: ScanFn, brands_fn: BrandsFn, ttl: int = _DEFAULT_TTL
    ) -> None:
        self.logger = logging.getLogger("discovery")
        self._scan_fn = scan_fn
        self._brands_fn = brands_fn
        self._results: ExpiringDict = ExpiringDict(ttl=ttl)
        self._scanning = False

    @property
    def scanning(self) -> bool:
        """True while a :meth:`scan` triggered earlier is still in flight."""
        return self._scanning

    def current(self, client_type: Optional[str] = None) -> List[DiscoveryResult]:
        """Live (non-expired) results, optionally narrowed to one ``client_type``."""
        results = [value for _, value in self._results.items()]
        if client_type is not None:
            results = [result for result in results if result.type == client_type]
        return results

    async def scan(
        self, timeout: float = 5.0, client_type: Optional[str] = None
    ) -> None:
        """Run a discovery pass, folding fresh devices into the store.

        Concurrent triggers collapse: while a scan is in flight, further calls
        return immediately instead of stacking duplicate network scans. Each
        brand's devices land as that brand finishes, so instant passive
        (multicast) hits show up before a slower subnet scan completes. The
        check-and-set is synchronous (no ``await`` between them), so on a single
        event loop two callers can never both pass the guard.
        """
        if self._scanning:
            return

        self._scanning = True
        try:
            brands = [client_type] if client_type else list(self._brands_fn())
            await asyncio.gather(
                *(self._scan_one(brand, timeout) for brand in brands),
                return_exceptions=True,
            )
        finally:
            self._scanning = False

    async def _scan_one(self, brand: str, timeout: float) -> None:
        try:
            devices = await self._scan_fn(brand, timeout)
        except Exception:
            self.logger.exception("discovery scan failed for %s", brand)
            return

        for device in devices or []:
            self._store(brand, device)

    def _store(self, brand: str, device: DiscoveredDevice) -> None:
        # Dedupe on (brand, stable id); a re-scan overwrites in place, refreshing
        # the TTL so a device that keeps answering never expires under the UI.
        #
        # One physical printer can be sighted through several paths at once --
        # a multicast announcement that carries its serial AND a subnet probe
        # that only knows its host -- so the two key shapes are reconciled
        # here: the serial entry is authoritative, and a host-only sighting of
        # an already-known box refreshes that entry instead of duplicating it.
        host = _normalise_host(device.host)
        if device.serial:
            # A host-only sighting of the same box may already be stored under
            # its host key; the serial-keyed entry supersedes it.
            try:
                del self._results[(brand, host)]
            except KeyError:
                pass
            key = (brand, device.serial)
        else:
            for existing_key in self._results.keys():
                if existing_key[0] != brand:
                    continue
                existing = self._results.get(existing_key)
                if existing is not None and _normalise_host(existing.host) == host:
                    # Same box, poorer facts: keep the richer entry, refresh
                    # its TTL so the box doesn't expire under the UI.
                    self._results[existing_key] = existing
                    return
            key = (brand, host)
        self._results[key] = DiscoveryResult(
            type=brand,
            host=device.host,
            name=device.name,
            serial=device.serial,
            extra=dict(device.extra),
        )
