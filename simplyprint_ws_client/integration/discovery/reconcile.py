"""The one owner of "is this discovered device the same printer as a config".

Correlation is by **hardware identity** -- the brand's ``stable_hardware_id``
(serial / guid) or, failing that, the device's MAC. It is deliberately *not* the
``unique_id``: that is the stable slot reference (assigned once, survives a
hardware swap), while this matches a re-discovered *physical* device back to its
slot. The host/IP is only a last-resort tiebreaker for a manually-added printer
that has no hardware id -- so a printer changing IP on a DHCP network is followed,
not re-added.

This one rule replaces the bags-of-keys that used to drift apart: add-time
de-duplication, hiding already-added devices from the discovery list, and a
running client's re-discovery host update.

Which config fields may carry an address comes from the config class
(:attr:`~simplyprint_ws_client.core.config.PrinterConfig.network_address_fields`),
so a brand with an unusually-named address field extends its own config rather
than this module.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _norm(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _address_fields(config) -> Tuple[str, ...]:
    return type(config).network_address_fields


def config_hardware_id(config) -> Optional[str]:
    """A config's hardware-match id: its brand id (serial/guid), else its MAC."""
    return _norm(config.stable_hardware_id()) or _norm(getattr(config, "mac", None))


def device_hardware_id(
    host: Optional[str],
    serial: Optional[str],
    extra: Optional[dict],
    *,
    warm: bool = False,
) -> Optional[str]:
    """A discovered device's hardware-match id: serial, else MAC.

    The serial (the SSDP norm) wins; otherwise the MAC -- taken from ``extra`` when
    a discovery source already supplied it, else resolved from the host's OS
    neighbour table. ``warm`` lets a create path pay a brief knock to populate that
    table; the read-only already-configured filter leaves it ``False`` and relies
    on the warm cache the discovery probe just left behind.
    """
    extra = extra or {}
    identity = _norm(serial) or _norm(extra.get("mac"))
    if identity is not None:
        return identity
    if not host:
        return None
    from simplyprint_ws_client.integration.discovery.mac import resolve_mac

    return _norm(resolve_mac(host, warm=warm))


class DeviceReconciler:
    """Correlates discovered devices against one brand's stored configs.

    Built per config manager (one brand). Owns the single hardware-identity rule
    used by add-time de-duplication and the discovery already-configured filter.
    """

    def __init__(self, manager):
        self._manager = manager

    def matching(
        self, *, hardware_id: Optional[str], host: Optional[str]
    ) -> Optional[object]:
        """The stored config that is the same printer as this device, or None."""
        if self._manager is None:
            return None
        if hardware_id is not None:
            for config in self._manager.get_all():
                if config_hardware_id(config) == hardware_id:
                    return config
        # Fallback: address-only match when neither side has a hardware id, so a
        # manually-added (IP-only) printer is still de-duplicated.
        norm_host = _norm(host)
        if norm_host is not None:
            for config in self._manager.get_all():
                if config_hardware_id(config) is not None:
                    continue
                if any(
                    _norm(getattr(config, key, None)) == norm_host
                    for key in _address_fields(config)
                ):
                    return config
        return None

    def is_configured(self, result) -> bool:
        """True when a discovered ``result`` is already a stored printer."""
        hardware_id = device_hardware_id(result.host, result.serial, result.extra)
        return self.matching(hardware_id=hardware_id, host=result.host) is not None

    def duplicate_of(self, config) -> Optional[Tuple[object, str, str]]:
        """An existing config that is the same printer as ``config``, with the
        matched ``(existing, field, value)`` for the error message, or None."""
        hardware_id = config_hardware_id(config)
        host = next(
            (
                value
                for key in _address_fields(config)
                if (value := getattr(config, key, None))
            ),
            None,
        )
        match = self.matching(hardware_id=hardware_id, host=host)
        if match is None:
            return None
        if hardware_id is not None and config_hardware_id(match) == hardware_id:
            return match, "hardware id", hardware_id
        for key in _address_fields(config):
            value = _norm(getattr(config, key, None))
            if value is not None and _norm(getattr(match, key, None)) == value:
                return match, key, str(getattr(config, key))
        return match, "hardware id", str(hardware_id)
