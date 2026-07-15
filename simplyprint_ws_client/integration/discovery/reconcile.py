"""The one owner of "is this discovered device the same printer as a config".

Correlation is by **hardware identity** -- the config/device's direct
``hardware_identity()`` (serial / guid / MAC). It is deliberately *not* the
``unique_id``: that is the stable slot reference (assigned once, survives a
hardware swap), while this matches a re-discovered *physical* device back to its
slot. The host/IP is only a last-resort tiebreaker for a manually-added printer
that has no hardware id -- so a printer changing IP on a DHCP network is followed,
not re-added.

This one rule replaces the bags-of-keys that used to drift apart: add-time
de-duplication, hiding already-added devices from the discovery list, and a
running client's re-discovery host update.

Configs and discovery results expose their address values directly. The matcher
never inspects attribute names or brand-specific fields.
"""

from __future__ import annotations

from ipaddress import ip_address
from typing import Optional, Protocol, Tuple
from urllib.parse import urlsplit


class NetworkIdentity(Protocol):
    def hardware_identity(self) -> Optional[str]: ...

    def network_addresses(self) -> Tuple[str, ...]: ...

    def primary_network_address(self) -> Optional[str]: ...


class DeviceReconciler:
    """Correlates discovered devices against one brand's stored configs.

    Built per config manager (one brand). Owns the single hardware-identity rule
    used by add-time de-duplication and the discovery already-configured filter.
    """

    def __init__(self, manager):
        self._manager = manager

    @staticmethod
    def normalize_identity(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @staticmethod
    def normalize_address(value: Optional[str]) -> Optional[str]:
        """Canonical host portion of an IP, hostname, or device URL."""
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"0", "none"}:
            return None
        literal = text[1:-1] if text.startswith("[") and text.endswith("]") else text
        try:
            return str(ip_address(literal)).lower()
        except ValueError:
            pass
        try:
            parsed = urlsplit(text if "://" in text else f"//{text}")
            host = parsed.hostname
        except ValueError:
            host = None
        normalized = (host or text.rstrip("/")).strip().lower().rstrip(".")
        return normalized or None

    @classmethod
    def _addresses(cls, identity: NetworkIdentity) -> Tuple[str, ...]:
        return tuple(
            address
            for raw in identity.network_addresses()
            if (address := cls.normalize_address(raw)) is not None
        )

    @classmethod
    def same_network_address(
        cls, left: NetworkIdentity, right: NetworkIdentity
    ) -> bool:
        """Whether two identity objects advertise at least one common host."""
        return bool(set(cls._addresses(left)).intersection(cls._addresses(right)))

    @classmethod
    def resolved_hardware_identity(
        cls, device: NetworkIdentity, *, warm: bool = False
    ) -> Optional[str]:
        """A discovered identity's explicit id, or its resolved MAC fallback."""
        identity = cls.normalize_identity(device.hardware_identity())
        if identity is not None:
            return identity
        address = cls.normalize_address(device.primary_network_address())
        if address is None:
            return None
        from simplyprint_ws_client.integration.discovery.mac import resolve_mac

        return cls.normalize_identity(resolve_mac(address, warm=warm))

    @classmethod
    def same_device(
        cls,
        config: NetworkIdentity,
        device: NetworkIdentity,
        *,
        allow_address_match: bool = True,
        warm: bool = False,
    ) -> bool:
        """Whether ``device`` identifies ``config`` under the shared policy.

        Hardware identity wins. Address fallback is allowed only for a config
        that has no hardware identity of its own; a weak sighting must never
        override a stored serial/guid/MAC.
        """
        config_id = cls.normalize_identity(config.hardware_identity())
        device_id = cls.resolved_hardware_identity(device, warm=warm)
        if device_id is not None and config_id == device_id:
            return True
        return (
            allow_address_match
            and config_id is None
            and cls.same_network_address(config, device)
        )

    def matching(self, device: NetworkIdentity) -> Optional[object]:
        """The stored config that is the same printer as this device, or None."""
        if self._manager is None:
            return None
        return next(
            (
                config
                for config in self._manager.get_all()
                if self.same_device(config, device)
            ),
            None,
        )

    def is_configured(self, result) -> bool:
        """True when a discovered ``result`` is already a stored printer."""
        return self.matching(result) is not None

    def duplicate_of(self, config) -> Optional[Tuple[object, str, str]]:
        """An existing config that is the same printer as ``config``, with the
        matched ``(existing, field, value)`` for the error message, or None."""
        hardware_id = self.normalize_identity(config.hardware_identity())
        match = self.matching(config)
        if match is None:
            return None
        if (
            hardware_id is not None
            and self.normalize_identity(match.hardware_identity()) == hardware_id
        ):
            return match, "hardware id", hardware_id
        common = set(self._addresses(config)).intersection(self._addresses(match))
        return match, "network address", sorted(common)[0]
