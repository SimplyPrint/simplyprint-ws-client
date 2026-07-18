"""The identity seam: slot id assigned once (never re-keyed) + hardware-id capture.

Ported from the integration when the seam was promoted into the library (2.0
slice A); the assertions are unchanged, only the brand config was replaced by a
neutral test config.
"""

import uuid
from typing import Optional
from unittest.mock import patch

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.integration.discovery.identity import (
    assign_unique_id,
    capture_hardware_id,
)
from simplyprint_ws_client.integration.discovery import DiscoveredDevice
from simplyprint_ws_client.integration.discovery.reconcile import DeviceReconciler


class SerialConfig(PrinterConfig):
    """A neutral brand-like config: serial as the hardware id, host as address."""

    serial: Optional[str] = None
    host: Optional[str] = None

    def hardware_identity(self) -> Optional[str]:
        return self.serial or super().hardware_identity()

    def network_addresses(self) -> tuple[str, ...]:
        return (self.host,) if self.host else ()


class UriConfig(PrinterConfig):
    """A config reaching its device through a custom-named address field."""

    device_uri: Optional[str] = None

    def network_addresses(self) -> tuple[str, ...]:
        return (self.device_uri,) if self.device_uri else ()


def test_assign_unique_id_mints_a_stable_uuid_when_missing() -> None:
    config = SerialConfig.get_blank()
    assert config.unique_id is None

    assigned = assign_unique_id(config)

    assert uuid.UUID(assigned)  # a valid slot-reference UUID
    assert config.unique_id == assigned


def test_assign_unique_id_never_rekeys_an_existing_slot_id() -> None:
    # The slot id is stable for life: present -> preserved, even when pending.
    config = SerialConfig.get_new()
    config.serial = "SERIAL-1"
    config.unique_id = "existing-slot-id"

    assert assign_unique_id(config) == "existing-slot-id"
    assert config.unique_id == "existing-slot-id"


def test_assign_unique_id_preserves_a_legacy_derived_slot_id() -> None:
    # Configs minted under the old salted-sha1 scheme keep their id untouched.
    config = SerialConfig.get_new()
    config.unique_id = "0c2b4b7a9d1e3f5a7c9b1d3f5a7c9b1d3f5a7c9b"  # sha1-shaped

    assert assign_unique_id(config) == config.unique_id


def test_capture_hardware_id_resolves_mac_when_brand_has_no_serial() -> None:
    config = SerialConfig.get_new()
    config.serial = None
    config.host = "10.0.0.7"

    with patch(
        "simplyprint_ws_client.integration.discovery.mac.resolve_mac",
        return_value="aa:bb:cc:dd:ee:ff",
    ):
        capture_hardware_id(config)

    assert config.mac == "aa:bb:cc:dd:ee:ff"


def test_capture_hardware_id_skips_when_brand_has_a_serial() -> None:
    config = SerialConfig.get_new()
    config.serial = "SERIAL-1"
    config.host = "10.0.0.7"

    # Must not need the network when the brand already has a hardware id.
    capture_hardware_id(config)

    assert config.mac is None


def test_capture_hardware_id_uses_the_config_address_values() -> None:
    # The typed address hook returns values, not field names for reflection.
    config = UriConfig.get_new()
    config.device_uri = "10.0.0.9"

    with patch(
        "simplyprint_ws_client.integration.discovery.mac.resolve_mac",
        return_value="11:22:33:44:55:66",
    ) as resolve:
        capture_hardware_id(config)

    resolve.assert_called_once_with("10.0.0.9")
    assert config.mac == "11:22:33:44:55:66"


class _Manager:
    def __init__(self, *configs) -> None:
        self._configs = configs

    def get_all(self):
        return self._configs


def test_reconciler_matches_the_first_class_discovered_hardware_id() -> None:
    config = SerialConfig.get_new()
    config.serial = "SERIAL-1"
    config.host = "10.0.0.7"
    device = DiscoveredDevice(host="10.0.0.99", hardware_id=" serial-1 ", serial=None)

    assert DeviceReconciler(_Manager(config)).matching(device) is config


def test_reconciler_uses_direct_address_fallback_for_identityless_config() -> None:
    config = UriConfig.get_new()
    config.device_uri = "http://PRINTER.local/"
    device = DiscoveredDevice(host="printer.local")

    assert DeviceReconciler(_Manager(config)).matching(device) is config


def test_reconciler_normalizes_urls_and_unbracketed_ipv6_addresses() -> None:
    assert (
        DeviceReconciler.normalize_address("HTTPS://Printer.Local:443/status")
        == "printer.local"
    )
    assert DeviceReconciler.normalize_address("fe80::1") == "fe80::1"


def test_reconciler_does_not_replace_stored_identity_with_an_address_match() -> None:
    config = SerialConfig.get_new()
    config.serial = "SERIAL-1"
    config.host = "10.0.0.7"

    assert not DeviceReconciler.same_device(
        config,
        DiscoveredDevice(host="10.0.0.7", hardware_id="SERIAL-2"),
    )
