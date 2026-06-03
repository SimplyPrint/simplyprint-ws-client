import hashlib
from dataclasses import dataclass
from typing import Optional

from simplyprint_ws_client import PrinterConfig


@dataclass
class _HwConfig(PrinterConfig):
    """A config whose stable hardware id is its serial (test double)."""

    serial: Optional[str] = None

    def stable_hardware_id(self) -> Optional[str]:
        return self.serial


def test_config_fields():
    config1 = PrinterConfig.get_blank()
    config2 = PrinterConfig.get_blank()

    assert config1.partial_eq(**config1.as_dict())
    assert config2.partial_eq(**config2.as_dict())

    assert config1.is_empty()
    assert config2.is_default()

    assert config1.as_dict() == config2.as_dict()

    assert config1.as_dict() == {
        "id": 0,
        "token": "0",
        "name": None,
        "in_setup": None,
        "short_id": None,
        "unique_id": config1.unique_id,
        "public_ip": None,
    }

    config1.id = 1

    assert not config1.partial_eq(**config2.as_dict())
    assert not config2.partial_eq(**config1.as_dict())

    assert not config1.is_empty()

    assert not config1.is_pending()

    config2.token = "super_cool_token"
    config2.unique_id = "super_cool_id"

    assert not config2.is_empty()
    assert not config2.is_default()

    assert config2.partial_eq(unique_id="super_cool_id")
    assert config2.partial_eq(token="super_cool_token")

    config3 = PrinterConfig(
        **{
            "id": None,
            "in_setup": None,
            "short_id": None,
            "name": None,
            "public_ip": None,
            "token": None,
            "unique_id": None,
        }
    )
    assert config3.is_empty()

    config4 = PrinterConfig(
        **{
            "id": None,
            "in_setup": None,
            "short_id": None,
            "name": None,
            "public_ip": None,
            "token": None,
            "unique_id": "140686326013968",
        }
    )
    assert not config4.is_empty()


def test_stable_hardware_id_default_is_none():
    """The base hook has no hardware id; brands override it."""
    assert PrinterConfig.get_blank().stable_hardware_id() is None


def test_derive_unique_id_is_salted_sha1_of_hardware_id():
    cfg = _HwConfig.get_blank()
    cfg.serial = "SN-123"

    expected = hashlib.sha1(b"saltvalue:SN-123").hexdigest()
    # Pins the exact formula: sha1(salt + ":" + hardware_id).
    assert cfg.derive_unique_id("saltvalue") == expected
    # Deterministic for the same (salt, device).
    assert cfg.derive_unique_id("saltvalue") == expected
    # Installation-scoped: a different salt derives a different id.
    assert cfg.derive_unique_id("other-salt") != expected
    # Device-scoped: a different hardware id derives a different id.
    cfg.serial = "SN-999"
    assert cfg.derive_unique_id("saltvalue") != expected


def test_derive_unique_id_is_none_without_hardware_id():
    assert _HwConfig.get_blank().derive_unique_id("salt") is None
    assert PrinterConfig.get_blank().derive_unique_id("salt") is None


def test_ensure_unique_id_prefers_derived_while_pending():
    cfg = _HwConfig.get_blank()
    cfg.serial = "SN-123"

    assigned = cfg.ensure_unique_id("salt")
    assert assigned == hashlib.sha1(b"salt:SN-123").hexdigest()
    assert cfg.unique_id == assigned


def test_ensure_unique_id_replaces_placeholder_while_pending():
    # get_new mints a random placeholder; while the printer is still in setup
    # (id == 0) the seam swaps it for the stable hardware-derived id.
    cfg = _HwConfig.get_new()
    placeholder = cfg.unique_id
    assert placeholder
    cfg.serial = "SN-123"

    derived = cfg.ensure_unique_id("salt")
    assert derived == hashlib.sha1(b"salt:SN-123").hexdigest()
    assert derived != placeholder


def test_ensure_unique_id_never_rekeys_a_registered_printer():
    cfg = _HwConfig.get_new()
    cfg.serial = "SN-123"
    cfg.id = 42  # backend-assigned -> registered, no longer pending

    existing = cfg.unique_id
    # Even with a stable hardware id available, a registered printer keeps its
    # id (re-keying would orphan backend correlation and logs).
    assert cfg.ensure_unique_id("salt") == existing
    assert cfg.unique_id == existing


def test_ensure_unique_id_falls_back_to_random_without_hardware_id():
    cfg = PrinterConfig.get_blank()
    assert cfg.unique_id is None

    assigned = cfg.ensure_unique_id("salt")
    assert assigned and assigned == cfg.unique_id
    # No hardware id to derive from, so the assigned id is a random fallback.
    assert cfg.derive_unique_id("salt") is None
    # Idempotent while pending with no hardware id: it keeps the random id.
    assert cfg.ensure_unique_id("salt") == assigned
