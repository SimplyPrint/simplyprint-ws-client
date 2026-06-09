import uuid
from typing import Optional

from simplyprint_ws_client import PrinterConfig


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
        "mac": None,
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
    """The base hook has no brand hardware id; brands override it. The neutral
    ``mac`` fallback is applied by the matching seam, not here."""
    assert PrinterConfig.get_blank().stable_hardware_id() is None
    assert _HwConfig.get_blank().stable_hardware_id() is None


def test_get_new_mints_a_stable_unique_id_uuid():
    # unique_id is the slot reference: a proper UUID, minted once at creation and
    # then stable (it is stored, not re-derived). get_blank leaves it unset so an
    # empty placeholder config stays empty.
    cfg = _HwConfig.get_new()
    assert uuid.UUID(cfg.unique_id)  # a valid UUID
    assert cfg.unique_id == cfg.unique_id  # stable
    assert PrinterConfig.get_blank().unique_id is None


def test_unique_id_is_independent_of_the_hardware_id():
    # Two printers that happen to share a serial still get distinct slot ids
    # (the slot id is not derived from the hardware id).
    a, b = _HwConfig.get_new(), _HwConfig.get_new()
    a.serial = b.serial = "SN-123"
    assert a.unique_id != b.unique_id


def test_mac_is_the_neutral_hardware_fallback_field():
    # The MAC is stored separately from the slot id, for matching a re-discovered
    # device when the brand exposes no serial/guid.
    cfg = PrinterConfig.get_blank()
    assert cfg.mac is None
    cfg.mac = "aa:bb:cc:dd:ee:ff"
    assert cfg.as_dict()["mac"] == "aa:bb:cc:dd:ee:ff"
