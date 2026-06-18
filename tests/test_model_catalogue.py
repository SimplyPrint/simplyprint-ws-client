"""ModelCatalogue + PrinterSpec.model_catalogue() contract pins.

The model catalogue is the brand-agnostic seam between "what model is this
printer?" and the surfaces that consume the answer (presentation, discovery,
onboarding). These tests pin:

* the base hook defaults to ``None`` and ``provides`` flips for an overrider
  (mirrors the ``mdns_spec`` pin in ``test_discovery_mdns.py``)
* :meth:`model_aware_presentation` enriches the base presentation from
  ``config.device_type`` via the catalogue, and falls through to the plain
  default when no catalogue is set
* :class:`RowModelCatalogue` and :class:`EnumModelCatalogue` round-trip:
  every choice value resolves back through ``label`` / ``image_url``, and
  ``resolve`` maps device-reported strings to known values conservatively
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pytest

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.integration.model_catalogue import (
    EnumModelCatalogue,
    ModelCatalogue,
    RowModelCatalogue,
)
from simplyprint_ws_client.integration.presentation import PrinterPresentation
from simplyprint_ws_client.integration.spec import PrinterSpec, ProductMetadata

pytestmark = pytest.mark.contract


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


class _Config(PrinterConfig):
    device_type: Optional[str] = None


class _NoCatalogueSpec(PrinterSpec):
    KEY = "nocat"
    metadata = ProductMetadata(
        display_name="NoCat",
        image_url="/img/nocat.png",
        supported_transports=("http",),
        capabilities=(),
    )

    @classmethod
    def build(cls):
        return cls(
            key=cls.KEY,
            client_factory=lambda config, **kw: None,
            config_factory=_Config,
        )


class _RowCatalogueSpec(_NoCatalogueSpec):
    KEY = "rowcat"
    metadata = ProductMetadata(
        display_name="RowCat",
        image_url="/img/rowcat.png",
        supported_transports=("http",),
        capabilities=(),
    )

    @classmethod
    def model_catalogue(cls) -> Optional[ModelCatalogue]:
        return RowModelCatalogue(
            [
                ("row_a", "RowCat A", 100),
                ("row_b", "RowCat B", None),
            ],
            brand_prefix="RowCat",
        )


# A tiny fake enum exercising the EnumModelCatalogue path: members carry a
# ``get_name()`` (the universal method the catalogue reads), an ``Unknown``
# sentinel, and a brand-specific ``is_pro_series`` (which the catalogue does
# NOT read -- it stays on the enum for brand-internal use).


class _FakeDeviceType(Enum):
    Pro = "PRO"
    Lite = "LITE"
    Unknown = "unknown"

    def get_name(self) -> str:
        return {
            _FakeDeviceType.Pro: "Pro Model",
            _FakeDeviceType.Lite: "Lite Model",
        }.get(self, self.value)

    def is_pro_series(self) -> bool:
        return self is _FakeDeviceType.Pro


class _EnumCatalogueSpec(_NoCatalogueSpec):
    KEY = "enumcat"
    metadata = ProductMetadata(
        display_name="EnumCat",
        image_url="/img/enumcat.png",
        supported_transports=("http",),
        capabilities=(),
    )

    @classmethod
    def model_catalogue(cls) -> Optional[ModelCatalogue]:
        return EnumModelCatalogue(
            _FakeDeviceType,
            {_FakeDeviceType.Pro: 200, _FakeDeviceType.Lite: 201},
            unknown_value="unknown",
        )


# --------------------------------------------------------------------------- #
# Hook default + provides()
# --------------------------------------------------------------------------- #


class TestHookDefault:
    def test_base_returns_none(self):
        assert PrinterSpec.model_catalogue() is None

    def test_provides_false_for_base(self):
        assert _NoCatalogueSpec.provides("model_catalogue") is False

    def test_provides_true_for_overrider(self):
        assert _RowCatalogueSpec.provides("model_catalogue") is True
        assert _EnumCatalogueSpec.provides("model_catalogue") is True

    def test_overrider_returns_catalogue_instance(self):
        assert isinstance(_RowCatalogueSpec.model_catalogue(), ModelCatalogue)
        assert isinstance(_EnumCatalogueSpec.model_catalogue(), ModelCatalogue)


# --------------------------------------------------------------------------- #
# model_aware_presentation
# --------------------------------------------------------------------------- #


def _config(device_type: Optional[str] = None) -> "_Config":
    return _Config(id=1, token="t", device_type=device_type)


class TestModelAwarePresentation:
    def test_no_catalogue_falls_through_to_default(self):
        config = _config("anything")
        pres = _NoCatalogueSpec.model_aware_presentation(config)
        assert pres.image_url == "/img/nocat.png"
        assert pres.model_name is None

    def test_catalogue_with_unset_device_type_keeps_default(self):
        config = _config(None)
        pres = _RowCatalogueSpec.model_aware_presentation(config)
        assert pres.image_url == "/img/rowcat.png"
        assert pres.model_name is None

    def test_catalogue_with_unknown_value_keeps_default(self):
        config = _config("")
        pres = _RowCatalogueSpec.model_aware_presentation(config)
        assert pres.image_url == "/img/rowcat.png"
        assert pres.model_name is None

    def test_row_catalogue_resolves_model_name_and_image(self):
        config = _config("row_a")
        pres = _RowCatalogueSpec.model_aware_presentation(config)
        assert pres.model_name == "A"
        assert pres.image_url == "/img/pimg/100.webp"

    def test_row_catalogue_unknown_slug_falls_back_to_brand_image(self):
        config = _config("not-a-slug")
        pres = _RowCatalogueSpec.model_aware_presentation(config)
        assert pres.image_url == "/img/rowcat.png"
        assert pres.model_name is None

    def test_enum_catalogue_resolves_model_name_and_image(self):
        config = _config("PRO")
        pres = _EnumCatalogueSpec.model_aware_presentation(config)
        assert pres.model_name == "Pro Model"
        assert pres.image_url == "/img/pimg/200.webp"

    def test_enum_catalogue_unknown_value_keeps_default(self):
        config = _config("unknown")
        pres = _EnumCatalogueSpec.model_aware_presentation(config)
        assert pres.image_url == "/img/enumcat.png"
        assert pres.model_name is None

    def test_returns_printer_presentation_instance(self):
        config = _config("row_a")
        pres = _RowCatalogueSpec.model_aware_presentation(config)
        assert isinstance(pres, PrinterPresentation)


# --------------------------------------------------------------------------- #
# RowModelCatalogue round-trip
# --------------------------------------------------------------------------- #


class TestRowModelCatalogue:
    @pytest.fixture
    def cat(self):
        return RowModelCatalogue(
            [
                ("centauri", "Elegoo Centauri", 591),
                ("centauri_2", "Elegoo Centauri 2", None),
            ],
            brand_prefix="Elegoo",
            aliases={"centauri2": "centauri_2"},
        )

    def test_choices_lists_known_models_plus_unknown(self, cat):
        choices = cat.choices().choices
        values = [c.value for c in choices]
        assert "centauri" in values
        assert "centauri_2" in values
        assert "" in values  # trailing unknown

    def test_label_round_trips_for_every_known_choice(self, cat):
        for choice in cat.choices().choices:
            if choice.value == "":
                continue  # the trailing unknown sentinel
            assert cat.label(choice.value) == choice.label

    def test_label_none_for_unknown_slug(self, cat):
        assert cat.label("nope") is None
        assert cat.label(None) is None
        assert cat.label("") is None

    def test_image_url_resolves_when_sp_model_id_present(self, cat):
        assert cat.image_url("centauri") == "/img/pimg/591.webp"

    def test_image_url_none_when_sp_model_id_absent(self, cat):
        assert cat.image_url("centauri_2") is None

    def test_image_url_none_for_unknown_slug(self, cat):
        assert cat.image_url("nope") is None
        assert cat.image_url(None) is None

    def test_model_name_strips_brand_prefix(self, cat):
        assert cat.model_name("centauri") == "Centauri"

    def test_model_name_none_without_brand_prefix_config(self):
        cat = RowModelCatalogue([("x", "X Model", 1)])
        assert cat.model_name("x") == "X Model"

    def test_resolve_by_full_label(self, cat):
        assert cat.resolve("Elegoo Centauri") == "centauri"

    def test_resolve_by_stripped_label(self, cat):
        assert cat.resolve("Centauri") == "centauri"

    def test_resolve_by_alias(self, cat):
        assert cat.resolve("centauri2") == "centauri_2"

    def test_resolve_none_for_no_match(self, cat):
        assert cat.resolve("not a model") is None
        assert cat.resolve(None) is None
        assert cat.resolve("") is None

    def test_resolve_normalises_whitespace_and_case(self, cat):
        assert cat.resolve("  elegoo   centauri  ") == "centauri"


# --------------------------------------------------------------------------- #
# EnumModelCatalogue round-trip
# --------------------------------------------------------------------------- #


class TestEnumModelCatalogue:
    @pytest.fixture
    def cat(self):
        return EnumModelCatalogue(
            _FakeDeviceType,
            {_FakeDeviceType.Pro: 200, _FakeDeviceType.Lite: 201},
            unknown_value="unknown",
        )

    def test_choices_exclude_unknown_member(self, cat):
        choices = cat.choices().choices
        values = [c.value for c in choices]
        assert "PRO" in values
        assert "LITE" in values
        assert "unknown" not in values
        assert "" in values  # trailing unknown

    def test_label_round_trips_for_every_known_choice(self, cat):
        for choice in cat.choices().choices:
            if choice.value == "":
                continue
            assert cat.label(choice.value) == choice.label

    def test_label_none_for_unknown_value(self, cat):
        assert cat.label("unknown") is None
        assert cat.label(None) is None
        assert cat.label("nope") is None

    def test_image_url_resolves_when_sp_model_id_present(self, cat):
        assert cat.image_url("PRO") == "/img/pimg/200.webp"

    def test_image_url_none_for_unknown_value(self, cat):
        assert cat.image_url("unknown") is None
        assert cat.image_url(None) is None

    def test_resolve_via_from_model_id(self, cat):
        # _FakeDeviceType has no from_model_id; resolve falls back to cls(value)
        assert cat.resolve("PRO") == "PRO"
        assert cat.resolve("LITE") == "LITE"

    def test_resolve_none_for_unknown(self, cat):
        assert cat.resolve("nope") is None
        assert cat.resolve(None) is None

    def test_resolve_none_for_unknown_sentinel(self, cat):
        assert cat.resolve("unknown") is None

    def test_custom_resolver_is_used_when_supplied(self):
        calls = []

        def resolver(s):
            calls.append(s)
            return _FakeDeviceType.Pro if s == "pro-code" else None

        cat = EnumModelCatalogue(
            _FakeDeviceType,
            {_FakeDeviceType.Pro: 200},
            resolver=resolver,
            unknown_value="unknown",
        )
        assert cat.resolve("pro-code") == "PRO"
        assert cat.resolve("nope") is None
        assert calls == ["pro-code", "nope"]

    def test_unknown_value_class_default_is_empty_string(self):
        # Row brands default to "" (the picker's trailing unknown value);
        # EnumModelCatalogue overrides with the enum's Unknown .value.
        assert RowModelCatalogue([("x", "X", 1)]).unknown_value == ""
        assert EnumModelCatalogue(
            _FakeDeviceType, {}, unknown_value="unknown"
        ).unknown_value == "unknown"


# --------------------------------------------------------------------------- #
# ModelCatalogue ABC cannot be instantiated directly
# --------------------------------------------------------------------------- #


class TestAbcContract:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ModelCatalogue()

    def test_subclass_must_implement_all_four(self):
        class _Incomplete(ModelCatalogue):
            def choices(self):
                return None

        with pytest.raises(TypeError):
            _Incomplete()
