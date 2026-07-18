"""Per-type model catalogue: the four universal operations every surface
(presentation, discovery, onboarding) needs, keyed by the string value stored
in a config's ``device_type`` field.

This is the brand-agnostic seam between "what model is this printer?" and the
surfaces that consume the answer -- the printer-card presentation, the
discovery candidate dict, and the onboarding model picker. Each integration
exposes one concrete :class:`ModelCatalogue` through
:attr:`simplyprint_ws_client.integration.spec.IntegrationSpec.model_catalogue`;
the field defaults to ``None`` (no catalogue / single-model), so
surfaces simply skip a ``None`` result.

Two generic, reusable implementations ship here:

* :class:`RowModelCatalogue` -- authored ``(slug, label, sp_model_id)`` rows.
  Covers integrations that keep no model enum in code (the natural shape for
  brands whose model list is a small authored table).
* :class:`EnumModelCatalogue` -- wraps a brand's existing ``DeviceType``-style
  enum. The enum keeps its brand-specific methods (capability gating, nozzle
  maps, MMS defaults); only the four universal operations delegate here, so the
  catalogue is the one place presentation/discovery/onboarding read model
  identity.

The :class:`~simplyprint_ws_client.integration.flow.onboarding_steps.ModelChoiceCatalog`
(the onboarding picker) is what :meth:`ModelCatalogue.choices` returns -- this
module deliberately reuses the existing picker type rather than inventing a
parallel one, so the flow engine sees the same option type whether the
catalogue came from rows or an enum.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional, Sequence, Tuple

from simplyprint_ws_client.integration.flow.onboarding_steps import (
    UNKNOWN_MODEL_LABEL,
    ModelChoiceCatalog,
)


class ModelCatalogue(ABC):
    """Brand-agnostic model catalogue: the four universal operations every
    surface (presentation, discovery, onboarding) needs, keyed by the string
    value stored in a config's ``device_type`` field.

    A "value" is the persisted string slug (an enum ``.value`` for enum brands,
    a authored slug for row brands). The catalogue is the only place a surface
    turns a value into a human label, a product photo URL, an onboarding picker,
    or back from a device-reported string to a known slug. Brand-specific
    capability logic (feature gating, nozzle maps, peripherals) stays off this
    type -- it lives on the brand's own enum/capability table, keyed by the
    value the catalogue resolved.
    """

    #: The value that means "model not set / unknown". ``""`` for row brands
    #: (the picker's trailing unknown option), the brand enum's ``Unknown``
    #: ``.value`` for enum brands. Surfaces compare
    #: ``value == catalogue.unknown_value`` rather than ``value`` being falsy,
    #: so a slug of ``"0"`` or an enum value of ``"unknown"`` is honoured.
    unknown_value: ClassVar[str] = ""

    @abstractmethod
    def choices(self) -> ModelChoiceCatalog:
        """The onboarding picker for this type's models (excludes the unknown
        sentinel; :class:`ModelChoiceCatalog` appends its own trailing unknown
        option)."""

    @abstractmethod
    def label(self, value: Optional[str]) -> Optional[str]:
        """Human-readable model label for a persisted value, or ``None`` if the
        value is unset/unknown. The full canonical label (including any brand
        prefix) -- the string the SP ``firmware.machine_name`` and the onboarding
        picker carry."""

    @abstractmethod
    def image_url(self, value: Optional[str]) -> Optional[str]:
        """Product-photo URL for a persisted value, or ``None`` if the value is
        unset/unknown/has no artwork. Surfaces fall back to the brand image when
        this returns ``None``."""

    @abstractmethod
    def resolve(self, device_string: Optional[str]) -> Optional[str]:
        """Map a device-reported model string (firmware ``MachineName``, mDNS
        ``variant``, SSDP ``dev_name`` ...) to a known catalogue value, or
        ``None`` if it does not match. Conservative by contract: returning
        ``None`` never clobbers a good user pick -- the caller decides whether
        to keep the existing value."""

    def model_name(self, value: Optional[str]) -> Optional[str]:
        """Brand-less model name for the printers-tab card, or ``None`` if the
        value is unset/unknown. The API serializer composes the card name as
        ``"{brand} {model_name}"``, so this strips any brand prefix the
        :meth:`label` carries. The default returns :meth:`label` unchanged
        (enum brands' ``get_name`` is already brand-less); row brands override
        to strip their brand prefix."""
        return self.label(value)


#: The shape of one authored row: ``(slug, label, sp_model_id)``. ``sp_model_id``
#: is the SimplyPrint model id used to resolve the product photo; ``None`` when
#: it isn't known yet (the picker then falls back to the generic printer image).
Row = Tuple[str, str, Optional[int]]


class RowModelCatalogue(ModelCatalogue):
    """A :class:`ModelCatalogue` backed by authored ``(slug, label, sp_model_id)``
    rows -- the natural shape for integrations that keep no model enum in code.

    The third row element is the SimplyPrint model id, used only to resolve the
    product photo; ``None`` means no artwork yet (the picker falls back to the
    generic printer image). ``brand_prefix`` is stripped from ``label`` when
    computing the printers-tab card name (the API serializer composes
    ``"{brand} {model_name}"``, so a doubled brand prefix is avoided).
    ``aliases`` maps compact firmware identifiers to known slugs (covers mDNS
    TXT ``type`` values that are not the human label).
    """

    def __init__(
        self,
        rows: Sequence[Row],
        *,
        brand_prefix: Optional[str] = None,
        aliases: Optional[dict[str, str]] = None,
        image_url_for: Optional[Callable[[int], str]] = None,
        unknown: str = UNKNOWN_MODEL_LABEL,
    ) -> None:
        self._rows: Tuple[Row, ...] = tuple(rows)
        self._by_slug = {row[0]: row for row in self._rows}
        self._brand_prefix = brand_prefix
        self._aliases = dict(aliases) if aliases else {}
        self._image_url_for = image_url_for or _default_image_url
        self._unknown_label = unknown

    def choices(self) -> ModelChoiceCatalog:
        # ModelChoiceCatalog.from_rows expects the third element to be a URL
        # string (or None), but our rows carry the raw SimplyPrint model id
        # (an int). Resolve each id to its product-photo URL here so the picker
        # gets usable image URLs, not raw ids.
        resolved_rows = [
            (
                row[0],
                row[1],
                self._image_url_for(row[2]) if row[2] is not None else None,
            )
            for row in self._rows
        ]
        return ModelChoiceCatalog.from_rows(resolved_rows, unknown=self._unknown_label)

    def label(self, value: Optional[str]) -> Optional[str]:
        row = self._by_slug.get(value or "")
        return row[1] if row else None

    def model_name(self, value: Optional[str]) -> Optional[str]:
        """Brand-less model name for the printers-tab card (strips the brand
        prefix). The API serializer composes ``"{brand} {model_name}"`` so a
        doubled brand is avoided. ``None`` if the value is unknown."""
        full = self.label(value)
        if not full or not self._brand_prefix:
            return full
        return full.removeprefix(self._brand_prefix + " ")

    def image_url(self, value: Optional[str]) -> Optional[str]:
        row = self._by_slug.get(value or "")
        if not row or row[2] is None:
            return None
        return self._image_url_for(row[2])

    def resolve(self, device_string: Optional[str]) -> Optional[str]:
        if not device_string:
            return None
        alias = self._aliases.get(_compact(device_string))
        if alias:
            return alias
        target = _norm(device_string)
        for slug, label, _ in self._rows:
            full = _norm(label)
            stripped = _norm(label.split(" ", 1)[1]) if " " in label else full
            if target == full or target == stripped:
                return slug
        return None


class EnumModelCatalogue(ModelCatalogue):
    """A :class:`ModelCatalogue` backed by a brand's existing ``DeviceType``-style
    enum. The enum keeps its brand-specific methods (capability gating, nozzle
    maps, MMS defaults, file roots); only the four universal operations delegate
    here, so the catalogue is the one place presentation/discovery/onboarding
    read model identity.

    ``sp_model_ids`` maps each enum member to its SimplyPrint model id (the
    product photo resolver). ``resolver`` maps a device-reported string to an
    enum member and ``label`` maps a member to its human name. Both functions
    are explicit integration dependencies: the catalogue never guesses method
    names or falls back to enum construction. ``unknown_value`` is the enum's
    unknown member value so the ``is it set?`` check is uniform across row and
    enum brands.
    """

    def __init__(
        self,
        enum_cls,
        sp_model_ids: dict,
        *,
        resolver: Callable[[Optional[str]], object | None],
        label: Callable[[object], str],
        unknown_value: Optional[str] = None,
        image_url_for: Optional[Callable[[int], str]] = None,
        unknown: str = UNKNOWN_MODEL_LABEL,
    ) -> None:
        self._enum_cls = enum_cls
        self._sp_model_ids = dict(sp_model_ids)
        self._resolver = resolver
        self._label = label
        self._image_url_for = image_url_for or _default_image_url
        self._unknown_label = unknown
        if unknown_value is not None:
            self.unknown_value = unknown_value

    def _member(self, value: Optional[str]):
        if not value:
            return None
        try:
            return self._enum_cls(value)
        except ValueError:
            return None

    def choices(self) -> ModelChoiceCatalog:
        return ModelChoiceCatalog.from_items(
            (
                member
                for member in self._enum_cls
                if not _is_unknown_value(member.value, self.unknown_value)
            ),
            value=lambda member: str(member.value),
            label=self._label,
            image=lambda member: self._member_image(member),
            unknown=self._unknown_label,
        )

    def label(self, value: Optional[str]) -> Optional[str]:
        member = self._member(value)
        if member is None or _is_unknown_value(member.value, self.unknown_value):
            return None
        return self._label(member)

    def image_url(self, value: Optional[str]) -> Optional[str]:
        member = self._member(value)
        return self._member_image(member) if member is not None else None

    def _member_image(self, member) -> Optional[str]:
        model_id = self._sp_model_ids.get(member)
        return self._image_url_for(model_id) if model_id is not None else None

    def resolve(self, device_string: Optional[str]) -> Optional[str]:
        member = self._resolver(device_string)
        if member is None:
            return None
        if _is_unknown_value(member.value, self.unknown_value):
            return None
        return str(member.value)


# The product-photo URL scheme is owned by the integration (it knows its web
# path layout). The library catalogue resolves the *id*; the integration
# supplies the URL builder via the ``image_url_for`` constructor param. This
# default produces the conventional ``/img/pimg/<id>.webp`` shape so the
# catalogue is self-contained for tests and third-party use; the SimplyPrint
# app passes its own ``model_image_url`` to stay consistent with its assets.


def _default_image_url(model_id: int) -> str:
    return f"/img/pimg/{model_id}.webp"


def _is_unknown_value(value, unknown_value: str) -> bool:
    return str(value) == unknown_value


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9+]+", "", text.casefold())
