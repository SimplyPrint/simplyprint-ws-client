"""Neutral printer-card presentation contract for the public API.

The shape a ``/printers`` API returns for one printer is identical across
integrations -- an image, a connection summary, badges, secrets, and a set of
user-editable fields. Only *which* fields are editable and how they map onto a
brand's config differs, so an :class:`EditableField` carries the one place a
brand attribute name lives (the generic serializer/PATCH never names it). The
integration's own registry projects each client type's spec into one
:class:`PrinterPresentation`; this module stays brand-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EditableFieldOption:
    """One option entry for a ``select``-type EditableField."""

    value: str
    label: str


@dataclass(frozen=True)
class EditableField:
    """A user-editable printer setting, mapping a neutral API key to the brand's
    own config attribute. This is the ONLY place brand attribute names for
    editable settings live, so the generic serializer/PATCH never name them.

    Carries the full spec the client needs to render the control (label, type,
    validation, options for selects) -- the API surface is a contract, not a
    pseudo-custom-fields hack."""

    key: str  # neutral API key, e.g. "name", "webcam_url"
    config_attr: str  # brand config attribute, e.g. "name", "custom_webcam_url"
    label: str
    type: str = "string"  # one of: string|url|email|number|boolean|select|textarea
    required: bool = False
    placeholder: str | None = None
    description: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None
    options: tuple[EditableFieldOption, ...] = ()


#: Every config has a friendly `name`.
NAME_FIELD = EditableField(
    key="name",
    config_attr="name",
    label="Name",
    placeholder="My printer",
    description="Shown across the dashboard and on SimplyPrint.",
    max_length=64,
)


@dataclass(frozen=True)
class PrinterPresentation:
    image_url: str | None = None
    model_name: str | None = None
    connection: dict | None = None
    badges: list[dict] = field(default_factory=list)
    secrets: list[dict] = field(default_factory=list)
    private_fields: tuple[str, ...] = ()
    editable_fields: tuple[EditableField, ...] = (NAME_FIELD,)


def _as_text(value, default: str = "") -> str:
    if value is None:
        return default

    return str(value)


def public_secret(key: str, label: str, value) -> dict | None:
    value = _as_text(value)
    if not value:
        return None

    return {"key": key, "label": label, "value": value}


def default_printer_presentation(image_url: str) -> PrinterPresentation:
    return PrinterPresentation(image_url=image_url)
