"""Neutral printer-card presentation contract for the public API.

The shape a ``/printers`` API returns for one printer is identical across
integrations -- an image, a connection summary, badges, secrets, and a set of
user-editable fields. Each :class:`EditableField` is built for one concrete
config and carries a current value plus the domain operation that writes it.
The generic serializer/PATCH path therefore needs no knowledge of config model
internals. The integration projects its spec into one
:class:`PrinterPresentation`; this module stays brand-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from simplyprint_ws_client._compat import StrEnum

if TYPE_CHECKING:
    from simplyprint_ws_client.core.config import PrinterConfig

EditableValue = str | int | float | bool | None
EditableWriter = Callable[[EditableValue], None]


class EditableFieldType(StrEnum):
    STRING = "string"
    URL = "url"
    EMAIL = "email"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    TEXTAREA = "textarea"


@dataclass(frozen=True)
class EditableFieldOption:
    """One option entry for a ``select``-type EditableField."""

    value: str
    label: str


@dataclass(frozen=True)
class EditableField:
    """One config-bound user-editable printer setting.

    Carries the full spec the client needs to render the control (label, type,
    validation, options for selects) -- the API surface is a contract, not a
    pseudo-custom-fields hack. ``value`` is the snapshot rendered by this
    presentation and ``write`` is the explicit domain operation used by PATCH.
    """

    key: str  # neutral API key, e.g. "host", "webcam_url"
    label: str
    value: EditableValue
    write: EditableWriter
    type: EditableFieldType = EditableFieldType.STRING
    required: bool = False
    placeholder: str | None = None
    description: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None
    options: tuple[EditableFieldOption, ...] = ()
    # A rarely-changed/optional field a renderer may tuck behind an "advanced"
    # disclosure so the common fields aren't crowded out. Presentation-only.
    advanced: bool = False


def webcam_url_field(config: "PrinterConfig") -> EditableField:
    """Build the universal webcam override field for one concrete config."""

    return EditableField(
        key="webcam_url",
        label="Webcam URL",
        value=config.custom_webcam_url,
        write=config.set_webcam_url,
        type=EditableFieldType.URL,
        placeholder="http://192.168.1.42:8080/?action=stream",
        description="Leave blank to use the printer's own camera.",
        advanced=True,
    )


def host_field(
    value: str | None,
    write: EditableWriter,
    *,
    label: str = "IP address",
    placeholder: str = "192.168.1.42",
    description: str = "The printer's address on your network. "
    "Changing it reconnects the printer.",
) -> EditableField:
    """Build a config-bound device-address field under the neutral ``host`` key."""

    return EditableField(
        key="host",
        label=label,
        value=value,
        write=write,
        placeholder=placeholder,
        description=description,
    )


@dataclass(frozen=True)
class PrinterPresentation:
    image_url: str | None = None
    model_name: str | None = None
    connection: dict | None = None
    badges: list[dict] = field(default_factory=list)
    secrets: list[dict] = field(default_factory=list)
    private_fields: tuple[str, ...] = ()
    editable_fields: tuple[EditableField, ...] = ()


def _as_text(value, default: str = "") -> str:
    if value is None:
        return default

    return str(value)


def public_secret(key: str, label: str, value) -> dict | None:
    value = _as_text(value)
    if not value:
        return None

    return {"key": key, "label": label, "value": value}


def default_printer_presentation(
    image_url: str, config: "PrinterConfig"
) -> PrinterPresentation:
    return PrinterPresentation(
        image_url=image_url,
        editable_fields=(webcam_url_field(config),),
    )
