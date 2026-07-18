"""Reusable printer-card presentation contract (see :mod:`.base`)."""

from simplyprint_ws_client.integration.presentation.base import (
    EditableField,
    EditableFieldOption,
    EditableFieldType,
    EditableValue,
    EditableWriter,
    PrinterPresentation,
    default_printer_presentation,
    host_field,
    public_secret,
    webcam_url_field,
)

__all__ = [
    "EditableField",
    "EditableFieldOption",
    "EditableFieldType",
    "EditableValue",
    "EditableWriter",
    "PrinterPresentation",
    "default_printer_presentation",
    "host_field",
    "public_secret",
    "webcam_url_field",
]
