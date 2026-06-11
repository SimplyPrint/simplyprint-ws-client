"""Reusable printer-card presentation contract (see :mod:`.base`)."""

from simplyprint_ws_client.integration.presentation.base import (
    EditableField,
    EditableFieldOption,
    NAME_FIELD,
    WEBCAM_URL_FIELD,
    PrinterPresentation,
    default_printer_presentation,
    host_field,
    public_secret,
)

__all__ = [
    "EditableField",
    "EditableFieldOption",
    "NAME_FIELD",
    "WEBCAM_URL_FIELD",
    "PrinterPresentation",
    "default_printer_presentation",
    "host_field",
    "public_secret",
]
