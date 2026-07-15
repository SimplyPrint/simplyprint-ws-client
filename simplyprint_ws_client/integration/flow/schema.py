"""The Pydantic <-> JSON-Schema bridge for flow prompts.

A prompt's answer contract is a Pydantic model: :func:`model_input_schema`
derives the JSON Schema a frontend turns into its native validator,
:func:`fields_from_schema` derives the neutral render fields from that schema,
and :func:`validate_input` validates an answer server-side, raising the
recoverable :class:`InputValidationError` so a step can re-offer its prompt.
Self-contained: it knows the screen descriptors in :mod:`.base`, never the
engine.
"""

from __future__ import annotations

import copy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from pydantic import BaseModel, ValidationError

from simplyprint_ws_client.integration.flow.base import Choice, FieldType, StepField

__all__ = [
    "InputModel",
    "InputValidationError",
    "fields_from_schema",
    "model_input_schema",
    "validate_input",
]

InputModel = Type[BaseModel]


class InputValidationError(ValueError):
    """A recoverable prompt-answer validation failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def model_input_schema(
    model: InputModel,
    *,
    values: Optional[Mapping[str, object]] = None,
    prefilled: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """The JSON Schema a prompt exposes for the answer object it accepts."""
    schema = copy.deepcopy(model.model_json_schema(mode="validation"))
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema

    field_order = {key: index for index, key in enumerate(model.model_fields)}
    for key, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        ui = prop.get("ui")
        if not isinstance(ui, dict):
            ui = {}
            prop["ui"] = ui
        if key in field_order:
            ui["order"] = field_order[key]

    if not values and not prefilled:
        return schema

    prefilled_keys = set(prefilled or ())
    for key, value in (values or {}).items():
        if value is None:
            continue
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        wire_value = str(value)
        prop["default"] = wire_value
        ui = prop.setdefault("ui", {})
        if isinstance(ui, dict):
            ui["value"] = wire_value
            if key in prefilled_keys:
                ui["prefilled"] = True
    return schema


def fields_from_schema(schema: Mapping[str, Any]) -> List[StepField]:
    """Derive neutral render fields from a Pydantic JSON Schema."""
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    required = set(schema.get("required") or ())
    fields: List[StepField] = []
    for key, raw in sorted(properties.items(), key=_schema_field_order):
        if not isinstance(raw, Mapping):
            continue
        prop = dict(raw)
        ui = prop.get("ui") if isinstance(prop.get("ui"), Mapping) else {}
        options = _schema_options(prop, ui)
        field_type = str(ui.get("type") or _schema_field_type(prop, options))
        help_text = ui.get("help_text") or prop.get("description")
        default = prop.get("default")
        value = ui.get("value")
        fields.append(
            StepField(
                key=str(key),
                label=str(ui.get("label") or prop.get("title") or key),
                field_type=field_type,
                required=str(key) in required,
                help_text=str(help_text) if help_text is not None else None,
                secret=bool(ui.get("secret")) or field_type in {"password", "secret"},
                choices=[str(option.value) for option in options] or None,
                options=options or None,
                default=str(default) if default is not None else None,
                placeholder=(
                    str(ui.get("placeholder"))
                    if ui.get("placeholder") is not None
                    else None
                ),
                value=str(value) if value is not None else None,
                prefilled=bool(ui.get("prefilled")),
            )
        )
    return fields


def _schema_field_order(item: Tuple[str, Any]) -> Tuple[int, int]:
    raw = item[1]
    if isinstance(raw, Mapping):
        ui = raw.get("ui")
        if isinstance(ui, Mapping):
            order = ui.get("order")
            if isinstance(order, int) and not isinstance(order, bool):
                return (0, order)
            if isinstance(order, str):
                try:
                    return (0, int(order))
                except ValueError:
                    pass
    return (1, 0)


def _schema_options(prop: Mapping[str, Any], ui: Mapping[str, Any]) -> List[Choice]:
    raw_options = ui.get("options") or []
    options: List[Choice] = []
    for item in raw_options:
        if isinstance(item, Mapping):
            value = str(item.get("value", ""))
            options.append(Choice(value=value, label=str(item.get("label") or value)))
        else:
            value = str(item)
            options.append(Choice(value=value, label=value))
    if options:
        return options

    raw_enum = prop.get("enum") or []
    return [Choice(value=str(value), label=str(value)) for value in raw_enum]


def _schema_field_type(prop: Mapping[str, Any], options: Sequence[Choice]) -> FieldType:
    if options:
        return "select"
    if prop.get("format") == "email":
        return "email"
    if prop.get("format") in {"uri", "url"}:
        return "url"
    if prop.get("type") in {"number", "integer"}:
        return "number"
    if prop.get("type") == "boolean":
        return "toggle"
    return "text"


def validate_input(
    model: InputModel,
    answer: Mapping[str, object],
    *,
    fields: Sequence[StepField] = (),
) -> Dict[str, object]:
    """Validate a prompt answer with Pydantic and return JSON-safe updates.

    Raises :class:`InputValidationError` with a human message when the answer is
    malformed, so a step can re-offer the same prompt as a recoverable failure.
    """
    try:
        parsed = model.model_validate(dict(answer))
    except ValidationError as exc:
        raise InputValidationError(_validation_message(exc, fields))
    return dict(parsed.model_dump(mode="json", exclude_none=True))


def _validation_message(exc: ValidationError, fields: Sequence[StepField]) -> str:
    labels = {field.key: field.label for field in fields}
    messages: List[str] = []
    for error in exc.errors():
        loc_parts = [str(part) for part in error.get("loc", ())]
        key = loc_parts[0] if loc_parts else "input"
        label = labels.get(key, key)
        message = str(error.get("msg") or "Invalid value")
        entry = f"{label}: {message}" if label else message
        if entry not in messages:
            messages.append(entry)
    return "; ".join(messages) or "Please check the values and try again."
