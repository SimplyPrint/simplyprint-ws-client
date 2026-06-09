"""Shared building blocks for add-printer onboarding flows.

These sit on top of the generic :mod:`~simplyprint_ws_client.contrib.flow` engine:
they are factories that build add-printer :class:`FieldsStep` s (a model picker, a
manual-address field). The library owns neutral step construction only --
integrations adapt their own model catalogues, image URLs, state keys, and address
field requirements before passing plain :class:`ModelChoice` objects into here.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from pydantic import ConfigDict, Field, create_model

from simplyprint_ws_client.contrib.flow.steps import FieldsStep

#: The trailing "I don't know my model" option every picker offers, so a user is
#: never blocked and an unknown model falls back to the generic guide.
UNKNOWN_MODEL_LABEL = "I'm not sure / other model"


@dataclass(frozen=True)
class ModelChoice:
    """One option in a model picker: the stored ``value`` and its ``label``, plus an
    optional ``image`` (a product photo URL) the web app shows on the card grid.

    This is the picker's own option type rather than the library ``Choice`` because
    a model has a photo; brands without artwork simply leave ``image`` ``None`` and
    the card falls back to a labelled tile. No brand name lives here -- the image is
    a plain URL a brand resolves for itself."""

    value: str
    label: str
    image: Optional[str] = None


@dataclass(frozen=True)
class ModelChoiceCatalog:
    """Owns a model-picker catalogue and builds its onboarding step."""

    choices: Tuple[ModelChoice, ...]

    @classmethod
    def from_rows(
        cls, rows: Sequence[Tuple], *, unknown: str = UNKNOWN_MODEL_LABEL
    ) -> "ModelChoiceCatalog":
        choices = [
            ModelChoice(
                value=str(row[0]),
                label=str(row[1]),
                image=str(row[2]) if len(row) > 2 and row[2] is not None else None,
            )
            for row in rows
        ]
        choices.append(ModelChoice(value="", label=unknown))
        return cls(tuple(choices))

    @classmethod
    def from_items(
        cls,
        items,
        *,
        value: Callable[[Any], str],
        label: Callable[[Any], str],
        image: Optional[Callable[[Any], Optional[str]]] = None,
        include: Optional[Callable[[Any], bool]] = None,
        unknown: str = UNKNOWN_MODEL_LABEL,
    ) -> "ModelChoiceCatalog":
        choices = [
            ModelChoice(
                value=str(value(item)),
                label=str(label(item)),
                image=image(item) if image is not None else None,
            )
            for item in items
            if include is None or include(item)
        ]
        choices.append(ModelChoice(value="", label=unknown))
        return cls(tuple(choices))

    def identify_step(
        self,
        *,
        state_key: str = "device_type",
        label: str = "Which printer are you adding?",
        field_title: str = "Printer model",
        content: Optional[Sequence[str]] = None,
    ) -> FieldsStep:
        options = [
            {
                "value": choice.value,
                "label": choice.label,
                **({"image": choice.image} if choice.image else {}),
            }
            for choice in self.choices
        ]
        input_model = create_model(
            "IdentifyInput",
            __config__=ConfigDict(extra="forbid"),
            **{
                state_key: (
                    str,
                    Field(
                        default="",
                        title=field_title,
                        json_schema_extra={
                            "ui": {
                                "type": "select",
                                "variant": "cards",
                                "options": options,
                            }
                        },
                    ),
                )
            },
        )

        return FieldsStep(
            "identify",
            label=label,
            content=(
                list(content)
                if content is not None
                else ["Pick your model so we can show the right setup steps."]
            ),
            input_model=input_model,
            include=lambda s: not s.get(state_key),
        )


@dataclass(frozen=True)
class ManualAddressStep:
    """Owns a one-field manual address step."""

    state_key: str = "host"
    value_type: Any = Union[IPv4Address, IPv6Address]
    step_id: str = "find"
    label: str = "Find your printer"
    field_title: str = "IP address"
    help_text: Optional[str] = None
    placeholder: Optional[str] = None
    footer: Any = None

    def build(self) -> FieldsStep:
        hint = self.help_text or "Your printer's address on the local network."
        ui = {"placeholder": self.placeholder} if self.placeholder else {}
        input_model = create_model(
            "ManualAddressInput",
            __config__=ConfigDict(extra="forbid"),
            **{
                self.state_key: (
                    self.value_type,
                    Field(
                        title=self.field_title,
                        description=hint,
                        json_schema_extra={"ui": ui} if ui else None,
                    ),
                )
            },
        )

        return FieldsStep(
            self.step_id,
            label=self.label,
            input_model=input_model,
            input_values=lambda state: {self.state_key: state.get(self.state_key)},
            footer=self.footer,
            include=lambda s: not s.get(self.state_key),
        )
