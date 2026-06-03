"""Shared building blocks for the per-brand add-printer flows.

Every brand opens the same way: pick the model (so the right guide shows and,
where relevant, options gate), then enter the host -- prefilled when a discovered
device carried it in, and skipped once nothing is left to ask. These helpers
build those two steps from the generic step-kit, so a brand declares only its
model list. No brand name appears here: a brand passes its own model choices. The
state keys are ``device_type`` (a model string; ``""`` = "not sure / any") and
``host``.
"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import List, Optional, Sequence, Tuple

from simplyprint_ws_client.contrib.flow import FieldsStep
from pydantic import BaseModel, ConfigDict, Field

from simplyprint_ws_client.contrib.onboarding.model_images import model_image_url

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


def model_choices_from_enum(
    enum_cls, *, unknown: str = UNKNOWN_MODEL_LABEL
) -> List[ModelChoice]:
    """Model-picker options from a brand ``DeviceType`` enum (skipping ``Unknown``),
    with the trailing 'not sure' fallback appended. Carries a per-model image when the
    enum exposes ``get_image_url``. Only the neutral options cross into the flow; the
    brand enum stays in the brand package."""
    choices = [
        ModelChoice(
            value=member.value,
            label=member.get_name(),
            image=member.get_image_url() if hasattr(member, "get_image_url") else None,
        )
        for member in enum_cls
        if member.name != "Unknown"
    ]
    choices.append(ModelChoice(value="", label=unknown))
    return choices


def model_choices(
    rows: Sequence[Tuple], *, unknown: str = UNKNOWN_MODEL_LABEL
) -> List[ModelChoice]:
    """Model-picker options from an authored list (for brands with no model enum),
    with the trailing 'not sure' fallback appended. Each row is ``(value, label)`` or
    ``(value, label, model_id)``; the optional SimplyPrint ``model_id`` resolves the
    product photo the card grid shows (see :mod:`.model_images`)."""
    choices = [
        ModelChoice(
            value=row[0],
            label=row[1],
            image=model_image_url(row[2])
            if len(row) > 2 and row[2] is not None
            else None,
        )
        for row in rows
    ]
    choices.append(ModelChoice(value="", label=unknown))
    return choices


def identify_step(
    choices: Sequence[ModelChoice],
    *,
    label: str = "Which printer are you adding?",
    content: Optional[Sequence[str]] = None,
) -> FieldsStep:
    """The model picker: a single ``select`` over ``choices``. Skipped when the model
    is already known (a discovered/seeded ``device_type``), so it never interrupts a
    quick-start. Belongs in the flow's ``identify`` phase."""

    class IdentifyInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        device_type: str = Field(
            default="",
            title="Printer model",
            json_schema_extra={
                "ui": {
                    # ``select`` keeps the terminal picker a dropdown; the web app
                    # reads ``variant`` to render the same options as a card grid.
                    "type": "select",
                    "variant": "cards",
                    "options": [
                        {
                            "value": choice.value,
                            "label": choice.label,
                            **({"image": choice.image} if choice.image else {}),
                        }
                        for choice in choices
                    ],
                }
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
        input_model=IdentifyInput,
        include=lambda s: not s.get("device_type"),
    )


def host_step(
    *,
    label: str = "Find your printer",
    help_text: Optional[str] = None,
    footer=None,
) -> FieldsStep:
    """A one-field host form for brands whose only manual input is the IP address.

    ``help_text`` is the hint under the input (where to read the address off the
    printer). The host is prefilled when a discovered device seeded it (shown as a
    "detected" summary) and the step is skipped once the host is known -- so a
    fully-discovered printer reaches verify with no screen. ``footer`` is markdown
    (static or a callable of state) shown below the submit. Belongs in ``find``.
    """
    hint = help_text or "Your printer's address on the local network."

    class HostInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        host: IPv4Address | IPv6Address = Field(
            title="IP address",
            description=hint,
            json_schema_extra={"ui": {"placeholder": "192.168.1.42"}},
        )

    return FieldsStep(
        "find",
        label=label,
        input_model=HostInput,
        input_values=lambda state: {"host": state.get("host")},
        footer=footer,
        include=lambda s: not s.get("host"),
    )
