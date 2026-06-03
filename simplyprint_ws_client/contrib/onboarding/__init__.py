"""Reusable add-printer onboarding building blocks.

The model-picker product-photo convention (:mod:`.model_images`) and the
generic add-printer flow steps (:mod:`.steps`) every integration composes: pick
the model, then enter the host. An integration declares only its model list; no
brand name lives here.
"""

from simplyprint_ws_client.contrib.onboarding.model_images import (
    DEFAULT_MODEL_IMAGE,
    model_image_url,
)
from simplyprint_ws_client.contrib.onboarding.steps import (
    UNKNOWN_MODEL_LABEL,
    ModelChoice,
    host_step,
    identify_step,
    model_choices,
    model_choices_from_enum,
)

__all__ = [
    "DEFAULT_MODEL_IMAGE",
    "model_image_url",
    "UNKNOWN_MODEL_LABEL",
    "ModelChoice",
    "host_step",
    "identify_step",
    "model_choices",
    "model_choices_from_enum",
]
