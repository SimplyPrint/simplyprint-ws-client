"""The model-picker product-photo convention, in one place.

An integration's web app serves SimplyPrint's printer-model photos under
``/img/pimg/`` as ``{model_id}.webp``. The add-printer flow shows that photo on
the model-select card grid; each integration maps its own models to a SimplyPrint
``model_id`` and resolves the URL through here. A leaf module on purpose -- low
level enum modules import it without dragging in the flow layer, and the path
shape lives in exactly one spot.

No brand name appears here: the id is a plain integer an integration resolves for
itself.
"""

from __future__ import annotations

#: Shown when a chosen model has no photo (a model SimplyPrint's DB lacks, or a
#: brand whose boards aren't in it). The web app also falls back to this on a
#: broken-image error -- keep the two in sync.
DEFAULT_MODEL_IMAGE = "/img/pimg/default.webp"


def model_image_url(model_id: int) -> str:
    """Web path to the product photo for SimplyPrint ``model_id``."""
    return f"/img/pimg/{model_id}.webp"
