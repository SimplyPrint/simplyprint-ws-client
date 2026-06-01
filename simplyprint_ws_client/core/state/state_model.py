"""``StateModel`` -- the change-tracking base for ``PrinterState`` and its tree.

The engine was promoted into the shared reactive-model layer
(:class:`simplyprint_ws_client.contrib.model.ReactiveModel`) so brand device
models and the library state model share one base + one annotation vocabulary.
``StateModel`` is that base under the name the state tree has always used; this
module re-exports it (and the ``Exclusive`` / ``Untracked`` field markers) so
every existing ``from .state_model import StateModel`` call site is unchanged.
"""

from simplyprint_ws_client.contrib.model.annotations import Exclusive, Untracked
from simplyprint_ws_client.contrib.model.reactive import ReactiveModel

__all__ = ["StateModel", "Exclusive", "Untracked"]

#: The state tree's change-tracking base is the shared reactive model.
StateModel = ReactiveModel
