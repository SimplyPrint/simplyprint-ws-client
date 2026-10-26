"""A layout entry's chain count, capped only where the unit declares a cap."""

from simplyprint_ws_client.core.state.models import MultiMaterialSolution
from simplyprint_ws_client.core.state.printer import MaterialLayoutEntry


def test_a_chainable_unit_without_a_cap_keeps_its_chains():
    entry = MaterialLayoutEntry(nozzle=0, mms=MultiMaterialSolution.BOXTURTLE, chains=3)
    assert entry.get_chains() == 3


def test_a_capped_unit_is_capped():
    entry = MaterialLayoutEntry(nozzle=0, mms=MultiMaterialSolution.BAMBU_AMS, chains=9)
    assert entry.get_chains() == 4
