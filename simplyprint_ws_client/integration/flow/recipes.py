"""Composed flow recipes -- the skeletons every integration was rebuilding.

The flow *engine* (:mod:`.base`, :mod:`.steps`) is healthy and generic; what
five integrations duplicated was the add-printer skeleton one rung above it:
``identify -> find (address + verify) -> [brand phases] -> done``. The composer
below owns that skeleton once; a brand supplies only its genuinely-different
parts -- the model catalogue, the probe, the config fold, and any extra phases
(pairing, passwords) in between.

Phase ids (``identify`` / ``find`` / ``done``) are part of the web/onboarding
contract and stay byte-stable.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from simplyprint_ws_client.integration.flow import (
    ActionStep,
    Flow,
    FlowState,
    ManualAddressStep,
    Phase,
)


def standard_add_printer_flow(
    *,
    title: str,
    make_config: Callable[[FlowState], object],
    models=None,
    identify_label: Optional[str] = None,
    verify: Optional[Callable] = None,
    address_help: Optional[str] = None,
    address_placeholder: Optional[str] = None,
    address_footer: Optional[List[str]] = None,
    extra_phases: Iterable[Phase] = (),
    seedable: frozenset = frozenset({"host", "name"}),
    flow_id: str = "add-printer",
    produces: str = "printer",
) -> Flow:
    """The standard LAN add-printer flow: identify the model, find the printer
    by address (with an optional verification probe), run any brand phases
    (pairing, credentials), done.

    ``make_config`` folds the verified flow state into a fresh config (the
    flow's ``finish``). ``models`` is a ``ModelChoiceCatalog`` (omitted = no
    identify phase). ``verify`` is an ``async (state, answer) -> mapping`` probe
    that raises ``FlowError`` when the host is not this brand's printer.
    """
    phases: List[Phase] = []

    if models is not None:
        phases.append(
            Phase(
                "identify",
                "Model",
                steps=[models.identify_step(label=identify_label)],
            )
        )

    find_steps = [
        ManualAddressStep(
            help_text=address_help,
            footer=address_footer,
            placeholder=address_placeholder,
        ).build()
    ]
    if verify is not None:
        find_steps.append(
            ActionStep("verify", verify, label="Verify", show_in_outline=False)
        )
    phases.append(Phase("find", "Find printer", steps=find_steps))

    phases.extend(extra_phases)
    phases.append(Phase("done", "Done"))

    return Flow(
        id=flow_id,
        title=title,
        produces=produces,
        seedable=seedable,
        phases=phases,
        finish=make_config,
    )
