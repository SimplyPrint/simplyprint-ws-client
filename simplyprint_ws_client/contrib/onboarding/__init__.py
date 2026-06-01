"""Brand-agnostic printer onboarding API.

Owns the neutral ``discover -> select -> verify -> multi-stage setup ->
create_config`` pipeline (:func:`onboard_printer`) and the value objects + hooks
ABC (:class:`PrinterOnboardingHooks`) each brand integration plugs into. The
shared layer never touches a terminal: interactive input is collected through
injected callbacks, so a side-car CLI or web wizard can supply the real I/O.

See :mod:`simplyprint_ws_client.contrib.onboarding.base` for the contract.
"""

from simplyprint_ws_client.contrib.onboarding.base import (
    DiscoveredDevice,
    OnboardingError,
    PrinterOnboardingHooks,
    SetupStepResult,
    StepField,
    StepInputCallback,
    StepPrompt,
    VerificationResult,
    onboard_printer,
)

__all__ = [
    "DiscoveredDevice",
    "VerificationResult",
    "SetupStepResult",
    "StepField",
    "StepPrompt",
    "PrinterOnboardingHooks",
    "OnboardingError",
    "StepInputCallback",
    "onboard_printer",
]
