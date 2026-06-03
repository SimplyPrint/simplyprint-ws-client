"""The generic guided-flow engine.

One resumable, UI-neutral step machine for every guided interaction an
integration drives -- adding a printer, signing in to a cloud account, adopting
a cloud device are all a :class:`Flow` of :class:`Step` s yielding a typed
outcome. The engine (:func:`advance_flow`/:func:`run_flow`) owns the cursor,
prompt/answer plumbing and resumability; a brand supplies only the steps and the
terminal fold. Continuation state is plain data, so a stateless web caller can
seal it and resume.

See :mod:`simplyprint_ws_client.contrib.flow.base` for the engine and
:mod:`simplyprint_ws_client.contrib.flow.steps` for the reusable step kit.
"""

from simplyprint_ws_client.contrib.flow.base import (
    CURSOR_KEY,
    Advance,
    Ask,
    Choice,
    Done,
    Failed,
    Flow,
    FlowError,
    FlowResult,
    FlowState,
    InputModel,
    InputValidationError,
    Phase,
    Poll,
    Prompt,
    PromptCallback,
    Ready,
    Reject,
    Step,
    StepAction,
    StepField,
    StepOutcome,
    StepPrompt,
    active_position,
    advance_flow,
    model_input_schema,
    outline,
    resolve,
    run_flow,
    validate_input,
)
from simplyprint_ws_client.contrib.flow.steps import (
    ActionStep,
    ChoiceStep,
    FieldsStep,
    SelectStep,
)

__all__ = [
    # engine
    "Flow",
    "Step",
    "advance_flow",
    "run_flow",
    "resolve",
    "FlowError",
    "InputValidationError",
    "FlowState",
    "CURSOR_KEY",
    "InputModel",
    # screen descriptors
    "StepField",
    "StepPrompt",
    "Phase",
    "outline",
    "active_position",
    "Choice",
    "StepAction",
    "model_input_schema",
    "validate_input",
    # step outcomes
    "Ask",
    "Advance",
    "Reject",
    "StepOutcome",
    # flow results
    "Prompt",
    "Poll",
    "Done",
    "Ready",
    "Failed",
    "FlowResult",
    "PromptCallback",
    # step kit
    "FieldsStep",
    "ChoiceStep",
    "SelectStep",
    "ActionStep",
]
