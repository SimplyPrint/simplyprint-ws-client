"""The generic guided-flow engine: one resumable, UI-neutral step machine.

Every guided interaction an integration drives -- adding a LAN printer, signing
in to a cloud account, adopting a cloud device -- is the same shape: emit a
prompt, take the caller's answer, accumulate non-secret state, repeat until a
typed outcome falls out. This module owns that shape once, as :class:`Flow` plus
the engine that advances it, and pushes every brand- or step-specific decision
into a :class:`Step`.

The engine never touches a terminal and never holds a live object between steps:
all continuation state is plain data in :data:`FlowState`, so a stateless caller
(a web request) can seal it, hand it back next call, and resume exactly where it
left off. The same flow runs in-process via :func:`run_flow` (CLI/TUI/tests) or
one increment at a time via :func:`advance_flow` (the web round-trip).

Two small vocabularies:

* A :class:`Step` returns a *step outcome* -- :class:`Ask` (need input),
  :class:`Advance` (satisfied, merge updates, move on) or :class:`Reject` (soft
  failure, re-offer the prompt). Hard failures raise :class:`FlowError`.
* Advancing the whole flow yields a *flow result* -- :class:`Prompt`,
  :class:`Poll`, :class:`Done` or :class:`Failed` -- the thing a driver renders.

The prompt descriptors (:class:`StepField`, :class:`StepPrompt`) are the neutral
contract a generic frontend renders; no flow ever describes a brand-specific
widget, only fields.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

T = TypeVar("T")

#: Accumulated, **non-secret** continuation state, carried across steps and
#: round-tripped by a stateless caller (sealed). Plain data only -- never a live
#: socket, api handle or token; a step reconstructs what it needs from these
#: facts each call, which is what makes a flow resumable.
FlowState = Dict[str, object]

#: The reserved key the engine uses to remember which step is current. Steps and
#: ``finish`` ignore it; it is non-secret continuation state like any other.
CURSOR_KEY = "__flow_cursor__"


# -- prompt descriptors (the neutral frontend contract) ----------------------


@dataclass(frozen=True)
class StepField:
    """One input a prompt asks for, described so any renderer can draw it.

    ``choices`` turns it into a selection; ``secret`` masks the echo;
    ``field_type`` is a renderer hint (``text``/``password``/``number``/...).
    A flow never names a brand here -- only the field it needs.
    """

    key: str
    label: str
    field_type: str = "text"
    required: bool = True
    help_text: Optional[str] = None
    secret: bool = False
    choices: Optional[List[str]] = None
    default: Optional[str] = None
    placeholder: Optional[str] = None


@dataclass(frozen=True)
class StepPrompt:
    """A request for input handed back to the caller to render and answer.

    A prompt is a labelled group of :class:`StepField` s -- one for a single
    input, several for a form. ``poll`` marks a prompt that needs no field, only
    that the user acts on the device (press Allow) before the step is retried;
    a driver shows ``label``/``help_text`` and re-advances on a timer.
    """

    key: str
    label: str
    help_text: Optional[str] = None
    fields: List[StepField] = field(default_factory=list)
    poll: bool = False


# -- step outcomes (what a Step.run returns) ---------------------------------


@dataclass(frozen=True)
class Ask:
    """This step needs the caller to answer ``prompt`` before it can proceed.

    The engine returns it to the caller as a :class:`Prompt` (or :class:`Poll`
    when ``prompt.poll``); the same step runs again next call with the answer.
    """

    prompt: StepPrompt
    retry_after: float = 2.0


@dataclass(frozen=True)
class Advance:
    """This step is satisfied: merge ``updates`` into the flow state, move on."""

    updates: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Reject:
    """This step failed but the caller can retry (a wrong code, a bad password).

    The engine returns a :class:`Failed`; the cursor stays put, so re-advancing
    with a fresh answer re-runs this step. ``prompt`` re-offers the input.
    """

    message: str
    prompt: Optional[StepPrompt] = None


StepOutcome = Union[Ask, Advance, Reject]


# -- flow results (what advancing the whole flow yields) ---------------------


@dataclass(frozen=True)
class Prompt:
    """The flow needs input: render ``prompt``, answer it, advance with ``state``."""

    prompt: StepPrompt
    state: FlowState


@dataclass(frozen=True)
class Poll:
    """The flow is waiting on an out-of-band action: re-advance after a delay."""

    prompt: StepPrompt
    state: FlowState
    retry_after: float = 2.0


@dataclass(frozen=True)
class Done(Generic[T]):
    """The flow finished and produced its outcome ``value``."""

    value: T


@dataclass(frozen=True)
class Failed:
    """A recoverable failure: show ``message``, re-offer ``prompt``, retry."""

    message: str
    prompt: Optional[StepPrompt]
    state: FlowState


FlowResult = Union[Prompt, Poll, "Done", Failed]


class FlowError(RuntimeError):
    """A hard failure that aborts the flow (host unreachable, account gone).

    A neutral error type so a driver can tell a flow failure from an arbitrary
    runtime error without importing a brand exception. Steps raise it for
    unrecoverable conditions; use :class:`Reject` for anything the user can fix
    by answering again.
    """


async def resolve(value):
    """Await ``value`` if awaitable, else return it -- so a step, a ``finish``
    or a driver callback may be written sync or async without the engine caring."""
    if inspect.isawaitable(value):
        return await value
    return value


# -- the Step contract -------------------------------------------------------


class Step(ABC):
    """One stage of a flow. Subclass for bespoke logic; or compose the kit in
    :mod:`simplyprint_ws_client.contrib.flow.steps`.

    A step is asked to :meth:`run` with the accumulated ``state`` and the
    ``answer`` to the prompt it last :class:`Ask` ed (``None`` before it has
    asked, or when it asks nothing). It returns a :class:`StepOutcome`. The
    engine owns the cursor, answer routing and resumability; the step owns only
    *what this stage does*.
    """

    #: Stable identifier for the step, used in prompts and for debugging.
    key: str = ""

    @abstractmethod
    async def run(
        self, state: Mapping[str, object], answer: Optional[Mapping[str, object]]
    ) -> StepOutcome:
        """Advance this stage by one increment. See the class docstring."""


@dataclass(frozen=True)
class Flow(Generic[T]):
    """A guided interaction: an ordered list of steps plus a terminal fold.

    Declarative -- a brand builds one by listing steps (each capturing its own
    closures for discovery, probing, submitting) and a ``finish`` that folds the
    accumulated :data:`FlowState` into the typed outcome ``T`` (a
    ``PrinterConfig``, a saved account, ...). The engine drives it; the flow is
    pure data.
    """

    id: str
    title: str
    steps: Sequence[Step]
    finish: Callable[[Mapping[str, object]], Union[T, Awaitable[T]]]
    #: Human label of what the flow yields (for a UI; never load-bearing).
    produces: str = ""


# -- the engine --------------------------------------------------------------


async def advance_flow(
    flow: "Flow[T]",
    state: Optional[Mapping[str, object]] = None,
    answer: Optional[Mapping[str, object]] = None,
) -> "FlowResult":
    """Advance ``flow`` by one caller-visible increment and return the result.

    Resumes at the cursor recorded in ``state`` (step 0 on the first call),
    hands ``answer`` to that step, and runs forward through every step that
    :class:`Advance` s without asking -- so the caller always lands on the next
    real :class:`Prompt`/:class:`Poll`, the terminal :class:`Done`, or a
    recoverable :class:`Failed`, never an intermediate. ``state`` is treated as
    immutable; a fresh state dict rides on the result.
    """
    work: FlowState = dict(state or {})
    cursor = int(work.get(CURSOR_KEY, 0))
    pending = answer

    while cursor < len(flow.steps):
        step = flow.steps[cursor]
        outcome = await step.run(work, pending)
        pending = None  # an answer is consumed only by the step that asked it

        if isinstance(outcome, Ask):
            work[CURSOR_KEY] = cursor
            if outcome.prompt.poll:
                return Poll(outcome.prompt, work, outcome.retry_after)
            return Prompt(outcome.prompt, work)

        if isinstance(outcome, Reject):
            work[CURSOR_KEY] = cursor
            return Failed(outcome.message, outcome.prompt, work)

        if isinstance(outcome, Advance):
            if outcome.updates:
                work.update(outcome.updates)
            cursor += 1
            work[CURSOR_KEY] = cursor
            continue

        raise FlowError(
            f"step {step.key!r} returned {type(outcome).__name__}, not a StepOutcome"
        )

    value = await resolve(flow.finish(work))
    return Done(value)


#: A driver's prompt handler: given the prompt to answer (and a non-empty message
#: when re-asking after a :class:`Failed`), return the answer(s). May be async.
PromptCallback = Callable[
    [StepPrompt, Optional[str]],
    Union[
        str,
        Mapping[str, object],
        None,
        Awaitable[Union[str, Mapping[str, object], None]],
    ],
]


async def run_flow(
    flow: "Flow[T]",
    *,
    on_prompt: Optional[PromptCallback] = None,
    on_poll: Optional[
        Callable[[StepPrompt, float], Union[None, Awaitable[None]]]
    ] = None,
    initial_state: Optional[Mapping[str, object]] = None,
    max_iterations: int = 128,
) -> T:
    """Drive ``flow`` to completion in-process and return its outcome.

    The synchronous-caller counterpart to :func:`advance_flow`: loops advancing
    the flow, asking ``on_prompt`` to answer each :class:`Prompt` (and each
    recoverable :class:`Failed`) and waiting via ``on_poll`` on each
    :class:`Poll`. ``initial_state`` seeds facts known up front (a host typed on
    the command line). Raises :class:`FlowError` if input is needed but no
    ``on_prompt`` was given, or if the flow does not terminate.
    """
    state: FlowState = dict(initial_state or {})
    pending: Optional[Mapping[str, object]] = None

    for _ in range(max_iterations):
        result = await advance_flow(flow, state, pending)
        pending = None

        if isinstance(result, Done):
            return result.value

        state = result.state

        if isinstance(result, Poll):
            if on_poll is not None:
                await resolve(on_poll(result.prompt, result.retry_after))
            continue

        # Prompt or Failed: both need the caller to (re-)answer a prompt.
        if on_prompt is None:
            raise FlowError(
                result.message
                if isinstance(result, Failed)
                else "flow requires input but no on_prompt callback was provided"
            )
        message = result.message if isinstance(result, Failed) else None
        answer = await resolve(on_prompt(result.prompt, message))
        pending = _as_mapping(answer, result.prompt)

    raise FlowError(f"flow {flow.id!r} did not terminate within {max_iterations} steps")


def _as_mapping(
    answer: Union[str, Mapping[str, object], None], prompt: StepPrompt
) -> Optional[Mapping[str, object]]:
    """Normalise a driver's answer to the ``{field_key: value}`` a step reads.

    A bare scalar answers a single-field prompt by that field's key (or the
    prompt key); a mapping is passed through; ``None`` stays ``None``.
    """
    if answer is None or isinstance(answer, Mapping):
        return answer
    key = prompt.fields[0].key if prompt.fields else prompt.key
    return {key: answer}
