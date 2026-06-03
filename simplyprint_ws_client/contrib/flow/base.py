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
    FrozenSet,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
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

#: A predicate on accumulated flow state, gating whether a phase or step runs.
#: A step's predicate may be async (the engine awaits it); a phase's must be
#: synchronous, since the outline is resolved outside the engine's async loop.
Predicate = Callable[[Mapping[str, object]], Union[bool, Awaitable[bool]]]


# screen descriptors -- the neutral, data-driven frontend contract


@dataclass(frozen=True)
class Choice:
    """A labelled, selectable option.

    Used both for a select field's options and for a ``choice``/``discovery``
    screen's options. ``value`` is what the answer carries; ``label`` (and the
    optional ``description``/``icon``/``badge``) are presentation only, so a
    backend can serve a region list or a device list fully described.
    ``recommended`` lets a renderer mark the suggested default option (e.g. the
    recommended connection mode); presentation-only, the engine never reads it.
    """

    value: str
    label: str
    description: Optional[str] = None
    icon: Optional[str] = None
    badge: Optional[str] = None
    recommended: bool = False


@dataclass(frozen=True)
class Validation:
    """Declarative validation hints for a field.

    Mirrored to the client (so a superForm/zod schema can be built from them) and
    enforceable server-side. All optional; ``message`` overrides the default text.
    """

    min: Optional[float] = None
    max: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class StepAction:
    """A secondary action on a screen beside the primary submit.

    Resend a code, rescan the network, switch mode. The caller invokes it by
    ``id`` (no answer); the engine routes the ``id`` to the current step, which
    handles it (e.g. re-issue the code and re-offer the same screen).
    """

    id: str
    label: str
    kind: str = "button"


@dataclass(frozen=True)
class StepField:
    """One input a screen asks for, described so any renderer can draw it.

    ``field_type`` is a renderer hint (``text``/``email``/``password``/``number``/
    ``otp``/``toggle``/``textarea``/``select``); ``options`` (labelled) or
    ``choices`` (plain) turn it into a selection; ``validation`` carries
    declarative rules the client mirrors into its form schema. A flow never names
    a brand here -- only the field it needs.

    ``value`` + ``prefilled`` describe a field whose answer is *already known* (a
    fact a discovered/seeded device carried in): the renderer shows it collapsed
    in a "detected" summary with an edit affordance rather than as an open input,
    and submits ``value`` unless the user edits it. ``prefilled`` is the flag the
    UI keys on; ``value`` is the known answer.
    """

    key: str
    label: str
    field_type: str = "text"
    required: bool = True
    help_text: Optional[str] = None
    secret: bool = False
    choices: Optional[List[str]] = None
    options: Optional[List[Choice]] = None
    default: Optional[str] = None
    placeholder: Optional[str] = None
    validation: Optional[Validation] = None
    value: Optional[str] = None
    prefilled: bool = False


@dataclass(frozen=True)
class StepPrompt:
    """One screen the engine hands back for the caller to render and answer.

    Data-driven so the frontend renders generically. ``kind`` selects the
    renderer (``form`` fields, ``choice``/``discovery`` ``options``, ``poll``
    wait, ``review`` summary, ``info``); ``content`` is markdown blocks the UI
    shows above the inputs (instructions, FAQ accordions, callouts); ``fields``
    are inputs, ``options`` selectables, ``actions`` secondary buttons. ``poll``
    marks a wait-on-device screen the driver re-advances on a timer.
    """

    key: str
    label: str
    help_text: Optional[str] = None
    kind: str = "form"
    content: List[str] = field(default_factory=list)
    fields: List[StepField] = field(default_factory=list)
    options: List[Choice] = field(default_factory=list)
    actions: List[StepAction] = field(default_factory=list)
    #: Markdown blocks rendered *below* the inputs (after the submit button) --
    #: where a "where do I find this?" walkthrough or FAQ belongs, so it doesn't
    #: push the form down. ``content`` is the above-the-inputs counterpart.
    footer: List[str] = field(default_factory=list)
    poll: bool = False


@dataclass(frozen=True)
class Phase:
    """One stage of a flow, owning the steps that realize it plus its outline label.

    A flow is an ordered list of phases; each phase groups the :class:`Step` s that
    run while the user is in it and the ``label`` a progress stepper shows. The
    engine flattens phases into one step sequence to drive the cursor, but keeps the
    grouping so a UI knows, up front, every phase, its substeps, and which is active.

    ``include`` branches a flow: when its predicate is false for the current state
    the whole phase (all its steps) is skipped and it drops out of the resolved
    :func:`outline` -- so a LAN phase and a cloud phase coexist in one flow and only
    the chosen one runs and shows. It must be synchronous (a pure check on
    already-accumulated state). A phase may hold no steps (a terminal "Done" marker).
    """

    id: str
    label: str
    steps: Sequence["Step"] = ()
    include: Optional[Predicate] = None


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
class Ready:
    """Every step is done, but the outcome was not committed (``finalize=False``).

    Lets a two-phase caller seal state at the terminal boundary and commit later:
    a web ``verify`` endpoint advances to here without building/persisting the
    result, then ``setup-step`` resumes the same state with ``finalize=True`` to
    fold it. ``state`` resumes straight to the terminal -- no step re-runs.
    """

    state: FlowState


@dataclass(frozen=True)
class Failed:
    """A recoverable failure: show ``message``, re-offer ``prompt``, retry."""

    message: str
    prompt: Optional[StepPrompt]
    state: FlowState


FlowResult = Union[Prompt, Poll, "Done", "Ready", Failed]


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
    #: Human label for this substep, surfaced in a flow's resolved outline so a UI
    #: can name the steps within each phase and mark the active one.
    label: str = ""

    @abstractmethod
    async def run(
        self,
        state: Mapping[str, object],
        answer: Optional[Mapping[str, object]],
        action: Optional[str] = None,
    ) -> StepOutcome:
        """Advance this stage by one increment.

        ``answer`` is the caller's response to the screen this step last
        :class:`Ask` ed; ``action`` is a secondary action the caller invoked on
        that screen (e.g. ``resend``/``rescan``) with no answer. Both are routed
        only to the step at the cursor and consumed once. See the class docstring.
        """


@dataclass(frozen=True)
class Flow(Generic[T]):
    """A guided interaction: an ordered list of :class:`Phase` s plus a terminal fold.

    Declarative -- a brand builds one by grouping its steps into phases (each step
    capturing its own closures for discovery, probing, submitting) and a ``finish``
    that folds the accumulated :data:`FlowState` into the typed outcome ``T`` (a
    ``PrinterConfig``, a saved account, ...). The engine flattens the phases to drive
    the cursor; the flow is pure data.
    """

    id: str
    title: str
    phases: Sequence[Phase]
    finish: Callable[[Mapping[str, object]], Union[T, Awaitable[T]]]
    #: Human label of what the flow yields (for a UI; never load-bearing).
    produces: str = ""
    #: The state keys a *caller* may seed when starting the flow (the facts a
    #: discovered/deep-linked device carries in: a host, a serial, a model, a
    #: branch mode). A stateless front door intersects an untrusted launch context
    #: with this allowlist, so only what a flow opts into can pre-fill or skip a
    #: step -- a secret (an access code, a password) is never listed and so can
    #: never be seeded. Empty by default: a flow seeds nothing until it says so.
    seedable: FrozenSet[str] = frozenset()


def _walk(flow: "Flow") -> List[Tuple[Phase, Step]]:
    """Flatten a flow's phases into the ``(phase, step)`` sequence the cursor indexes."""
    return [(phase, step) for phase in flow.phases for step in phase.steps]


def outline(flow: "Flow", state: Optional[Mapping[str, object]] = None) -> List[Phase]:
    """The phases that apply to ``state``, in order -- a flow's resolved outline.

    Drops any phase whose ``include`` predicate is false for the accumulated state
    (the inactive side of a branch), so the result is exactly the path the user is
    on. With no branches every phase is returned. ``include`` is called
    synchronously (phase predicates are pure state checks).
    """
    work = dict(state or {})
    return [p for p in flow.phases if p.include is None or bool(p.include(work))]


def active_position(
    flow: "Flow", state: Optional[Mapping[str, object]] = None
) -> Optional[Tuple[Phase, Step]]:
    """The ``(phase, step)`` the cursor in ``state`` rests on, or ``None`` if exhausted.

    The active step is wherever the engine paused (an :class:`Ask`/:class:`Reject`),
    so a UI can mark the live phase and substep straight from the sealed state. Past
    the last step (the flow is done) there is no active position.
    """
    flat = _walk(flow)
    cursor = int(dict(state or {}).get(CURSOR_KEY, 0))
    return flat[cursor] if 0 <= cursor < len(flat) else None


async def advance_flow(
    flow: "Flow[T]",
    state: Optional[Mapping[str, object]] = None,
    answer: Optional[Mapping[str, object]] = None,
    *,
    action: Optional[str] = None,
    finalize: bool = True,
) -> "FlowResult":
    """Advance ``flow`` by one caller-visible increment and return the result.

    Resumes at the cursor recorded in ``state`` (step 0 on the first call),
    hands ``answer`` to that step, and runs forward through every step that
    :class:`Advance` s without asking -- including steps whose phase ``include`` is
    false, which skip silently -- so the caller always lands on the next real
    :class:`Prompt`/:class:`Poll`, the terminal :class:`Done`, or a recoverable
    :class:`Failed`, never an intermediate. ``state`` is treated as immutable; a
    fresh state dict rides on the result.

    When ``finalize`` is ``False`` and every step is exhausted, returns
    :class:`Ready` (state sealed at the terminal boundary) instead of folding the
    outcome -- the two-phase web bridge advances to here in ``verify`` and folds
    in ``setup-step``.
    """
    work: FlowState = dict(state or {})
    flat = _walk(flow)
    cursor = int(work.get(CURSOR_KEY, 0))
    pending = answer
    pending_action = action

    while cursor < len(flat):
        phase, step = flat[cursor]
        # A skipped branch's steps advance untouched, never seeing the answer.
        if phase.include is not None and not bool(phase.include(work)):
            cursor += 1
            work[CURSOR_KEY] = cursor
            continue

        outcome = await step.run(work, pending, pending_action)
        # An answer/action is consumed only by the step at the cursor.
        pending = None
        pending_action = None

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

    if not finalize:
        return Ready(work)

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
