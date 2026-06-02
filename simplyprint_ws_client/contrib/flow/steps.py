"""The reusable step kit -- compose a flow from these instead of hand-writing a
state machine.

Three primitives cover the common shapes; subclass :class:`Step` directly when a
stage is genuinely bespoke:

* :class:`FieldsStep` -- collect a form once, advance with the answers.
* :class:`SelectStep` -- discover/list candidates and settle on one.
* :class:`ActionStep` -- run an async action that owns its own control flow
  (probe, log in, verify a code, persist). The general escape hatch: its action
  may itself ask, advance or reject.

Every step takes an optional ``include`` predicate that skips the stage when it
returns false, so a flow can carry conditional steps (ask the serial only when
discovery didn't supply one; run the code-challenge only when login returned
one) without the engine knowing the condition.
"""

from __future__ import annotations

from typing import (
    Awaitable,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .base import (
    Advance,
    Ask,
    FlowError,
    Reject,
    Step,
    StepField,
    StepOutcome,
    StepPrompt,
    resolve,
)

#: A predicate on flow state, sync or async, gating whether a step runs.
Include = Callable[[Mapping[str, object]], Union[bool, Awaitable[bool]]]


async def _included(include: Optional[Include], state: Mapping[str, object]) -> bool:
    return include is None or bool(await resolve(include(state)))


class FieldsStep(Step):
    """Collect a set of inputs once, then advance with them merged into state.

    ``fields`` is a static list or a callable of state -- so a step can drop a
    field it already knows (ask the serial only when discovery didn't supply
    one). An answer missing a required field re-offers the prompt.
    """

    def __init__(
        self,
        key: str,
        *,
        label: str,
        fields: Union[
            Sequence[StepField], Callable[[Mapping[str, object]], Sequence[StepField]]
        ],
        help_text: Optional[str] = None,
        include: Optional[Include] = None,
    ) -> None:
        self.key = key
        self._label = label
        self._fields = fields
        self._help = help_text
        self._include = include

    def _resolve_fields(self, state: Mapping[str, object]) -> List[StepField]:
        fields = self._fields(state) if callable(self._fields) else self._fields
        return list(fields)

    def _prompt(self, fields: Sequence[StepField]) -> StepPrompt:
        return StepPrompt(
            key=self.key, label=self._label, help_text=self._help, fields=list(fields)
        )

    async def run(
        self, state: Mapping[str, object], answer: Optional[Mapping[str, object]]
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()

        fields = self._resolve_fields(state)

        if answer is None:
            return Ask(self._prompt(fields))

        missing = [f.key for f in fields if f.required and not answer.get(f.key)]
        if missing:
            return Reject("Please fill in: " + ", ".join(missing), self._prompt(fields))

        updates = {f.key: answer[f.key] for f in fields if f.key in answer}
        return Advance(updates)


class SelectStep(Step):
    """Discover or list candidates and settle on exactly one.

    ``source(state) -> [item]`` produces candidates (a LAN scan, an account's
    devices); ``option(item) -> (value, label)`` renders each as a choice;
    ``pick(state, item) -> updates`` folds the chosen item's facts into state.
    Auto-advances when one candidate is found, or when ``skip_when`` says the
    choice is already made (a host typed up front). With no candidates it
    advances untouched -- leaving a later step to collect input manually -- so a
    brand without discovery is just a flow whose first selectable step is empty.

    ``source`` may be re-run on the answer call, so back it with a cached
    snapshot for an expensive scan rather than re-scanning.
    """

    def __init__(
        self,
        key: str,
        *,
        source: Callable[
            [Mapping[str, object]], Union[Sequence[object], Awaitable[Sequence[object]]]
        ],
        option: Callable[[object], Tuple[str, str]],
        pick: Callable[
            [Mapping[str, object], object],
            Union[Mapping[str, object], Awaitable[Mapping[str, object]]],
        ],
        label: str,
        help_text: Optional[str] = None,
        auto: bool = True,
        skip_when: Optional[Include] = None,
        include: Optional[Include] = None,
    ) -> None:
        self.key = key
        self._source = source
        self._option = option
        self._pick = pick
        self._label = label
        self._help = help_text
        self._auto = auto
        self._skip_when = skip_when
        self._include = include

    def _prompt(self, items: Sequence[object]) -> StepPrompt:
        choices = [self._option(item)[1] for item in items]
        return StepPrompt(
            key=self.key,
            label=self._label,
            help_text=self._help,
            fields=[StepField(key=self.key, label=self._label, choices=choices)],
        )

    def _match(self, items: Sequence[object], chosen: object) -> Optional[object]:
        for item in items:
            _value, label = self._option(item)
            if str(label) == str(chosen):
                return item
        return None

    async def run(
        self, state: Mapping[str, object], answer: Optional[Mapping[str, object]]
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()
        if self._skip_when is not None and await resolve(self._skip_when(state)):
            return Advance()

        items = list(await resolve(self._source(state)))

        if answer is not None:
            chosen = answer.get(self.key)
            item = self._match(items, chosen)
            if item is None:
                return Reject("That option is no longer available", self._prompt(items))
            return Advance(await resolve(self._pick(state, item)))

        if not items:
            return Advance()
        if len(items) == 1 and self._auto:
            return Advance(await resolve(self._pick(state, items[0])))
        return Ask(self._prompt(items))


class ActionStep(Step):
    """Run an async action that owns its own control flow.

    The general escape hatch for logic that is neither a plain form nor a
    selection: probe a host, attempt a login (ask credentials, then submit),
    verify a code, persist a result. ``action(state, answer)`` returns a
    :class:`StepOutcome` itself -- so it may :class:`Ask` then, on the next call,
    :class:`Advance` or :class:`Reject` -- or returns a plain mapping/``None`` as
    shorthand for :class:`Advance`. Raise :class:`FlowError` for a hard failure.
    """

    def __init__(
        self,
        key: str,
        action: Callable[
            [Mapping[str, object], Optional[Mapping[str, object]]],
            Union[StepOutcome, Mapping[str, object], None, Awaitable],
        ],
        *,
        include: Optional[Include] = None,
    ) -> None:
        self.key = key
        self._action = action
        self._include = include

    async def run(
        self, state: Mapping[str, object], answer: Optional[Mapping[str, object]]
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()

        result = await resolve(self._action(state, answer))

        if isinstance(result, (Ask, Advance, Reject)):
            return result
        if result is None:
            return Advance()
        if isinstance(result, Mapping):
            return Advance(result)
        raise FlowError(
            f"action {self.key!r} returned {type(result).__name__}, "
            "expected a StepOutcome, mapping or None"
        )
