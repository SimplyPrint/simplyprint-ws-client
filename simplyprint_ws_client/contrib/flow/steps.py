"""The reusable step kit -- compose a flow from these instead of hand-writing a
state machine.

* :class:`FieldsStep`   -- collect a form (with markdown content) once, advance.
* :class:`ChoiceStep`   -- a branch point: pick one of a fixed set of options; the
  chosen value is merged into state so later steps ``include`` on it.
* :class:`SelectStep`   -- discover/list candidates (live scan + optional manual
  entry) and settle on one.
* :class:`ActionStep`   -- run an async action that owns its own control flow
  (probe, log in, verify a code, persist), optionally handling secondary screen
  actions (``resend``) via ``on_action``.

Every step takes an optional ``include`` predicate that skips the stage when it
returns false -- so a flow branches by writing a value (a ChoiceStep) and gating
later steps on it (``include=lambda s: s.get("mode") == "lan"``) without the
engine knowing the condition.
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
    Choice,
    FlowError,
    InputModel,
    InputValidationError,
    Reject,
    Step,
    StepAction,
    StepField,
    StepOutcome,
    StepPrompt,
    model_input_schema,
    resolve,
    validate_input,
)

#: A predicate on flow state gating whether a step runs. Unlike a phase's (which
#: must be sync), a step's may be async -- the engine awaits it.
Include = Callable[[Mapping[str, object]], Union[bool, Awaitable[bool]]]

#: Markdown blocks shown above a screen's inputs -- a static list, or a callable of
#: state so a step can tailor the copy to what's known (show only the guide for the
#: picked model). Sync or async.
Content = Union[
    Sequence[str], Callable[[Mapping[str, object]], Union[Sequence[str], Awaitable]]
]


async def _included(include: Optional[Include], state: Mapping[str, object]) -> bool:
    return include is None or bool(await resolve(include(state)))


async def _resolve_content(content: Content, state: Mapping[str, object]) -> List[str]:
    """Resolve a step's ``content`` against state: call it if it's a callable,
    else use the list as-is. Always returns a fresh list."""
    if callable(content):
        content = await resolve(content(state))
    return list(content or [])


class FieldsStep(Step):
    """Collect a set of inputs once (with optional markdown ``content`` shown
    above them), then advance with them merged into state.

    ``fields`` is a static list or a callable of state -- so a step can drop a
    field it already knows. An answer missing a required field re-offers the
    screen.
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
        content: Optional[Content] = None,
        footer: Optional[Content] = None,
        actions: Optional[Sequence[StepAction]] = None,
        kind: str = "form",
        input_model: Optional[InputModel] = None,
        include: Optional[Include] = None,
        show_in_outline: bool = True,
    ) -> None:
        self.key = key
        self.label = label
        self.show_in_outline = show_in_outline
        self._fields = fields
        self._help = help_text
        self._content = content or []
        self._footer = footer or []
        self._actions = list(actions or [])
        self._kind = kind
        self._input_model = input_model
        self._include = include

    def _resolve_fields(self, state: Mapping[str, object]) -> List[StepField]:
        fields = self._fields(state) if callable(self._fields) else self._fields
        return list(fields)

    def _prompt(
        self,
        fields: Sequence[StepField],
        content: Sequence[str],
        footer: Sequence[str],
    ) -> StepPrompt:
        return StepPrompt(
            key=self.key,
            label=self.label,
            help_text=self._help,
            kind=self._kind,
            content=list(content),
            fields=list(fields),
            input_schema=(
                model_input_schema(self._input_model)
                if self._input_model is not None
                else None
            ),
            actions=list(self._actions),
            footer=list(footer),
        )

    async def run(
        self,
        state: Mapping[str, object],
        answer: Optional[Mapping[str, object]],
        action: Optional[str] = None,
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()

        fields = self._resolve_fields(state)
        content = await _resolve_content(self._content, state)
        footer = await _resolve_content(self._footer, state)

        if answer is None:
            return Ask(self._prompt(fields, content, footer))

        # A prefilled field carries its known value, so a renderer may collapse it
        # into a summary and not re-submit it; fall back to that value when the
        # answer omits the field (the user didn't edit it).
        def effective(f: StepField):
            if f.key in answer:
                return answer[f.key]
            return f.value

        missing = [f.key for f in fields if f.required and not effective(f)]
        if missing:
            return Reject(
                "Please fill in: " + ", ".join(missing),
                self._prompt(fields, content, footer),
            )

        updates = {f.key: effective(f) for f in fields if effective(f) is not None}
        if self._input_model is not None:
            try:
                updates = validate_input(self._input_model, updates, fields=fields)
            except InputValidationError as exc:
                return Reject(str(exc), self._prompt(fields, content, footer))
        return Advance(updates)


class ChoiceStep(Step):
    """A branch point: present a fixed set of options and merge the chosen value.

    Later steps gate on it (``include=lambda s: s.get(key) == "lan"``), so a flow
    forks without the engine knowing the condition. ``content`` is markdown shown
    above the options.
    """

    def __init__(
        self,
        key: str,
        *,
        options: Sequence[Choice],
        label: str,
        help_text: Optional[str] = None,
        content: Optional[Content] = None,
        include: Optional[Include] = None,
        show_in_outline: bool = True,
    ) -> None:
        self.key = key
        self.show_in_outline = show_in_outline
        self._options = list(options)
        self.label = label
        self._help = help_text
        self._content = content or []
        self._include = include

    def _prompt(self, content: Sequence[str]) -> StepPrompt:
        return StepPrompt(
            key=self.key,
            label=self.label,
            help_text=self._help,
            kind="choice",
            content=list(content),
            options=list(self._options),
        )

    async def run(
        self,
        state: Mapping[str, object],
        answer: Optional[Mapping[str, object]],
        action: Optional[str] = None,
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()

        # Already chosen (e.g. seeded by a deep link) -> keep it, skip the screen.
        if state.get(self.key) is not None:
            return Advance()

        content = await _resolve_content(self._content, state)

        if answer is None:
            return Ask(self._prompt(content))

        chosen = answer.get(self.key)
        if chosen not in {option.value for option in self._options}:
            return Reject("Please choose an option.", self._prompt(content))
        return Advance({self.key: chosen})


class SelectStep(Step):
    """Discover or list candidates (a ``discovery`` screen: live options plus an
    optional manual-entry field) and settle on exactly one.

    ``source(state) -> [item]`` produces candidates (a LAN scan, an account's
    devices); ``option(item) -> (value, label)`` renders each; ``pick(state,
    item) -> updates`` folds the chosen item's facts into state. When
    ``manual_field`` is given the screen also offers manual entry, and a filled
    manual answer routes through ``manual(state, value) -> updates``. Auto-advances
    when one candidate is found, or when ``skip_when`` says the choice is already
    made (a host typed up front). ``source`` may re-run on the answer call, so
    back it with a cached snapshot for an expensive scan.
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
        content: Optional[Content] = None,
        kind: str = "discovery",
        manual_field: Optional[StepField] = None,
        manual: Optional[
            Callable[
                [Mapping[str, object], str],
                Union[Mapping[str, object], Awaitable[Mapping[str, object]]],
            ]
        ] = None,
        auto: bool = True,
        skip_when: Optional[Include] = None,
        include: Optional[Include] = None,
        show_in_outline: bool = True,
    ) -> None:
        self.key = key
        self.show_in_outline = show_in_outline
        self._source = source
        self._option = option
        self._pick = pick
        self.label = label
        self._help = help_text
        self._content = content or []
        self._kind = kind
        self._manual_field = manual_field
        self._manual = manual
        self._auto = auto
        self._skip_when = skip_when
        self._include = include

    def _prompt(self, items: Sequence[object], content: Sequence[str]) -> StepPrompt:
        options = [
            Choice(value=str(self._option(item)[0]), label=str(self._option(item)[1]))
            for item in items
        ]
        return StepPrompt(
            key=self.key,
            label=self.label,
            help_text=self._help,
            kind=self._kind,
            content=list(content),
            options=options,
            fields=[self._manual_field] if self._manual_field is not None else [],
        )

    def _match(self, items: Sequence[object], chosen: object) -> Optional[object]:
        for item in items:
            value, _label = self._option(item)
            if str(value) == str(chosen):
                return item
        return None

    async def run(
        self,
        state: Mapping[str, object],
        answer: Optional[Mapping[str, object]],
        action: Optional[str] = None,
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()
        if self._skip_when is not None and await resolve(self._skip_when(state)):
            return Advance()

        items = list(await resolve(self._source(state)))

        if answer is not None:
            # A filled manual-entry field wins over a list selection.
            if self._manual_field is not None and answer.get(self._manual_field.key):
                value = str(answer[self._manual_field.key])
                handler = self._manual or (lambda _s, v: {self.key: v})
                return Advance(await resolve(handler(state, value)))

            item = self._match(items, answer.get(self.key))
            if item is None:
                content = await _resolve_content(self._content, state)
                return Reject(
                    "That option is no longer available",
                    self._prompt(items, content),
                )
            return Advance(await resolve(self._pick(state, item)))

        if not items and self._manual_field is None:
            return Advance()
        if len(items) == 1 and self._auto and self._manual_field is None:
            return Advance(await resolve(self._pick(state, items[0])))
        content = await _resolve_content(self._content, state)
        return Ask(self._prompt(items, content))


class ActionStep(Step):
    """Run an async action that owns its own control flow.

    The general escape hatch for logic that is neither a plain form nor a
    selection: probe a host, attempt a login (ask credentials, then submit),
    verify a code, persist a result. ``action(state, answer)`` returns a
    :class:`StepOutcome` itself -- so it may :class:`Ask` then, on the next call,
    :class:`Advance` or :class:`Reject` -- or returns a plain mapping/``None`` as
    shorthand for :class:`Advance`. ``on_action`` maps a screen action id (e.g.
    ``resend``) to a handler ``(state) -> StepOutcome`` invoked when the caller
    triggers it. Raise :class:`FlowError` for a hard failure.
    """

    def __init__(
        self,
        key: str,
        action: Callable[
            [Mapping[str, object], Optional[Mapping[str, object]]],
            Union[StepOutcome, Mapping[str, object], None, Awaitable],
        ],
        *,
        label: str = "",
        include: Optional[Include] = None,
        show_in_outline: bool = True,
        on_action: Optional[
            Mapping[
                str, Callable[[Mapping[str, object]], Union[StepOutcome, Awaitable]]
            ]
        ] = None,
    ) -> None:
        self.key = key
        self.label = label
        self.show_in_outline = show_in_outline
        self._action = action
        self._include = include
        self._on_action = dict(on_action or {})

    @staticmethod
    def _as_outcome(result, key: str) -> StepOutcome:
        if isinstance(result, (Ask, Advance, Reject)):
            return result
        if result is None:
            return Advance()
        if isinstance(result, Mapping):
            return Advance(result)
        raise FlowError(
            f"action {key!r} returned {type(result).__name__}, "
            "expected a StepOutcome, mapping or None"
        )

    async def run(
        self,
        state: Mapping[str, object],
        answer: Optional[Mapping[str, object]],
        action: Optional[str] = None,
    ) -> StepOutcome:
        if not await _included(self._include, state):
            return Advance()

        if action is not None and action in self._on_action:
            return self._as_outcome(
                await resolve(self._on_action[action](state)), self.key
            )

        return self._as_outcome(await resolve(self._action(state, answer)), self.key)
