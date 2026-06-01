"""Brand-agnostic onboarding API surface.

Every printer integration today hardcodes its own ``add a printer`` flow with
ad-hoc stdin/CLI prompts. The shapes differ in wording only -- underneath
they all walk the same pipeline:

    DISCOVERY -> SELECT -> VERIFY -> MULTI-STAGE SETUP -> CREATE CONFIG -> connect

This module owns that pipeline as a *neutral* orchestrator
(:func:`onboard_printer`) plus a small set of value objects, and pushes every
brand specific decision out to the :class:`PrinterOnboardingHooks` ABC. A brand
supplies only its steps -- how to scan the LAN, how to handshake a host, what
extra auth stages it needs, and how to fold the gathered facts into a
``PrinterConfig``. The orchestration (looping setup stages, threading the
selected device through to verification, asking the *caller* -- never
stdin -- for the next step's input) lives here, once.

HARD RULES enforced by the integration's ``tests/test_architecture.py`` and the
library's own contrib brand-free contract test:

* No brand imports, enums, strings, ports, topic shapes or model field names.
* No interactive stdin reads and no stdout writes -- this layer never touches a
  terminal. The
  orchestrator collects interactive input through an injected
  ``step_input_callback``; a side-car CLI or the new UI provides the real I/O.
* The only non-stdlib dependency is the neutral ``PrinterConfig`` *type* from
  ``simplyprint_ws_client``.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Union

from simplyprint_ws_client import PrinterConfig

__all__ = [
    "DiscoveredDevice",
    "VerificationResult",
    "StepField",
    "SetupStepResult",
    "StepPrompt",
    "PrinterOnboardingHooks",
    "OnboardingError",
    "StepInputCallback",
    "onboard_printer",
]


@dataclass(frozen=True)
class DiscoveredDevice:
    """A device a brand's discovery turned up on the network.

    Only the network-neutral facts every brand can supply: a reachable
    ``host`` (IP or hostname) plus an optional human ``name`` and ``serial``.
    Brand-specific discovery payload (model codes, signed tokens, SSDP headers)
    rides in ``extra`` so the brand can read it back during :meth:`verify`
    without this neutral type ever naming a brand field.
    """

    host: str
    name: Optional[str] = None
    serial: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of probing a single host -- the handshake / ``check_device`` step.

    ``device_type`` is the brand's own model/type token reduced to a plain
    string (so this neutral type carries no brand enum). ``serial`` and ``host``
    pin the device; any further brand facts needed to build a config (access
    codes, firmware, capabilities) ride in ``extra``.
    """

    device_type: str
    host: str
    serial: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StepField:
    """A single input field in a setup prompt.

    UI and CLI renderers consume the same neutral field description. Brands can
    return one field or many without the prompt transport knowing anything about
    the brand-specific setup stage.
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
    """A request for input the orchestrator hands to its ``step_input_callback``.

    A multi-stage setup hook returns one of these to drive an interactive stage
    (entering an access code, confirming a pairing, choosing a profile) without
    ever reading stdin itself. The caller -- CLI, web wizard, test --
    renders ``key``/``label``/``help_text`` however it likes and returns the
    user's answer. ``secret`` marks a field whose echo should be masked;
    ``choices`` offers a closed set when the stage is a selection.

    ``poll`` marks a stage that needs no input from the user, only that they do
    something on the device (press Allow on a touchscreen) before the same step
    is retried. The caller shows ``label``/``help_text`` and re-runs the step on
    a timer rather than asking for a field -- a neutral way to express a
    wait-on-device stage without the transport knowing which brand it is.
    """

    key: str
    label: str
    help_text: Optional[str] = None
    secret: bool = False
    choices: Optional[List[str]] = None
    default: Optional[str] = None
    fields: List[StepField] = field(default_factory=list)
    poll: bool = False


@dataclass(frozen=True)
class SetupStepResult:
    """Result of one multi-stage setup stage.

    ``config_update`` accumulates neutral key/value facts gathered so far.
    ``next_step`` names the stage to run next (``None`` => setup is complete and
    the orchestrator advances to :meth:`PrinterOnboardingHooks.create_config`).
    ``prompt`` carries the input request for the *next* stage when one is
    needed, so the orchestrator can ask the caller and feed the answer back in.
    """

    config_update: Dict[str, object] = field(default_factory=dict)
    next_step: Optional[str] = None
    prompt: Optional[StepPrompt] = None


class OnboardingError(RuntimeError):
    """Raised when onboarding cannot proceed (no device selected, verify failed).

    A neutral error type so callers can distinguish an onboarding-flow failure
    from arbitrary runtime errors without importing a brand exception.
    """


#: Signature of the input provider the orchestrator calls instead of reading stdin.
#: Given the :class:`StepPrompt` the current stage needs answered, it returns the
#: caller-supplied answer(s) (and may be async, e.g. an HTTP round-trip to a UI).
StepInputCallback = Callable[
    [StepPrompt],
    Union[str, Dict[str, object], Awaitable[Union[str, Dict[str, object]]]],
]


class PrinterOnboardingHooks(ABC):
    """Brand-supplied steps the neutral orchestrator drives.

    A brand integration subclasses this and implements only what is genuinely
    brand specific. The orchestrator owns *when* each step runs and *how* the
    pieces thread together; the brand owns *what* each step does. Every method
    speaks the neutral value objects above -- no brand type ever appears in a
    signature, satisfying the core hook-annotation guard.
    """

    @abstractmethod
    async def discover(self, timeout: float) -> List[DiscoveredDevice]:
        """Scan the network for this brand's printers for up to ``timeout`` seconds.

        Returns the devices found (possibly empty). Discovery is optional from
        the orchestrator's point of view: a brand without LAN discovery may
        return ``[]`` and rely on a manually supplied ``host``.
        """

    @abstractmethod
    async def verify(self, host: str) -> VerificationResult:
        """Probe ``host`` and confirm a printer of this brand answers.

        This is the handshake / ``check_device`` step. Raise
        :class:`OnboardingError` (or any exception) if ``host`` is unreachable
        or not a printer of this brand; the orchestrator surfaces the failure.
        """

    @abstractmethod
    async def multi_stage_setup(
        self,
        result: VerificationResult,
        step_input: Optional[Dict[str, object]],
    ) -> SetupStepResult:
        """Advance the post-verification setup state machine by one stage.

        Called repeatedly by the orchestrator. ``step_input`` is ``None`` on the
        first call and otherwise carries the answer(s) the caller gave for the
        previous stage's :class:`StepPrompt`. Return a :class:`SetupStepResult`
        whose ``next_step`` is ``None`` once setup is done. A brand with no extra
        stages returns a single terminal result immediately.
        """

    @abstractmethod
    def create_config(
        self,
        verify: VerificationResult,
        setup: SetupStepResult,
    ) -> PrinterConfig:
        """Fold the gathered facts into a persistable ``PrinterConfig``.

        Pure, synchronous assembly: combine the verified device identity with
        the accumulated ``setup.config_update`` into the brand's concrete
        ``PrinterConfig`` subclass. The orchestrator persists/connects it.
        """


async def _resolve(value):
    """Await ``value`` if it is awaitable, else return it as-is.

    Lets a brand implement any hook -- or the caller's ``step_input_callback`` --
    as either sync or async without the orchestrator caring which.
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def onboard_printer(
    hooks: PrinterOnboardingHooks,
    *,
    host: Optional[str] = None,
    selected_device: Optional[DiscoveredDevice] = None,
    discover_timeout: float = 5.0,
    select_callback: Optional[
        Callable[[List[DiscoveredDevice]], Union[DiscoveredDevice, None, Awaitable]]
    ] = None,
    step_input_callback: Optional[StepInputCallback] = None,
    max_setup_stages: int = 32,
) -> PrinterConfig:
    """Drive one printer from discovery to a ready-to-persist ``PrinterConfig``.

    The single neutral entry point that owns the onboarding pipeline:

    1. **Resolve a host.** Prefer an explicit ``host``; else a ``selected_device``;
       else run :meth:`PrinterOnboardingHooks.discover` and pick a device --
       automatically when exactly one is found, otherwise via ``select_callback``
       (the caller's chooser; stdin lives there, never here).
    2. **Verify** the host via :meth:`PrinterOnboardingHooks.verify`.
    3. **Multi-stage setup**: loop :meth:`PrinterOnboardingHooks.multi_stage_setup`,
       asking ``step_input_callback`` to answer each stage's :class:`StepPrompt`
       until a stage reports ``next_step is None``.
    4. **Create config** via :meth:`PrinterOnboardingHooks.create_config`.

    :raises OnboardingError: if no host can be resolved or no device is selected.
    """
    host = await _resolve_host(
        hooks,
        host=host,
        selected_device=selected_device,
        discover_timeout=discover_timeout,
        select_callback=select_callback,
    )

    verify_result = await hooks.verify(host)

    setup_result = await _run_setup_stages(
        hooks,
        verify_result,
        step_input_callback=step_input_callback,
        max_setup_stages=max_setup_stages,
    )

    return hooks.create_config(verify_result, setup_result)


async def _resolve_host(
    hooks: PrinterOnboardingHooks,
    *,
    host: Optional[str],
    selected_device: Optional[DiscoveredDevice],
    discover_timeout: float,
    select_callback,
) -> str:
    """Settle on a single reachable host to verify (discover + select as needed)."""
    if host:
        return host

    if selected_device is not None:
        return selected_device.host

    devices = await hooks.discover(discover_timeout)

    if not devices:
        raise OnboardingError("no host supplied and discovery found no devices")

    if len(devices) == 1 and select_callback is None:
        return devices[0].host

    if select_callback is None:
        raise OnboardingError(
            "multiple devices discovered but no select_callback was provided"
        )

    chosen = await _resolve(select_callback(devices))

    if chosen is None:
        raise OnboardingError("no device selected")

    return chosen.host


async def _run_setup_stages(
    hooks: PrinterOnboardingHooks,
    verify_result: VerificationResult,
    *,
    step_input_callback: Optional[StepInputCallback],
    max_setup_stages: int,
) -> SetupStepResult:
    """Loop the brand setup state machine, gathering answers, until it terminates.

    ``config_update`` is carried forward across stages so the terminal result
    handed to :meth:`PrinterOnboardingHooks.create_config` holds the union of
    everything gathered.
    """
    step_input: Optional[Dict[str, object]] = None
    accumulated: Dict[str, object] = {}

    for _ in range(max_setup_stages):
        result = await hooks.multi_stage_setup(verify_result, step_input)

        if result.config_update:
            accumulated.update(result.config_update)

        if result.next_step is None:
            return SetupStepResult(
                config_update=accumulated,
                next_step=None,
                prompt=result.prompt,
            )

        if result.prompt is None:
            # Stage advances without needing input from the caller.
            step_input = None
            continue

        if step_input_callback is None:
            raise OnboardingError(
                "setup requires input but no step_input_callback was provided"
            )

        answer = await _resolve(step_input_callback(result.prompt))
        step_input = answer if isinstance(answer, dict) else {result.prompt.key: answer}

    raise OnboardingError(
        f"multi-stage setup did not terminate within {max_setup_stages} stages"
    )
