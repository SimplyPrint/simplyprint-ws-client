"""Tests for the generic guided-flow engine (``contrib/flow``).

The engine is brand-free, so these exercise it with in-memory fakes that mirror
the two real shapes it must subsume: an *onboarding* flow
(discover -> verify -> setup -> create config) and an *account login* flow
(submit credentials -> maybe answer a code challenge -> save account). If both
drive cleanly here, the engine is a genuine superset of the two parallel
surfaces it replaces.
"""

from dataclasses import dataclass
from typing import List, Optional

import pytest

from simplyprint_ws_client.contrib.flow import (
    ActionStep,
    Advance,
    Ask,
    Done,
    Failed,
    FieldsStep,
    Flow,
    FlowError,
    Phase,
    Prompt,
    Reject,
    SelectStep,
    StepField,
    StepPrompt,
    active_position,
    advance_flow,
    outline,
    run_flow,
)


class Driver:
    """Answers prompts from a per-prompt-key script; records what it was asked."""

    def __init__(self, **scripts):
        self.scripts = {key: list(values) for key, values in scripts.items()}
        self.asked: List[tuple] = []

    async def __call__(self, prompt: StepPrompt, message: Optional[str]):
        self.asked.append((prompt.key, message))
        answers = self.scripts.get(prompt.key)
        if not answers:
            raise AssertionError(f"unexpected prompt {prompt.key!r} (msg={message!r})")
        return answers.pop(0)


@dataclass
class FakeDevice:
    host: str
    serial: Optional[str] = None
    name: str = ""


def make_add_printer_flow(devices: List[FakeDevice], reachable: set):
    async def discover(_state):
        return devices

    def option(device: FakeDevice):
        return (device.host, f"{device.name or device.host}")

    async def pick(_state, device: FakeDevice):
        return {"host": device.host, "serial": device.serial or ""}

    async def verify(state, _answer):
        host = state["host"]
        if host not in reachable:
            raise FlowError(f"no printer reachable at {host}")
        return {"device_type": "fake-x1"}

    def setup_fields(state):
        fields = [StepField(key="access_code", label="Access code", secret=True)]
        if not state.get("serial"):
            fields.append(StepField(key="serial", label="Serial"))
        return fields

    async def finish(state):
        return {
            "host": state["host"],
            "serial": state.get("serial"),
            "access_code": state.get("access_code"),
            "device_type": state.get("device_type"),
        }

    return Flow(
        id="add-printer",
        title="Add a printer",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    SelectStep(
                        "device",
                        source=discover,
                        option=option,
                        pick=pick,
                        label="Choose a printer",
                        skip_when=lambda s: bool(s.get("host")),
                    ),
                    ActionStep("verify", verify),
                    FieldsStep("setup", label="Set up", fields=setup_fields),
                ],
            )
        ],
        finish=finish,
    )


@pytest.mark.asyncio
async def test_onboarding_single_device_auto_selected():
    flow = make_add_printer_flow(
        [FakeDevice(host="10.0.0.5", serial="SER123", name="Printer A")],
        reachable={"10.0.0.5"},
    )
    driver = Driver(setup=[{"access_code": "1234"}])

    result = await run_flow(flow, on_prompt=driver)

    assert result == {
        "host": "10.0.0.5",
        "serial": "SER123",
        "access_code": "1234",
        "device_type": "fake-x1",
    }
    # One device + known serial => no select prompt, only the setup form (no serial field).
    assert driver.asked == [("setup", None)]


@pytest.mark.asyncio
async def test_onboarding_manual_host_skips_discovery_and_asks_serial():
    flow = make_add_printer_flow([], reachable={"192.168.1.9"})
    driver = Driver(setup=[{"access_code": "abcd", "serial": "MANUAL1"}])

    result = await run_flow(
        flow, on_prompt=driver, initial_state={"host": "192.168.1.9"}
    )

    assert result["host"] == "192.168.1.9"
    assert result["serial"] == "MANUAL1"
    # Discovery skipped (host given); setup must include the serial field.
    assert driver.asked == [("setup", None)]


@pytest.mark.asyncio
async def test_onboarding_multiple_devices_prompts_choice():
    flow = make_add_printer_flow(
        [
            FakeDevice(host="10.0.0.5", serial="A", name="Alpha"),
            FakeDevice(host="10.0.0.6", serial="B", name="Beta"),
        ],
        reachable={"10.0.0.5", "10.0.0.6"},
    )
    # SelectStep options are value-keyed; the chosen value is the device host.
    driver = Driver(device=["10.0.0.6"], setup=[{"access_code": "9999"}])

    result = await run_flow(flow, on_prompt=driver)

    assert result["host"] == "10.0.0.6"
    assert result["serial"] == "B"
    assert [key for key, _ in driver.asked] == ["device", "setup"]


@pytest.mark.asyncio
async def test_onboarding_unreachable_host_raises_flow_error():
    flow = make_add_printer_flow(
        [FakeDevice(host="10.0.0.5", serial="A")], reachable=set()
    )
    driver = Driver(setup=[{"access_code": "x"}])

    with pytest.raises(FlowError, match="no printer reachable"):
        await run_flow(flow, on_prompt=driver)


@pytest.mark.asyncio
async def test_onboarding_stateless_round_trip_matches_in_process():
    """Driving advance_flow one increment at a time (the web path), sealing and
    handing back state, must reach the same outcome as run_flow."""
    flow = make_add_printer_flow(
        [
            FakeDevice(host="10.0.0.5", name="Alpha"),
            FakeDevice(host="10.0.0.6", name="Beta"),
        ],
        reachable={"10.0.0.6"},
    )

    # Step 1: no answer -> the select prompt.
    step = await advance_flow(flow)
    assert isinstance(step, Prompt)
    assert step.prompt.key == "device"

    # Round-trip the (sealed) state with the chosen option (value = host).
    sealed = dict(step.state)
    step = await advance_flow(flow, sealed, {"device": "10.0.0.6"})
    assert isinstance(step, Prompt)  # verify passed, now the setup form
    assert step.prompt.key == "setup"

    sealed = dict(step.state)
    step = await advance_flow(flow, sealed, {"access_code": "777", "serial": "S9"})
    assert isinstance(step, Done)
    assert step.value["host"] == "10.0.0.6"
    assert step.value["access_code"] == "777"


def make_account_login_flow(*, valid: dict, challenge_users: set, codes: dict):
    creds_prompt = StepPrompt(
        key="login",
        label="Sign in",
        fields=[
            StepField(key="region", label="Region", choices=["NA", "EU"]),
            StepField(key="username", label="Email"),
            StepField(key="password", label="Password", secret=True),
        ],
    )
    code_prompt = StepPrompt(
        key="verify",
        label="Verification",
        help_text="We emailed you a code.",
        fields=[StepField(key="code", label="Code")],
    )

    async def login(_state, answer):
        if answer is None:
            return Ask(creds_prompt)
        user, password = answer.get("username"), answer.get("password")
        if (user, password) not in valid:
            return Reject("Invalid email or password", creds_prompt)
        if user in challenge_users:
            return Advance({"_challenge": user, "region": answer.get("region")})
        return Advance({"uid": valid[(user, password)], "username": user})

    async def verify(state, answer):
        if answer is None:
            return Ask(code_prompt)
        user = state["_challenge"]
        if answer.get("code") != codes.get(user):
            return Reject("Invalid or expired code", code_prompt)
        return Advance({"uid": "uid-" + user, "username": user})

    async def finish(state):
        return {"uid": state["uid"], "username": state["username"]}

    return Flow(
        id="account-login",
        title="Sign in",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    ActionStep("login", login),
                    ActionStep(
                        "verify", verify, include=lambda s: bool(s.get("_challenge"))
                    ),
                ],
            )
        ],
        finish=finish,
    )


@pytest.mark.asyncio
async def test_login_direct_success_skips_challenge():
    flow = make_account_login_flow(
        valid={("a@x.io", "pw"): "42"}, challenge_users=set(), codes={}
    )
    driver = Driver(login=[{"region": "NA", "username": "a@x.io", "password": "pw"}])

    result = await run_flow(flow, on_prompt=driver)

    assert result == {"uid": "42", "username": "a@x.io"}
    # The conditional verify step never asked for a code.
    assert [key for key, _ in driver.asked] == ["login"]


@pytest.mark.asyncio
async def test_login_challenge_path_asks_for_code():
    flow = make_account_login_flow(
        valid={("a@x.io", "pw"): "42"},
        challenge_users={"a@x.io"},
        codes={"a@x.io": "246810"},
    )
    driver = Driver(
        login=[{"region": "EU", "username": "a@x.io", "password": "pw"}],
        verify=[{"code": "246810"}],
    )

    result = await run_flow(flow, on_prompt=driver)

    assert result == {"uid": "uid-a@x.io", "username": "a@x.io"}
    assert [key for key, _ in driver.asked] == ["login", "verify"]


@pytest.mark.asyncio
async def test_login_bad_password_then_retry_succeeds():
    flow = make_account_login_flow(
        valid={("a@x.io", "right"): "42"}, challenge_users=set(), codes={}
    )
    driver = Driver(
        login=[
            {"region": "NA", "username": "a@x.io", "password": "wrong"},
            {"region": "NA", "username": "a@x.io", "password": "right"},
        ]
    )

    result = await run_flow(flow, on_prompt=driver)

    assert result["uid"] == "42"
    # The login prompt was re-offered with the rejection message.
    assert driver.asked[0] == ("login", None)
    assert driver.asked[1] == ("login", "Invalid email or password")


@pytest.mark.asyncio
async def test_login_bad_code_then_retry_succeeds():
    flow = make_account_login_flow(
        valid={("a@x.io", "pw"): "42"},
        challenge_users={"a@x.io"},
        codes={"a@x.io": "999000"},
    )
    driver = Driver(
        login=[{"region": "NA", "username": "a@x.io", "password": "pw"}],
        verify=[{"code": "000000"}, {"code": "999000"}],
    )

    result = await run_flow(flow, on_prompt=driver)

    assert result["uid"] == "uid-a@x.io"
    assert driver.asked == [
        ("login", None),
        ("verify", None),
        ("verify", "Invalid or expired code"),
    ]


@pytest.mark.asyncio
async def test_login_no_callback_raises_with_failed_message():
    flow = make_account_login_flow(
        valid={("a@x.io", "pw"): "42"}, challenge_users=set(), codes={}
    )
    # No on_prompt: the first Prompt has no answerer.
    with pytest.raises(FlowError, match="requires input"):
        await run_flow(flow)


@pytest.mark.asyncio
async def test_poll_waits_then_advances():
    approved = {"value": False}

    async def approve(_state, _answer):
        if approved["value"]:
            return Advance({"approved": True})
        return Ask(
            StepPrompt(key="approve", label="Press Allow on the device", poll=True)
        )

    async def on_poll(_prompt, _delay):
        approved["value"] = True  # the user pressed Allow between polls

    polled = []

    async def record_poll(prompt, delay):
        polled.append((prompt.key, delay))
        await on_poll(prompt, delay)

    flow = Flow(
        id="pair",
        title="Pair",
        phases=[Phase("main", "Main", steps=[ActionStep("approve", approve)])],
        finish=lambda s: "paired" if s.get("approved") else "no",
    )

    result = await run_flow(flow, on_poll=record_poll)
    assert result == "paired"
    assert polled == [("approve", 2.0)]


@pytest.mark.asyncio
async def test_scalar_answer_normalised_to_single_field():
    """A bare scalar answer maps onto a single-field prompt's field key."""

    async def ask_name(_state, answer):
        if answer is None:
            return Ask(
                StepPrompt(
                    key="name",
                    label="Name",
                    fields=[StepField(key="name", label="Name")],
                )
            )
        return Advance({"name": answer["name"]})

    flow = Flow(
        id="x",
        title="x",
        phases=[Phase("main", "Main", steps=[ActionStep("name", ask_name)])],
        finish=lambda s: s["name"],
    )

    # on_prompt returns a bare string, not a mapping.
    async def on_prompt(_prompt, _msg):
        return "Ada"

    assert await run_flow(flow, on_prompt=on_prompt) == "Ada"


@pytest.mark.asyncio
async def test_fields_step_rejects_missing_required():
    flow = Flow(
        id="f",
        title="f",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "creds",
                        label="Creds",
                        fields=[StepField(key="token", label="Token", required=True)],
                    )
                ],
            )
        ],
        finish=lambda s: s.get("token"),
    )

    step = await advance_flow(flow)
    assert isinstance(step, Prompt)

    # Answer omits the required field -> recoverable Failed, same prompt re-offered.
    step = await advance_flow(flow, step.state, {})
    assert isinstance(step, Failed)
    assert "token" in step.message.lower()
    assert step.prompt is not None and step.prompt.key == "creds"

    # Supplying it advances to completion.
    step = await advance_flow(flow, step.state, {"token": "secret"})
    assert isinstance(step, Done)
    assert step.value == "secret"


@pytest.mark.asyncio
async def test_fields_step_validates_with_pydantic_schema():
    from ipaddress import IPv4Address, IPv6Address
    from typing import Union

    from pydantic import BaseModel, ConfigDict, Field

    class HostInput(BaseModel):
        model_config = ConfigDict(extra="forbid")

        host: Union[IPv4Address, IPv6Address] = Field(title="IP address")

    flow = Flow(
        id="f",
        title="f",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "host",
                        label="Host",
                        fields=[StepField(key="host", label="IP address")],
                        input_model=HostInput,
                    )
                ],
            )
        ],
        finish=lambda s: s.get("host"),
    )

    step = await advance_flow(flow)
    assert isinstance(step, Prompt)
    assert step.prompt.input_schema is not None
    assert step.prompt.input_schema["properties"]["host"]["anyOf"] == [
        {"format": "ipv4", "type": "string"},
        {"format": "ipv6", "type": "string"},
    ]

    failed = await advance_flow(flow, step.state, {"host": "not-an-ip"})
    assert isinstance(failed, Failed)
    assert "IP address" in failed.message
    assert failed.prompt is not None and failed.prompt.key == "host"

    done = await advance_flow(flow, step.state, {"host": "::1"})
    assert isinstance(done, Done)
    assert done.value == "::1"


@pytest.mark.asyncio
async def test_finalize_false_stops_at_terminal_then_commits():
    """The two-phase bridge: advance to the terminal boundary without folding
    (finalize=False -> Ready), then resume the sealed state to commit (Done)."""
    from simplyprint_ws_client.contrib.flow import Ready, advance_flow

    flow = make_add_printer_flow(
        [FakeDevice(host="10.0.0.5", serial="S", name="A")], reachable={"10.0.0.5"}
    )

    # verify-phase: gather facts (select+verify) but don't build the config; the
    # setup form still wants the access code, so we land on its prompt.
    step = await advance_flow(flow, {"host": "10.0.0.5"}, None, finalize=False)
    assert isinstance(step, Prompt) and step.prompt.key == "setup"

    # An immediate-terminal flow (no prompts) returns Ready under finalize=False.
    bare = Flow(
        id="bare",
        title="bare",
        phases=[
            Phase("main", "Main", steps=[ActionStep("noop", lambda s, a: Advance())])
        ],
        finish=lambda s: "committed",
    )
    ready = await advance_flow(bare, {}, None, finalize=False)
    assert isinstance(ready, Ready)
    done = await advance_flow(bare, ready.state, None)
    assert isinstance(done, Done) and done.value == "committed"


@pytest.mark.asyncio
async def test_cursor_is_carried_but_ignored_by_finish():
    from simplyprint_ws_client.contrib.flow import CURSOR_KEY

    seen = {}

    async def finish(state):
        seen.update(state)
        return "ok"

    flow = Flow(
        id="c",
        title="c",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "one", label="One", fields=[StepField(key="a", label="A")]
                    )
                ],
            )
        ],
        finish=finish,
    )

    step = await advance_flow(flow)
    step = await advance_flow(flow, step.state, {"a": "1"})
    assert isinstance(step, Done)
    # The cursor rode along in state but did not disturb the outcome.
    assert CURSOR_KEY in seen
    assert seen["a"] == "1"


@pytest.mark.asyncio
async def test_choice_step_branches_via_include():
    """A ChoiceStep writes the chosen value; later steps include on it, so one
    flow forks (LAN vs cloud) without the engine knowing the condition."""
    from simplyprint_ws_client.contrib.flow import Choice, ChoiceStep

    flow = Flow(
        id="connect",
        title="Connect",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    ChoiceStep(
                        "mode",
                        label="How do you want to connect?",
                        options=[
                            Choice("lan", "Local network"),
                            Choice("cloud", "Cloud account"),
                        ],
                    ),
                    FieldsStep(
                        "lan",
                        label="LAN",
                        fields=[StepField(key="lan_host", label="Host")],
                        include=lambda s: s.get("mode") == "lan",
                    ),
                    FieldsStep(
                        "cloud",
                        label="Cloud",
                        fields=[StepField(key="cloud_user", label="User")],
                        include=lambda s: s.get("mode") == "cloud",
                    ),
                ],
            )
        ],
        finish=lambda s: (s.get("mode"), s.get("lan_host"), s.get("cloud_user")),
    )

    lan = await run_flow(
        flow, on_prompt=Driver(mode=["lan"], lan=[{"lan_host": "10.0.0.5"}])
    )
    assert lan == ("lan", "10.0.0.5", None)

    cloud = await run_flow(
        flow, on_prompt=Driver(mode=["cloud"], cloud=[{"cloud_user": "a@b.io"}])
    )
    assert cloud == ("cloud", None, "a@b.io")


@pytest.mark.asyncio
async def test_choice_step_emits_choice_screen():
    from simplyprint_ws_client.contrib.flow import Choice, ChoiceStep, advance_flow

    flow = Flow(
        id="c",
        title="c",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    ChoiceStep(
                        "mode",
                        label="Pick",
                        content=["Most printers connect over your **LAN**."],
                        options=[Choice("lan", "LAN", description="Recommended")],
                    )
                ],
            )
        ],
        finish=lambda s: s["mode"],
    )
    step = await advance_flow(flow)
    assert isinstance(step, Prompt)
    assert step.prompt.kind == "choice"
    assert step.prompt.content == ["Most printers connect over your **LAN**."]
    assert step.prompt.options[0].value == "lan"
    assert step.prompt.options[0].description == "Recommended"


@pytest.mark.asyncio
async def test_action_step_handles_resend_action():
    """A screen action (resend) routes to the step's on_action handler with no
    answer, re-issuing without advancing."""
    from simplyprint_ws_client.contrib.flow import advance_flow

    sent = {"count": 0}

    def _code_prompt():
        return StepPrompt(
            key="code", label="Code", fields=[StepField(key="code", label="Code")]
        )

    def resend(_state):
        sent["count"] += 1
        return Ask(_code_prompt())

    async def verify(_state, answer):
        if answer is None:
            return Ask(_code_prompt())
        return Advance({"code": answer.get("code")})

    flow = Flow(
        id="v",
        title="v",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[ActionStep("code", verify, on_action={"resend": resend})],
            )
        ],
        finish=lambda s: s["code"],
    )

    step = await advance_flow(flow)
    assert isinstance(step, Prompt) and step.prompt.key == "code"

    step = await advance_flow(flow, step.state, None, action="resend")
    assert isinstance(step, Prompt) and step.prompt.key == "code"
    assert sent["count"] == 1

    step = await advance_flow(flow, step.state, {"code": "123456"})
    assert isinstance(step, Done) and step.value == "123456"


@pytest.mark.asyncio
async def test_step_carries_markdown_content_and_kind():
    from simplyprint_ws_client.contrib.flow import advance_flow

    flow = Flow(
        id="i",
        title="i",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "setup",
                        label="Set up",
                        content=["## Where to find it", "Open **Settings → Network**."],
                        fields=[StepField(key="x", label="X")],
                    )
                ],
            )
        ],
        finish=lambda s: s["x"],
    )
    step = await advance_flow(flow)
    assert step.prompt.kind == "form"
    assert step.prompt.content == [
        "## Where to find it",
        "Open **Settings → Network**.",
    ]


@pytest.mark.asyncio
async def test_discovery_manual_entry_routes_through_manual():
    """A discovery step with no candidates still offers manual entry, and a filled
    manual field routes through the manual handler."""
    from simplyprint_ws_client.contrib.flow import advance_flow

    async def source(_state):
        return []

    flow = Flow(
        id="d",
        title="d",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    SelectStep(
                        "device",
                        source=source,
                        option=lambda item: (item, item),
                        pick=lambda _s, item: {"host": item},
                        label="Find your printer",
                        manual_field=StepField(key="manual_host", label="IP address"),
                        manual=lambda _s, value: {"host": value},
                    )
                ],
            )
        ],
        finish=lambda s: s["host"],
    )

    step = await advance_flow(flow)
    assert isinstance(step, Prompt) and step.prompt.kind == "discovery"
    assert step.prompt.fields[0].key == "manual_host"

    step = await advance_flow(flow, step.state, {"manual_host": "192.168.1.9"})
    assert isinstance(step, Done) and step.value == "192.168.1.9"


@pytest.mark.asyncio
async def test_choice_step_skipped_when_preseeded():
    """A pre-seeded choice value (a deep link) skips the choice screen entirely."""
    from simplyprint_ws_client.contrib.flow import Choice, ChoiceStep

    flow = Flow(
        id="c",
        title="c",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    ChoiceStep(
                        "mode",
                        label="?",
                        options=[Choice("lan", "LAN"), Choice("cloud", "Cloud")],
                    )
                ],
            )
        ],
        finish=lambda s: s["mode"],
    )
    # No on_prompt needed: the seeded value short-circuits the screen.
    assert await run_flow(flow, initial_state={"mode": "cloud"}) == "cloud"


@pytest.mark.asyncio
async def test_phase_include_branches_outline_and_marks_active_step():
    """Phases own their steps and branch via ``include``: the inactive side drops
    out of ``outline`` and never runs, and ``active_position`` names the live phase
    and substep straight from the cursor -- so a UI can render and track the stepper
    before every screen is answered, with no hand-tagged phase strings."""
    from simplyprint_ws_client.contrib.flow import Choice, ChoiceStep

    def is_cloud(s):
        return s.get("mode") == "cloud"

    def is_lan(s):
        return not is_cloud(s)

    flow = Flow(
        id="add",
        title="Add",
        phases=[
            Phase(
                "connect",
                "Connect",
                steps=[
                    ChoiceStep(
                        "mode",
                        label="How do you want to connect?",
                        options=[Choice("lan", "LAN"), Choice("cloud", "Cloud")],
                    )
                ],
            ),
            Phase(
                "find",
                "Find",
                include=is_lan,
                steps=[
                    FieldsStep(
                        "find",
                        label="Find it",
                        fields=[StepField(key="host", label="Host")],
                    )
                ],
            ),
            Phase(
                "sign-in",
                "Sign in",
                include=is_cloud,
                steps=[
                    FieldsStep(
                        "signin",
                        label="Sign in",
                        fields=[StepField(key="user", label="User")],
                    )
                ],
            ),
            Phase("done", "Done"),
        ],
        finish=lambda s: s.get("host", ""),
    )

    # Before any choice the default (LAN) side shows; the cloud phase is hidden.
    assert [p.id for p in outline(flow)] == ["connect", "find", "done"]
    # Choosing cloud swaps the branch in the resolved outline.
    assert [p.id for p in outline(flow, {"mode": "cloud"})] == [
        "connect",
        "sign-in",
        "done",
    ]

    # The active phase + substep fall straight out of the cursor in the sealed state.
    step = await advance_flow(flow)
    assert isinstance(step, Prompt) and step.prompt.key == "mode"
    phase, sub = active_position(flow, step.state)
    assert phase.id == "connect" and sub.key == "mode" and sub.label

    step = await advance_flow(flow, step.state, {"mode": "lan"})
    assert isinstance(step, Prompt) and step.prompt.key == "find"
    phase, sub = active_position(flow, step.state)
    assert phase.id == "find" and sub.key == "find"
    # The unchosen cloud phase never ran and is gone from the outline.
    assert "sign-in" not in [p.id for p in outline(flow, step.state)]


def test_outline_hides_steps_marked_not_visible():
    flow = Flow(
        id="hidden",
        title="hidden",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "host",
                        label="Host",
                        fields=[StepField(key="host", label="Host")],
                    ),
                    ActionStep(
                        "verify",
                        lambda _s, _a: Advance(),
                        label="Verify",
                        show_in_outline=False,
                    ),
                ],
            )
        ],
        finish=lambda _s: None,
    )

    [phase] = outline(flow)
    assert [step.key for step in phase.steps] == ["host"]


@pytest.mark.asyncio
async def test_flow_seedable_defaults_empty_and_carries_declared_keys():
    """``Flow.seedable`` is the allowlist a stateless front door intersects an
    untrusted launch context against. Empty by default; a flow opts keys in, and a
    secret is never opted in -- so it can never be seeded to skip a step."""
    bare = Flow(
        id="b",
        title="b",
        phases=[Phase("main", "Main", steps=[])],
        finish=lambda s: None,
    )
    assert bare.seedable == frozenset()

    seeded = Flow(
        id="s",
        title="s",
        phases=[Phase("main", "Main", steps=[])],
        finish=lambda s: None,
        seedable=frozenset({"mode", "host", "serial", "device_type"}),
    )
    context = {
        "mode": "lan",
        "host": "10.0.0.5",
        "serial": "S1",
        "access_code": "secret",  # not opted in
    }
    seeded_in = {k: v for k, v in context.items() if k in seeded.seedable}
    assert seeded_in == {"mode": "lan", "host": "10.0.0.5", "serial": "S1"}
    assert "access_code" not in seeded_in


@pytest.mark.asyncio
async def test_callable_content_curates_by_state():
    """A step's ``content`` may be a callable of state, so a screen shows only the
    guide for what's known (the picked model) instead of every guide at once."""

    def guide(state):
        model = state.get("device_type")
        if model == "x1":
            return ["Guide for **X1**."]
        if model == "a1":
            return ["Guide for **A1**."]
        return ["Generic: open Settings → Network."]

    flow = Flow(
        id="g",
        title="g",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "find",
                        label="Find it",
                        content=guide,
                        fields=[StepField(key="host", label="Host")],
                    )
                ],
            )
        ],
        finish=lambda s: s["host"],
    )

    step = await advance_flow(flow, {"device_type": "x1"})
    assert step.prompt.content == ["Guide for **X1**."]
    step = await advance_flow(flow, {"device_type": "a1"})
    assert step.prompt.content == ["Guide for **A1**."]
    step = await advance_flow(flow, {})  # unknown model -> generic
    assert step.prompt.content == ["Generic: open Settings → Network."]


@pytest.mark.asyncio
async def test_prefilled_field_value_used_when_answer_omits_it():
    """A prefilled field carries its known value: a renderer may collapse it into a
    summary and not re-submit it, yet the value is folded into state and satisfies
    the required check; an explicit answer overrides it."""
    flow = Flow(
        id="p",
        title="p",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "find",
                        label="Find it",
                        fields=[
                            StepField(
                                key="host", label="IP", value="10.0.0.5", prefilled=True
                            ),
                            StepField(
                                key="access_code", label="Access code", secret=True
                            ),
                        ],
                    )
                ],
            )
        ],
        finish=lambda s: (s.get("host"), s.get("access_code")),
    )

    step = await advance_flow(flow)
    assert isinstance(step, Prompt)
    host_field = step.prompt.fields[0]
    assert host_field.prefilled and host_field.value == "10.0.0.5"

    # The form submits only the missing secret; the prefilled host rides along.
    step = await advance_flow(flow, step.state, {"access_code": "abcd1234"})
    assert isinstance(step, Done)
    assert step.value == ("10.0.0.5", "abcd1234")


@pytest.mark.asyncio
async def test_prefilled_field_edited_value_overrides():
    flow = Flow(
        id="p2",
        title="p2",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "find",
                        label="Find it",
                        fields=[
                            StepField(
                                key="host", label="IP", value="10.0.0.5", prefilled=True
                            )
                        ],
                    )
                ],
            )
        ],
        finish=lambda s: s.get("host"),
    )
    step = await advance_flow(flow)
    # The user expanded the summary and corrected the value.
    step = await advance_flow(flow, step.state, {"host": "10.0.0.9"})
    assert isinstance(step, Done) and step.value == "10.0.0.9"


@pytest.mark.asyncio
async def test_recommended_choice_is_carried_to_the_screen():
    from simplyprint_ws_client.contrib.flow import Choice, ChoiceStep

    flow = Flow(
        id="r",
        title="r",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    ChoiceStep(
                        "mode",
                        label="How?",
                        options=[
                            Choice("lan", "Local network", recommended=True),
                            Choice("cloud", "Cloud account"),
                        ],
                    )
                ],
            )
        ],
        finish=lambda s: s["mode"],
    )
    step = await advance_flow(flow)
    assert step.prompt.options[0].recommended is True
    assert step.prompt.options[1].recommended is False


@pytest.mark.asyncio
async def test_step_footer_renders_below_and_curates_by_state():
    """``footer`` is markdown shown below the inputs (a "where do I find this?"
    walkthrough), and like ``content`` it may be a callable curated by state."""

    def guide(state):
        return [f"Guide for {state.get('device_type') or 'any'}"]

    flow = Flow(
        id="f",
        title="f",
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    FieldsStep(
                        "find",
                        label="Find it",
                        content=["Enter the IP below."],
                        footer=guide,
                        fields=[StepField(key="host", label="Host")],
                    )
                ],
            )
        ],
        finish=lambda s: s["host"],
    )

    step = await advance_flow(flow, {"device_type": "x1"})
    assert step.prompt.content == ["Enter the IP below."]
    assert step.prompt.footer == ["Guide for x1"]


@pytest.mark.asyncio
async def test_fully_seeded_device_insta_adds_in_one_advance():
    """The quick-start path: a discovered device seeds every fact a flow needs, so
    each step's include gate is satisfied and the flow reaches Done in a single
    advance -- no prompt shown (an insta-add)."""

    async def verify(state, _answer):
        # Reachable + identity already known; nothing to add.
        return Advance()

    flow = Flow(
        id="add",
        title="Add",
        seedable=frozenset({"host", "device_type"}),
        phases=[
            Phase(
                "main",
                "Main",
                steps=[
                    # identify: run only when the model is unknown (skipped once seeded).
                    FieldsStep(
                        "identify",
                        label="Pick model",
                        fields=[StepField(key="device_type", label="Model")],
                        include=lambda s: not s.get("device_type"),
                    ),
                    # find+info: run only when something required is still missing.
                    FieldsStep(
                        "find",
                        label="Find it",
                        fields=[StepField(key="host", label="Host")],
                        include=lambda s: not s.get("host"),
                    ),
                    ActionStep("verify", verify),
                ],
            )
        ],
        finish=lambda s: {"host": s.get("host"), "device_type": s.get("device_type")},
    )

    # Seed everything a discovered device carried -> straight to Done, no prompt.
    step = await advance_flow(flow, {"host": "10.0.0.5", "device_type": "x1"})
    assert isinstance(step, Done)
    assert step.value == {"host": "10.0.0.5", "device_type": "x1"}
