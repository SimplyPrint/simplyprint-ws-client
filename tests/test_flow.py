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
    Prompt,
    Reject,
    SelectStep,
    StepField,
    StepPrompt,
    advance_flow,
    run_flow,
)


# -- a scripted driver standing in for a CLI / web wizard --------------------


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


# -- onboarding-shaped flow --------------------------------------------------


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
    driver = Driver(device=["Beta"], setup=[{"access_code": "9999"}])

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

    # Round-trip the (sealed) state with the chosen option.
    sealed = dict(step.state)
    step = await advance_flow(flow, sealed, {"device": "Beta"})
    assert isinstance(step, Prompt)  # verify passed, now the setup form
    assert step.prompt.key == "setup"

    sealed = dict(step.state)
    step = await advance_flow(flow, sealed, {"access_code": "777", "serial": "S9"})
    assert isinstance(step, Done)
    assert step.value["host"] == "10.0.0.6"
    assert step.value["access_code"] == "777"


# -- account-login-shaped flow -----------------------------------------------


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
        steps=[
            ActionStep("login", login),
            ActionStep("verify", verify, include=lambda s: bool(s.get("_challenge"))),
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


# -- engine mechanics --------------------------------------------------------


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
        steps=[ActionStep("approve", approve)],
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
        steps=[ActionStep("name", ask_name)],
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
        steps=[
            FieldsStep(
                "creds",
                label="Creds",
                fields=[StepField(key="token", label="Token", required=True)],
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
async def test_cursor_is_carried_but_ignored_by_finish():
    from simplyprint_ws_client.contrib.flow import CURSOR_KEY

    seen = {}

    async def finish(state):
        seen.update(state)
        return "ok"

    flow = Flow(
        id="c",
        title="c",
        steps=[FieldsStep("one", label="One", fields=[StepField(key="a", label="A")])],
        finish=finish,
    )

    step = await advance_flow(flow)
    step = await advance_flow(flow, step.state, {"a": "1"})
    assert isinstance(step, Done)
    # The cursor rode along in state but did not disturb the outcome.
    assert CURSOR_KEY in seen
    assert seen["a"] == "1"
