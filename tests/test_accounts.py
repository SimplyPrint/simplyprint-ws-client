import pytest

from simplyprint_ws_client.device.accounts import (
    AccountResource,
    LoginChallenge,
    LoginResult,
    LoginStatus,
)


def test_login_result_enforces_completed_account():
    account = AccountResource(uid="1", region="eu", username="user")

    assert LoginResult.completed(account).account is account
    with pytest.raises(ValueError):
        LoginResult(LoginStatus.COMPLETED)


def test_login_result_enforces_challenge_payload():
    challenge = LoginChallenge(region="eu", method="code")

    assert LoginResult.challenge_required(challenge).challenge is challenge
    with pytest.raises(ValueError):
        LoginResult(LoginStatus.CHALLENGE)


def test_login_result_failed_carries_no_success_payloads():
    account = AccountResource(uid="1", region="eu", username="user")

    assert LoginResult.failed("bad credentials").message == "bad credentials"
    with pytest.raises(ValueError):
        LoginResult(LoginStatus.FAILED, account=account)
