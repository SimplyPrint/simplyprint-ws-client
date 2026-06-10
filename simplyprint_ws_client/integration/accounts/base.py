"""Brand-agnostic cloud-account provider surface.

Some integrations let a user sign in to the brand's cloud, then pick a printer
the cloud already knows about instead of pairing it on the LAN. Only some brands
back this, but its shape is general: a managed *account* resource plus a small
stateless *login* state machine (sign in -> maybe answer a code challenge ->
account saved), and a way to list and adopt the cloud's devices.

This module owns that surface as a *neutral* :class:`AccountProvider` Protocol
plus a handful of value objects. An integration supplies only the concrete
provider -- how it talks to its cloud, what a challenge looks like, how it folds
a cloud device into a ``PrinterConfig``. The integration's own registry projects
which client types back the surface; this module stays brand-neutral.

HARD RULES (machine-checked: brand-free NAME tokens + text + field-token scans):

* No brand imports, enums, strings, ports, topic shapes or model field names in
  this module.
* Every method speaks the neutral value objects below -- no brand type ever
  appears in a signature.
* Auth tokens never appear in a value object; only the non-secret continuation
  state of a login challenge is carried, and the caller round-trips it sealed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simplyprint_ws_client import PrinterConfig


__all__ = [
    "AccountResource",
    "AccountDevice",
    "LoginStatus",
    "LoginResult",
    "LoginChallenge",
    "AccountError",
    "InvalidRegion",
    "DeviceNotFound",
    "AccountProvider",
]


@dataclass(frozen=True)
class AccountResource:
    """A saved cloud account, with auth tokens deliberately omitted.

    ``uid`` is the brand's own account identifier reduced to a plain string so
    this neutral type carries no brand id type. ``region`` is the brand's region
    token (also a plain string). ``invalid`` marks an account whose stored auth
    no longer works; ``expires_soon`` flags one whose auth is about to lapse so
    the UI can prompt a re-login. ``expires_at`` is that lapse moment as a Unix
    timestamp (seconds) when the brand can supply it, so the UI can show a live
    countdown and offer a re-auth before it elapses; ``None`` when unknown.
    """

    uid: str
    region: str
    username: str
    invalid: bool = False
    expires_soon: bool = False
    expires_at: Optional[float] = None


@dataclass(frozen=True)
class AccountDevice:
    """A printer the cloud account already knows about.

    Only the network-neutral facts every brand can supply: a stable ``serial``,
    a human ``name``/``model``, whether it is ``online``, and whether it is
    ``paired`` (carries whatever the provider needs to connect -- the pairing
    secret itself is never exposed). Providers that have a per-model image may
    surface a resolved URL so the add UI does not guess any app convention.
    """

    serial: str
    name: str
    model: str = ""
    online: bool = False
    paired: bool = False
    model_image_url: Optional[str] = None


class LoginStatus(str, Enum):
    """The three terminal outcomes of one login / verify step.

    * ``COMPLETED`` -- the account is signed in and saved; ``LoginResult.account``
      is set.
    * ``CHALLENGE`` -- the brand needs an extra step (a code / 2FA); the
      ``LoginResult.challenge`` carries the non-secret continuation state.
    * ``FAILED`` -- the credentials / code were rejected.
    """

    COMPLETED = "completed"
    CHALLENGE = "challenge"
    FAILED = "failed"


@dataclass(frozen=True)
class LoginChallenge:
    """The continuation state of a pending login that needs a second step.

    Carries only non-secret facts the provider needs to resume the flow on the
    next call (the brand's region token, which kind of challenge it is, and any
    opaque continuation values keyed by name). Never a token. The caller seals
    this into an opaque signed token and hands it back to :meth:`verify` /
    :meth:`resend_code`; the provider reconstitutes its API from it.

    ``method`` names the challenge kind as a plain brand-neutral string (the
    brand's own login-type token reduced to text). ``prompt``/``help_text`` let a
    UI describe the step; ``can_resend`` says whether a fresh code can be
    requested. ``context`` holds any extra continuation values (an email, a
    server-issued key) the provider needs back -- all non-secret.
    """

    region: str
    method: str
    prompt: str = "Enter the verification code"
    help_text: Optional[str] = None
    can_resend: bool = False
    context: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LoginResult:
    """Outcome of one :meth:`PrinterAccountProvider.login` / :meth:`verify` step.

    Exactly one of ``account`` (when ``status`` is ``COMPLETED``) or ``challenge``
    (when ``status`` is ``CHALLENGE``) is set; both are ``None`` on ``FAILED``.
    ``message`` carries a human reason on failure.
    """

    status: LoginStatus
    account: Optional[AccountResource] = None
    challenge: Optional[LoginChallenge] = None
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is LoginStatus.COMPLETED:
            valid = self.account is not None and self.challenge is None
        elif self.status is LoginStatus.CHALLENGE:
            valid = self.account is None and self.challenge is not None
        else:
            valid = self.account is None and self.challenge is None

        if not valid:
            raise ValueError(f"invalid LoginResult for status {self.status.value}")

    @classmethod
    def completed(cls, account: AccountResource) -> "LoginResult":
        return cls(LoginStatus.COMPLETED, account=account)

    @classmethod
    def challenge_required(cls, challenge: LoginChallenge) -> "LoginResult":
        return cls(LoginStatus.CHALLENGE, challenge=challenge)

    @classmethod
    def failed(cls, message: str) -> "LoginResult":
        return cls(LoginStatus.FAILED, message=message)


class AccountError(RuntimeError):
    """Raised when an account operation cannot proceed (unknown region, the
    cloud rejected an otherwise-valid request, an account could not be saved).

    A neutral error type so callers can distinguish an account-flow failure from
    arbitrary runtime errors without importing a brand exception.
    """


class InvalidRegion(AccountError):
    """Raised when a login/verify is asked for a region the provider does not
    support. A distinct subclass so the caller can map it to an input-validation
    error (the caller picked an impossible region) rather than an upstream
    failure (the cloud rejected an otherwise-valid request)."""


class DeviceNotFound(AccountError):
    """Raised by :meth:`PrinterAccountProvider.adopt_device` when the account has
    no device with the given serial. A distinct subclass so the caller can map it
    to a not-found rather than a generic upstream failure."""


@runtime_checkable
class AccountProvider(Protocol):
    """Structural type for saved-account/device adoption capability.

    A :class:`~typing.Protocol`, deliberately **not** an ABC: a concrete
    capability satisfies it *by shape* and does not inherit it. This base surface
    avoids prescribing a login method; OAuth, device-code, API-key, and
    password/challenge flows can each expose their own flow-owned authenticator
    while still sharing saved-account storage and device adoption.
    """

    def get_accounts(self) -> List[AccountResource]: ...

    async def delete_account(self, uid: str) -> bool: ...

    async def get_devices(self, uid: str) -> Optional[List[AccountDevice]]: ...

    async def adopt_device(self, uid: str, serial: str) -> "PrinterConfig": ...


@runtime_checkable
class PasswordChallengeAccountProvider(AccountProvider, Protocol):
    """Optional credential shape for username/password plus code challenge flows."""

    async def login(self, region: str, username: str, password: str) -> LoginResult: ...

    async def verify(self, challenge: LoginChallenge, code: str) -> LoginResult: ...

    async def resend_code(self, challenge: LoginChallenge) -> bool: ...
