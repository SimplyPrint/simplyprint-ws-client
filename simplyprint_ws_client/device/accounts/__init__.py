"""Reusable cloud-account capability surface.

An integration whose brand lets users sign in to a cloud and adopt printers the
cloud already knows about implements the :class:`AccountProvider` Protocol; the
neutral value objects below are the only vocabulary that crosses the boundary
(no brand types, no auth tokens). The integration's own registry decides which
client types back the surface.
"""

from simplyprint_ws_client.device.accounts.base import (
    AccountDevice,
    AccountError,
    AccountProvider,
    AccountResource,
    DeviceNotFound,
    InvalidRegion,
    LoginChallenge,
    LoginResult,
    LoginStatus,
    PasswordChallengeAccountProvider,
)

__all__ = [
    "AccountDevice",
    "AccountError",
    "AccountProvider",
    "AccountResource",
    "DeviceNotFound",
    "InvalidRegion",
    "LoginChallenge",
    "LoginResult",
    "LoginStatus",
    "PasswordChallengeAccountProvider",
]
