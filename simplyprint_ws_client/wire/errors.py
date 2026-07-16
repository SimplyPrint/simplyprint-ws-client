"""Structured errors raised and emitted by connection transports."""

from __future__ import annotations

from typing import Optional, Type, TypeVar, Union


E = TypeVar("E", bound="TransportError")
ErrorCode = Union[int, str]


class TransportError(Exception):
    """Base class for standardized transport failures.

    ``transport_error`` preserves the native exception that came from the wire
    library. It is also installed as ``__cause__`` so ordinary exception tooling
    keeps the original traceback and type visible.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[ErrorCode] = None,
        transport_error: Optional[BaseException] = None,
    ) -> None:
        self.code = code
        self.transport_error = transport_error
        if message is None and transport_error is not None:
            message = str(transport_error)
        super().__init__(message or self.__class__.__name__)
        if transport_error is not None:
            self.__cause__ = transport_error

    @classmethod
    def wrap(
        cls: Type[E],
        error: BaseException,
        message: Optional[str] = None,
        *,
        code: Optional[ErrorCode] = None,
    ) -> E:
        """Build this error class around a native transport exception."""
        return cls(message, code=code, transport_error=error)


class NotConnected(TransportError):
    """Raised when a send hits a link without a live wire."""


class TransientError(TransportError):
    """A recoverable wire failure that the reconnect loop should retry from."""


class FatalError(TransportError):
    """A wire failure that looks unrecoverable, surfaced distinctly to callers."""


class AuthenticationError(FatalError):
    """The endpoint rejected the configured credentials."""
