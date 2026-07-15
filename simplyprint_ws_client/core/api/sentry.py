import logging
import re
import traceback
from typing import TYPE_CHECKING, Any, Callable, List

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.threading import ThreadingIntegration

from simplyprint_ws_client.const import VERSION

if TYPE_CHECKING:
    from simplyprint_ws_client.core.settings import ClientSettings

# This cannot be 0.
MAX_UNIQUE_EXCEPTIONS = 100
MAX_SAMPLES_PER_EXC = 5
DEFAULT_SAMPLE_RATE = 0.1

# Generic, brand-free PII redaction applied to every outgoing event. The aim is
# that only the error itself (message + traceback) leaves the device -- never a
# user's home path, a LAN address, a MAC/hardware id, or a secret. Patterns are
# pattern-based on purpose: this library cannot know brand-specific shapes (access
# codes, serials, printer names), so an integration registers those via
# ``Sentry.register_scrubber`` (see ``extra_scrubbers``).
_HOME_PATH = re.compile(r"(/(?:home|Users)/)[^/\s\"']+")
_WIN_HOME = re.compile(r"([A-Za-z]:\\Users\\)[^\\\s\"']+")
#: ``scheme://user:pass@host`` -- strips embedded credentials from any URL (e.g.
#: a broker connection string), keeping the scheme so the error stays legible.
_URL_USERINFO = re.compile(r"(://)[^/\s:@]+:[^/\s@]+@")
_MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
#: ``name = value`` / ``name: value`` for clearly-secret names (separator must be
#: ``=`` or ``:`` so "key error" / "status code: 200" are not mangled).
_SECRET_KV = re.compile(
    r"(?i)\b(token|access[_\-\s]?code|api[_\-]?key|secret[_\-]?key|password|passwd|pwd|secret|authorization)"
    r"(\s*[=:]\s*)"
    r"([^\s,;&\"']+)"
)
_BEARER = re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._\-]+)")
#: A long opaque run (tokens, JWT segments, base64/hex secrets).
_LONG_RUN = re.compile(r"\b[A-Za-z0-9_\-]{40,}\b")

_SCRUB_PATTERNS: List[tuple] = [
    (_HOME_PATH, r"\1<user>"),
    (_WIN_HOME, r"\1<user>"),
    (_URL_USERINFO, r"\1<redacted>@"),
    (_MAC, "<mac>"),
    (_IPV4, "<ip>"),
    (_SECRET_KV, r"\1\2<redacted>"),
    (_BEARER, r"\1 <redacted>"),
    (_LONG_RUN, "<redacted>"),
]


class Sentry:
    """
    Configuration object for client information.
    """

    integrations: List[Integration] = []

    #: Integration-supplied redactions, applied after the built-in patterns. An
    #: integration registers brand-specific scrubbers (access codes, serials,
    #: printer names) here; the library never knows their shapes.
    extra_scrubbers: List[Callable[[str], str]] = []

    # Hash of exception + count, if the count is greater than 5, we will not send the exception.
    __seen_exceptions = dict()

    @classmethod
    def add_integration(cls, integration: Integration):
        if cls.is_initialized():
            raise RuntimeError("Cannot add integrations after sentry is initialized")

        cls.integrations.append(integration)

    @classmethod
    def register_scrubber(cls, scrubber: Callable[[str], str]) -> None:
        """Register a brand-specific text redaction run on every event field.

        ``scrubber`` takes a string and returns it with brand-specific PII
        removed (e.g. an 8-char access code or a printer serial). It composes
        after the library's generic patterns.
        """
        cls.extra_scrubbers.append(scrubber)

    @classmethod
    def is_initialized(cls):
        return sentry_sdk.Hub.current.client is not None

    @classmethod
    def initialize_sentry(cls, settings: "ClientSettings"):
        if settings.sentry_dsn is None:
            return

        # Only report from real (managed/production) installs. Source and dev
        # runs must never send events -- their noise is not actionable.
        if settings.development:
            return

        if cls.is_initialized():
            return

        # Capture nothing as breadcrumbs (the INFO firehose), only ERROR+ as
        # events -- with their tracebacks intact.
        cls.add_integration(
            LoggingIntegration(
                level=None,
                event_level=logging.ERROR,
            )
        )

        cls.add_integration(ThreadingIntegration(propagate_hub=True))
        cls.add_integration(AsyncioIntegration())

        try:
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                # Errors only -- no performance tracing/profiling volume.
                traces_sample_rate=0.0,
                error_sampler=cls._error_sampler,
                before_send=cls._before_send,
                send_default_pii=False,
                integrations=cls.integrations,
                release=f"{settings.name}@{settings.version}",
                environment=(
                    "production" if not settings.development else "development"
                ),
            )

            sentry_sdk.set_tag("lib_version", VERSION)

            printer_ids = set()

            for integration in settings.resolved_integrations():
                manager = settings.new_config_manager(str(integration.id))
                printer_ids.update(str(config.id) for config in manager.get_all())

            sentry_sdk.set_extra("printer_ids", ",".join(list(printer_ids)))

        except Exception as e:
            logging.exception(e)

    @classmethod
    def _scrub_text(cls, text: str) -> str:
        for pattern, repl in _SCRUB_PATTERNS:
            text = pattern.sub(repl, text)
        for scrubber in cls.extra_scrubbers:
            try:
                text = scrubber(text)
            except Exception:  # noqa: BLE001 -- a bad scrubber must not drop the event
                pass
        return text

    @classmethod
    def _scrub_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._scrub_text(value)
        if isinstance(value, dict):
            return {key: cls._scrub_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._scrub_value(item) for item in value]
        return value

    @classmethod
    def _before_send(cls, event: dict, hint: dict) -> dict:
        """Redact PII from every string field of an outgoing event."""
        try:
            return cls._scrub_value(event)
        except Exception:  # noqa: BLE001 -- never let scrubbing drop a real error
            return event

    @classmethod
    def _get_sample_rate_from_hash(cls, exception_hash: int) -> float:
        if len(cls.__seen_exceptions) > MAX_UNIQUE_EXCEPTIONS:
            # We have too many unique exceptions, we will not send any additional exceptions.
            return 0.0

        seen_times = cls.__seen_exceptions.get(exception_hash, 0)
        cls.__seen_exceptions[exception_hash] = seen_times + 1

        # Based on the sample rate, we will send the exception
        # enough times to be able to see it in the logs.
        if seen_times > MAX_SAMPLES_PER_EXC:
            return 0.0

        # Always send unique exceptions
        if seen_times == 0:
            return 1.0

        return DEFAULT_SAMPLE_RATE

    @classmethod
    def _error_sampler(cls, context: dict, hint: dict) -> float:
        try:
            if "log_record" in hint:
                record: logging.LogRecord = hint["log_record"]
                return cls._get_sample_rate_from_hash(
                    hash((record.levelno, record.msg))
                )

            if "exc_info" in hint:
                exc_type, exc_value, tb = hint["exc_info"]
                traceback_string = "".join(traceback.format_tb(tb))
                return cls._get_sample_rate_from_hash(
                    hash((exc_type, exc_value, traceback_string))
                )
        except (AttributeError, Exception):
            pass

        return DEFAULT_SAMPLE_RATE
