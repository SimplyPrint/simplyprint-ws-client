import logging
import traceback
from typing import TYPE_CHECKING, List

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


class Sentry:
    """
    Configuration object for client information.
    """

    integrations: List[Integration] = []

    # Hash of exception + count, if the count is greater than 5, we will not send the exception.
    __seen_exceptions = dict()

    @classmethod
    def add_integration(cls, integration: Integration):
        if cls.is_initialized():
            raise RuntimeError("Cannot add integrations after sentry is initialized")

        cls.integrations.append(integration)

    @classmethod
    def is_initialized(cls):
        return sentry_sdk.Hub.current.client is not None

    @classmethod
    def initialize_sentry(cls, settings: "ClientSettings"):
        if settings.sentry_dsn is None:
            return

        if cls.is_initialized():
            return

        # Default integrations
        cls.add_integration(
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            )
        )

        cls.add_integration(ThreadingIntegration(propagate_hub=True))
        cls.add_integration(AsyncioIntegration())

        try:
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                enable_tracing=True,
                error_sampler=cls._error_sampler,
                traces_sampler=cls._traces_sampler,
                profiles_sampler=cls._profiles_sampler,
                integrations=cls.integrations,
                release=f"{settings.name}@{settings.version}",
                environment=(
                    "production" if not settings.development else "development"
                ),
            )

            sentry_sdk.set_tag("lib_version", VERSION)

            printer_ids = set()

            for spec in settings.resolved_client_specs():
                manager = settings.new_config_manager(spec.key)
                printer_ids.update(str(config.id) for config in manager.get_all())

            sentry_sdk.set_extra("printer_ids", ",".join(list(printer_ids)))

        except Exception as e:
            logging.exception(e)

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

    @classmethod
    def _traces_sampler(cls, context: dict) -> float:
        return DEFAULT_SAMPLE_RATE

    @classmethod
    def _profiles_sampler(cls, context: dict) -> float:
        return DEFAULT_SAMPLE_RATE
