"""A bounded, in-memory timeline for one printer job."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from simplyprint_ws_client.core.protocol.messages import JobInfoMsg
from simplyprint_ws_client.core.state import PrinterState, PrinterStatus


class JobOutcome(Enum):
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class NativeJobTerminal:
    """A printer-reported terminal event for one stable native job id."""

    native_id: str
    outcome: JobOutcome


@dataclass(frozen=True)
class NativeJobObservation:
    """The complete native job identity visible in one printer snapshot.

    ``active_id=None`` explicitly says that the snapshot has no active job.
    Passing no observation to ``PrinterClient.apply_status`` means that the
    integration cannot provide native identity and status edges remain the
    authority.
    """

    active_id: Optional[str]
    terminal: Optional[NativeJobTerminal] = None


@dataclass(frozen=True)
class StartJob:
    native_id: Optional[str]


@dataclass(frozen=True)
class FinishJob:
    native_id: Optional[str]
    outcome: Optional[JobOutcome]


JobTransition = StartJob | FinishJob


class JobEventKind(Enum):
    START = "start"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class TimelineMessage:
    version: int
    message: JobInfoMsg


@dataclass(frozen=True)
class _JobEvent:
    version: int
    kind: JobEventKind
    message: JobInfoMsg
    observed_at: float
    native_id: Optional[str]

    def pending_message(self) -> TimelineMessage:
        data = dict(self.message.data or {})
        delay = round(time.monotonic() - self.observed_at)
        if delay > 0:
            data["delay"] = delay
        return TimelineMessage(self.version, JobInfoMsg(data=data))


class JobTimeline:
    """Reconcile and retain the lifecycle of at most one printer job.

    Progress remains latest-value reactive state. Lifecycle edges cannot be
    collapsed that way: a job may start and finish while the SimplyPrint socket
    is unavailable, so start and terminal each have one explicit slot.
    """

    def __init__(self) -> None:
        self._start: Optional[_JobEvent] = None
        self._terminal: Optional[_JobEvent] = None
        self._active = False
        self._native_id: Optional[str] = None

    @property
    def has_pending(self) -> bool:
        return self._start is not None or self._terminal is not None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def native_id(self) -> Optional[str]:
        return self._native_id

    def guard_status(
        self,
        previous_status: Optional[PrinterStatus],
        new_status: PrinterStatus,
        observation: Optional[NativeJobObservation],
    ) -> PrinterStatus:
        """Keep an observed continuing job from transiently becoming idle."""
        if observation is None or not self._active:
            return new_status

        if self._native_id is None and observation.active_id is not None:
            self._native_id = observation.active_id

        terminal = observation.terminal
        terminal_matches = terminal is not None and (
            self._native_id is None or terminal.native_id == self._native_id
        )
        if (
            previous_status is not None
            and PrinterStatus.is_printing(previous_status)
            and new_status == PrinterStatus.OPERATIONAL
            and observation.active_id is not None
            and observation.active_id == self._native_id
            and not terminal_matches
        ):
            return previous_status
        return new_status

    def transitions(
        self,
        previous_status: Optional[PrinterStatus],
        new_status: PrinterStatus,
        observation: Optional[NativeJobObservation],
    ) -> tuple[JobTransition, ...]:
        status_started = (
            previous_status is not None
            and not PrinterStatus.is_printing(previous_status)
            and PrinterStatus.is_printing(new_status)
        )
        status_finished = (
            previous_status is not None
            and PrinterStatus.is_printing(previous_status)
            and new_status in (PrinterStatus.OPERATIONAL, PrinterStatus.OFFLINE)
        )

        if observation is None:
            if status_started:
                return (StartJob(None),)
            if status_finished:
                return (FinishJob(self._native_id, None),)
            return ()

        active_id = observation.active_id
        terminal = observation.terminal

        if not self._active:
            return (StartJob(active_id),) if status_started else ()

        tracked_id = self._native_id
        if tracked_id is None and active_id is not None:
            self._native_id = tracked_id = active_id

        terminal_matches = terminal is not None and (
            tracked_id is None or terminal.native_id == tracked_id
        )
        identity_changed = (
            active_id is not None and tracked_id is not None and active_id != tracked_id
        )
        ended = (
            terminal_matches
            or identity_changed
            or (active_id is None and status_finished)
        )

        if not ended:
            return ()

        outcome = terminal.outcome if terminal_matches and terminal else None
        terminal_id = (
            terminal.native_id if terminal_matches and terminal else tracked_id
        )
        transitions: list[JobTransition] = [FinishJob(terminal_id, outcome)]
        if identity_changed:
            transitions.append(StartJob(active_id))
        return tuple(transitions)

    def record_start(
        self,
        state: PrinterState,
        version: int,
        native_id: Optional[str],
    ) -> None:
        if self._start is not None:
            raise RuntimeError("a second job started before the first start was sent")
        self._active = True
        self._native_id = native_id
        self._start = self._event(state, version, JobEventKind.START, native_id)

    def record_terminal(
        self,
        state: PrinterState,
        version: int,
        native_id: Optional[str],
    ) -> None:
        if self._terminal is not None:
            raise RuntimeError("a second job ended before the first terminal was sent")
        self._active = False
        self._native_id = None
        self._terminal = self._event(state, version, JobEventKind.TERMINAL, native_id)

    @staticmethod
    def _event(
        state: PrinterState,
        version: int,
        kind: JobEventKind,
        native_id: Optional[str],
    ) -> _JobEvent:
        data = dict(JobInfoMsg.build(state))
        if kind == JobEventKind.START and state.job_info.filename is not None:
            data["filename"] = state.job_info.filename
        return _JobEvent(
            version=version,
            kind=kind,
            message=JobInfoMsg(data=data),
            observed_at=time.monotonic(),
            native_id=native_id,
        )

    def pending_messages(self) -> tuple[TimelineMessage, ...]:
        events = tuple(
            event for event in (self._start, self._terminal) if event is not None
        )
        return tuple(
            event.pending_message()
            for event in sorted(events, key=lambda event: event.version)
        )

    def commit(self, version: int) -> None:
        if self._start is not None and self._start.version == version:
            self._start = None
            return
        if self._terminal is not None and self._terminal.version == version:
            self._terminal = None
            return
        raise RuntimeError(f"unknown job timeline version {version}")

    def reset(self) -> None:
        self._start = None
        self._terminal = None
        self._active = False
        self._native_id = None
