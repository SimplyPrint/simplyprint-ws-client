__all__ = [
    "Client",
    "ClientConfigChangedEvent",
    "ClientStateChangeEvent",
    "ClientState",
    "PeripheralDefinitionEntry",
    "PeripheralDefinitions",
]

import asyncio
import contextlib
import logging
import random
import re
import weakref
from datetime import timedelta, datetime
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

try:
    from typing import NotRequired, TypedDict, Unpack
except ImportError:
    from typing_extensions import NotRequired, TypedDict, Unpack

from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.state import (
    Interval,
    PrinterState,
    NotificationEvent,
    NotificationEventKwargs,
)
from simplyprint_ws_client.core.protocol.connection import ConnectionMode
from simplyprint_ws_client.core.protocol.events import (
    SimplyPrintConnectionOutgoingEvent,
    SimplyPrintConnectionEstablishedEvent,
    SimplyPrintConnectionLostEvent,
    SimplyPrintConnectionIncomingEvent,
)
from simplyprint_ws_client.core.protocol.messages import (
    SetMaterialDataDemandData,
    MaterialDataMsg,
    PeripheralDefinitionsMsg,
    PeripheralMsg,
    MultiPrinterRemoveConnectionMsg,
    MultiPrinterRemovedMsg,
    MultiPrinterAddedMsg,
    PingMsg,
    PongMsg,
    PrinterSettingsMsg,
    StreamReceivedMsg,
    IntervalChangeMsg,
    CompleteSetupMsg,
    NewTokenMsg,
    ErrorMsg,
    ConnectedMsg,
    FileDemandData,
    RefreshMaterialDataDemandData,
    RefreshPeripheralsDemandData,
    WebcamSnapshotDemandData,
    ClientMsg,
    ServerMsgKind,
    MultiPrinterAddConnectionMsg,
    MachineDataMsg,
    WebcamStatusMsg,
    WebcamMsg,
    FirmwareMsg,
    FirmwareWarningMsg,
    ToolMsg,
    TemperatureMsg,
    AmbientTemperatureMsg,
    StateChangeMsg,
    JobInfoMsg,
    LatencyMsg,
    FileProgressMsg,
    FilamentSensorMsg,
    PowerControllerMsg,
    CpuInfoMsg,
    NotificationMsg,
    ResolveNotificationDemandData,
)
from simplyprint_ws_client.core.protocol.models import (
    ServerMsgType,
    ClientMsgType,
    DispatchMode,
    DemandMsgType,
    WriteOutcome,
)
from simplyprint_ws_client.core.state.models import FileProgressStateEnum
from simplyprint_ws_client.core.state.notifications import (
    NotificationEventPayload,
    NotificationsState,
)
from simplyprint_ws_client.core.state.printer import (
    AmbientTemperatureState,
    BedState,
    ChamberState,
    CpuInfoState,
    FileProgressState,
    JobInfoState,
    MaterialEntry,
    MaterialLayoutEntry,
    PeripheralsState,
    PeripheralState,
    PingPongState,
    PrinterFilamentSensorState,
    PrinterFirmwareState,
    PrinterFirmwareWarning,
    PrinterInfoState,
    PSUState,
    TemperatureState,
    ToolState,
    WebcamSettings,
    WebcamState,
)

from simplyprint_ws_client.events import EventBus, Event
from simplyprint_ws_client.events.event import sync_only
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.common.logging import printer_logger
from simplyprint_ws_client.common.model.reactive import ReactiveModel
from simplyprint_ws_client.common.utils.backoff import Backoff, ExponentialBackoff

if TYPE_CHECKING:
    from simplyprint_ws_client.common.asyncio.offload import Offload


class ClientState(IntEnum):
    """
    CONNECTING -> Connection not established, pending establishing.
    NOT_CONNECTED -> Connection established, not connected.
    PENDING_ADDED -> Connection established, pending add request.
    PENDING_REMOVE -> Connection established, pending remove request.
    CONNECTED -> Connection established, connected.

    These are the only states the client cares about, as we are interested in our *protocol* state.
    """

    CONNECTING = 0
    NOT_CONNECTED = 1
    PENDING_ADDED = 2
    PENDING_REMOVED = 3
    CONNECTED = 4


class VersionedState(NamedTuple):
    v: int
    s: ClientState


class ClientConfigChangedEvent(Event): ...


@sync_only
class ClientStateChangeEvent(Event): ...


class PeripheralDefinitionEntry(TypedDict):
    f: Literal["r", "w", "rw"]
    n: NotRequired[str]
    d: NotRequired[str]
    o: NotRequired[int]
    vt: NotRequired[Literal["bool", "percent"]]


PeripheralDefinitions = Dict[str, PeripheralDefinitionEntry]


TConfig = TypeVar("TConfig", bound=PrinterConfig)


def _field_version(
    model: ReactiveModel,
    field: str,
    nested: Iterable[ReactiveModel] = (),
) -> Optional[int]:
    """Latest change for one explicit state branch: ``model.field``'s own
    change, and any change in the ``nested`` subtrees."""
    latest = model.model_self_changed_fields.get(field)
    for child in nested:
        child_latest = child.model_latest_change()
        if child_latest is not None and (latest is None or child_latest > latest):
            latest = child_latest
    return latest


_Versions = Callable[[PrinterState], Iterable[Optional[int]]]

#: Every message kind built from the printer state, in build order (a tie in
#: version goes to the earlier kind), with the versions of the concrete state
#: branches it is built from (``None``: that branch has no change).
_MESSAGE_KINDS: Tuple[Tuple[Type[ClientMsg], _Versions], ...] = (
    (MachineDataMsg, lambda s: (_field_version(s, "info", (s.info,)),)),
    (WebcamStatusMsg, lambda s: (_field_version(s.webcam_info, "connected"),)),
    (
        WebcamMsg,
        lambda s: (_field_version(s, "webcam_settings", (s.webcam_settings,)),),
    ),
    (FirmwareMsg, lambda s: (_field_version(s, "firmware", (s.firmware,)),)),
    (
        FirmwareWarningMsg,
        lambda s: (_field_version(s, "firmware_warning", (s.firmware_warning,)),),
    ),
    (
        ToolMsg,
        lambda s: (_field_version(tool, "active_material") for tool in s.tools),
    ),
    (
        TemperatureMsg,
        lambda s: (
            _field_version(s.bed, "temperature", (s.bed.temperature,)),
            _field_version(s.chamber, "temperature", (s.chamber.temperature,)),
            *(
                _field_version(tool, "temperature", (tool.temperature,))
                for tool in s.tools
            ),
        ),
    ),
    (
        AmbientTemperatureMsg,
        lambda s: (_field_version(s.ambient_temperature, "ambient"),),
    ),
    (StateChangeMsg, lambda s: (_field_version(s, "status"),)),
    (JobInfoMsg, lambda s: (_field_version(s, "job_info", (s.job_info,)),)),
    (LatencyMsg, lambda s: (_field_version(s.latency, "pong"),)),
    (
        FileProgressMsg,
        lambda s: (_field_version(s, "file_progress", (s.file_progress,)),),
    ),
    (
        FilamentSensorMsg,
        lambda s: (_field_version(s, "filament_sensor", (s.filament_sensor,)),),
    ),
    (PowerControllerMsg, lambda s: (_field_version(s, "psu_info", (s.psu_info,)),)),
    (CpuInfoMsg, lambda s: (_field_version(s, "cpu_info", (s.cpu_info,)),)),
    (
        MaterialDataMsg,
        lambda s: (
            *(_field_version(tool, "materials", tool.materials) for tool in s.tools),
            *(_field_version(tool, "size") for tool in s.tools),
            *(_field_version(tool, "type") for tool in s.tools),
            *(_field_version(tool, "volume_type") for tool in s.tools),
            _field_version(s.bed, "type"),
            _field_version(s, "mms_layout", s.mms_layout),
        ),
    ),
    (
        NotificationMsg,
        lambda s: (
            _field_version(
                s.notifications, "notifications", s.notifications.notifications.values()
            ),
        ),
    ),
    (
        PeripheralMsg,
        lambda s: (
            _field_version(s.peripherals, "entries", s.peripherals.entries.values()),
        ),
    ),
)


def _message_changes(
    state: PrinterState,
) -> Iterable[Tuple[Type[ClientMsg], Iterable[Optional[int]]]]:
    """Messages affected by the concrete branches of this state tree.

    The full scan, every kind: the spec. ``consume`` reads only the kinds a
    change can have reached (:func:`_affected`), which tests hold to this.
    """
    for kind, versions in _MESSAGE_KINDS:
        yield kind, versions(state)


#: One bit per message kind, in ``_MESSAGE_KINDS`` order.
_KIND_BIT: Dict[Type[ClientMsg], int] = {
    kind: 1 << index for index, (kind, _) in enumerate(_MESSAGE_KINDS)
}
_ALL_KINDS = (1 << len(_MESSAGE_KINDS)) - 1

_Kinds = Tuple[Type[ClientMsg], ...]

#: The inverse of ``_MESSAGE_KINDS``, per state model class: the kinds a change
#: to *any* of its fields can affect (the class sits inside a branch whose whole
#: subtree is read), and the kinds a change to *one* field can affect (that
#: field is read on its own). A field in neither feeds no message.
#:
#: Read by class, so it relies on each state model sitting where its annotation
#: puts it. A subclass inherits its bases' entries; a model class with no entry
#: (an integration's own) affects every kind: unknown costs a full scan, never
#: a missed message.
_AFFECTS: Dict[type, Tuple[_Kinds, Dict[str, _Kinds]]] = {
    PrinterState: (
        (),
        {
            "status": (StateChangeMsg,),
            "info": (MachineDataMsg,),
            "webcam_settings": (WebcamMsg,),
            "firmware": (FirmwareMsg,),
            "firmware_warning": (FirmwareWarningMsg,),
            "job_info": (JobInfoMsg,),
            "file_progress": (FileProgressMsg,),
            "filament_sensor": (FilamentSensorMsg,),
            "psu_info": (PowerControllerMsg,),
            "cpu_info": (CpuInfoMsg,),
            "mms_layout": (MaterialDataMsg,),
            # Read through the child only, but a swapped-in child brings the
            # changes it carries.
            "bed": (TemperatureMsg, MaterialDataMsg),
            "chamber": (TemperatureMsg,),
            "tools": (ToolMsg, TemperatureMsg, MaterialDataMsg),
            "ambient_temperature": (AmbientTemperatureMsg,),
            "latency": (LatencyMsg,),
            "notifications": (NotificationMsg,),
            "peripherals": (PeripheralMsg,),
            "webcam_info": (WebcamStatusMsg,),
        },
    ),
    PrinterInfoState: ((MachineDataMsg,), {}),
    WebcamState: ((), {"connected": (WebcamStatusMsg,)}),
    WebcamSettings: ((WebcamMsg,), {}),
    PrinterFirmwareState: ((FirmwareMsg,), {}),
    PrinterFirmwareWarning: ((FirmwareWarningMsg,), {}),
    TemperatureState: ((TemperatureMsg,), {}),
    BedState: ((), {"temperature": (TemperatureMsg,), "type": (MaterialDataMsg,)}),
    ChamberState: ((), {"temperature": (TemperatureMsg,)}),
    ToolState: (
        (),
        {
            "active_material": (ToolMsg,),
            "temperature": (TemperatureMsg,),
            "materials": (MaterialDataMsg,),
            "size": (MaterialDataMsg,),
            "type": (MaterialDataMsg,),
            "volume_type": (MaterialDataMsg,),
        },
    ),
    MaterialEntry: ((MaterialDataMsg,), {}),
    MaterialLayoutEntry: ((MaterialDataMsg,), {}),
    AmbientTemperatureState: ((), {"ambient": (AmbientTemperatureMsg,)}),
    JobInfoState: ((JobInfoMsg,), {}),
    PingPongState: ((), {"pong": (LatencyMsg,)}),
    FileProgressState: ((FileProgressMsg,), {}),
    PrinterFilamentSensorState: ((FilamentSensorMsg,), {}),
    PSUState: ((PowerControllerMsg,), {}),
    CpuInfoState: ((CpuInfoMsg,), {}),
    NotificationsState: ((), {"notifications": (NotificationMsg,)}),
    NotificationEvent: ((NotificationMsg,), {}),
    NotificationEventPayload: ((NotificationMsg,), {}),
    PeripheralsState: ((), {"entries": (PeripheralMsg,)}),
    PeripheralState: ((PeripheralMsg,), {}),
}

#: ``(class, field)`` -> kind bits; computed on first use.
_AFFECTED: Dict[Tuple[type, Optional[str]], int] = {}


def _affected(cls: type, field: Optional[str]) -> int:
    """The kinds (bits) a change to ``field`` of a ``cls`` model can affect.

    ``field=None``: a change anywhere in such a model's subtree (a bind).
    """
    key = (cls, field)
    bits = _AFFECTED.get(key)
    if bits is None:
        bits = _AFFECTED[key] = _compute_affected(cls, field)
    return bits


def _compute_affected(cls: type, field: Optional[str]) -> int:
    known = [base for base in cls.__mro__ if base in _AFFECTS]
    if not known:
        return _ALL_KINDS
    bits = 0
    for base in known:
        any_field, by_field = _AFFECTS[base]
        kinds = list(any_field)
        if field is None:
            for field_kinds in by_field.values():
                kinds.extend(field_kinds)
        else:
            kinds.extend(by_field.get(field, ()))
        for kind in kinds:
            bits |= _KIND_BIT[kind]
    return bits


def _mark_present(model: ReactiveModel, *fields: str) -> None:
    """Mark the fields of ``model`` that hold a value as changed."""
    present = [field for field in fields if getattr(model, field) is not None]
    if present:
        model.model_set_changed(*present)


def _resend_job_info(state: PrinterState) -> None:
    # Only a running job's values; the edges are one-shot and never re-marked.
    if state.is_printing():
        _mark_present(
            state.job_info,
            "progress",
            "initial_estimate",
            "layer",
            "time",
            "filament",
            "filename",
            "delay",
            "skipped_objects",
        )


def _resend_file_progress(state: PrinterState) -> None:
    # A transfer in progress is a value; a terminal state is an edge.
    if state.file_progress.state in (
        FileProgressStateEnum.DOWNLOADING,
        FileProgressStateEnum.STARTED,
        FileProgressStateEnum.PENDING,
    ):
        state.file_progress.model_set_changed("state")


def _resend_tools(state: PrinterState) -> None:
    for tool in state.tools:
        _mark_present(tool, "active_material")


def _resend_temperatures(state: PrinterState) -> None:
    for temperature in (
        state.bed.temperature,
        state.chamber.temperature,
        *(tool.temperature for tool in state.tools),
    ):
        _mark_present(temperature, "actual")


def _resend_materials(state: PrinterState) -> None:
    _mark_present(state.bed, "type")
    for tool in state.tools:
        tool.model_set_changed("nozzle")
        for material in tool.materials:
            material.model_set_changed("ext")
    if state.mms_layout:
        state.model_set_changed("mms_layout")


def _resend_peripherals(state: PrinterState) -> None:
    for peripheral in state.peripherals.entries.values():
        _mark_present(peripheral, "value", "available", "updated")


#: How a reconnect re-marks each value-type message so its next build carries
#: the full current value (the inverse of :func:`_message_changes`). Status,
#: ``sp_version`` and the firmware name are re-marked unconditionally by
#: ``mark_common_fields_as_changed``, as on every connect, between the two
#: tables.
#:
#: Not here: latency (a measurement of the dead link) and notifications
#: (events, not values; re-issuing them is not known to be harmless).
_Resend = Tuple[Tuple[Type[ClientMsg], Callable[[PrinterState], None]], ...]

#: Re-marked before the status: a job edge still pending in job info must be
#: built while the job id is there, before a resent `operational` releases it.
_RESEND_BEFORE_STATUS: _Resend = (
    (JobInfoMsg, _resend_job_info),
    (FileProgressMsg, _resend_file_progress),
)

_RESEND_AFTER_STATUS: _Resend = (
    (
        MachineDataMsg,
        lambda s: _mark_present(s.info, *type(s.info).model_fields),
    ),
    (WebcamStatusMsg, lambda s: s.webcam_info.model_set_changed("connected")),
    (
        WebcamMsg,
        lambda s: s.webcam_settings.model_set_changed("flipH", "flipV", "rotate90"),
    ),
    (
        FirmwareMsg,
        lambda s: _mark_present(s.firmware, *type(s.firmware).model_fields),
    ),
    (
        FirmwareWarningMsg,
        lambda s: _mark_present(
            s.firmware_warning, "check_name", "warning_type", "severity", "url"
        ),
    ),
    (ToolMsg, _resend_tools),
    (TemperatureMsg, _resend_temperatures),
    (
        AmbientTemperatureMsg,
        lambda s: s.ambient_temperature.model_set_changed("ambient"),
    ),
    (FilamentSensorMsg, lambda s: _mark_present(s.filament_sensor, "state")),
    (PowerControllerMsg, lambda s: s.psu_info.model_set_changed("on")),
    (CpuInfoMsg, lambda s: _mark_present(s.cpu_info, "usage", "temp", "memory")),
    (MaterialDataMsg, _resend_materials),
    (PeripheralMsg, _resend_peripherals),
)

#: After a hub (re)opens, its printers' adds are spread over this many seconds
#: per printer on it: a full hub of 32 re-adds over ~1.6 s, not in one burst,
#: while a single printer waits at most 50 ms.
ADD_SPREAD_PER_PRINTER_S = 0.05


def _jitter(upper: float) -> float:
    """A uniform delay in ``[0, upper)``. Module-level so tests can pin it."""
    return random.uniform(0, upper) if upper > 0 else 0.0


class _Held(NamedTuple):
    """A message built by :meth:`Client.consume` whose reset waits for its write."""

    msg: ClientMsg
    #: The newest change it was built from: the reset clears up to this.
    v: int
    #: The job id its build released (``releases_job_id``), to restore if the
    #: frame never left; ``None`` if it released none.
    released_job_id: Optional[int]
    #: Handed over by :meth:`Client.take_held_edge`: its markers are already
    #: reset and its job id stays released; the taker resends it if need be.
    taken: bool = False


#: How long a multi-printer add or remove waits for the server's answer. The
#: server answers a remove within its own 30 s bound; a lost answer (a server
#: restart between request and reply) would otherwise leave the printer off
#: SimplyPrint for good, since only that answer moves the state on.
PENDING_REPLY_TIMEOUT = timedelta(seconds=45)


class Client(
    Generic[TConfig],
    EventLoopProvider[asyncio.AbstractEventLoop],
):
    """One printer's agent speaking the SimplyPrint cloud protocol — a scheduling unit.

    The complete cloud agent: the protocol state machine AND the default
    prioritized message handling (intervals, tokens, setup, materials,
    notifications — the old ``DefaultClient``, merged in 2.0). Opting out of a
    default is overriding the handler, not picking a thinner base class.

    Attributes:
        v: Client version
        msg_id: Current message id (incrementing)
        printer: Modifiable state of the printer.
        event_bus: Event bus for handling events.
        logger: Logger instance

        _state: Versioned state
        _should_be_allocated: Whether the client should be allocated (active)

        _pending_action_backoff: Backoff for pending actions
        _pending_action_delay: Delay for pending actions
        _pending_action_ts: Timestamp of last pending action
        _pending_action_log_ts: Timestamp of the last pending action log, so to limit log spam
    """

    printer: PrinterState
    event_bus: EventBus
    logger: logging.Logger

    _state: VersionedState
    _pending_action_backoff: Backoff

    #: The app's bounded blocking-work lanes, injected by ``ClientApp`` at
    #: construction. ``None`` outside the app (e.g. in unit tests); callers that
    #: offload must tolerate its absence (``FileTransfer`` falls back to a thread).
    offload: "Optional[Offload]"

    def __init__(
        self,
        config: TConfig,
        *,
        context: ClientContext,
    ) -> None:
        Generic.__init__(self)
        self.context = context
        EventLoopProvider.__init__(
            self,
            provider=context.event_loop_provider,
        )
        self.v = -1
        self.msg_id = -1
        #: Nesting depth of :meth:`batched_signals`, and whether a signal is owed.
        self._signals_held = 0
        self._signal_pending = False
        #: The change id when a FILE demand last set the job id (``_hold``).
        self._job_id_admitted_at: Optional[int] = None
        self.last_msg_id = -1
        #: Changes the last send left pending without a new change to wake the
        #: scheduler: rate-limited messages, frames that never left. The
        #: scheduler retries them on its next pass (see ``has_deferred``).
        self._deferred = False
        #: While set, ``consume`` leaves each message's reset to ``settle``,
        #: after the frame's write (the scheduler's path).
        self._settle_on_write = False
        self._held: Dict[int, _Held] = {}
        #: Kinds (``_KIND_BIT``) a change since the last ``consume`` can have
        #: affected, and kinds that still had a change at the last one. Every
        #: kind with a change is in one of them, so ``consume`` reads only
        #: these instead of scanning the whole tree for every kind.
        self._touched_kinds = _ALL_KINDS
        self._pending_kinds = 0
        #: The tree those describe: a replaced ``printer`` is scanned in full.
        self._scanned_printer: Optional[PrinterState] = None
        #: Message kinds the server has been sent at least once. A reconnect
        #: resends these in full; a kind never sent stays as it was (nothing
        #: is invented from defaults the server never saw).
        self._sent_kinds: Set[type] = set()
        #: Armed when an added printer loses its link; the next ``connected``
        #: then resends the full current state instead of the common fields.
        self._resend_state = False
        #: For the first send after that resend: one-shot edges kept from the
        #: dead link go out before the state.
        self._edges_first = False
        self._should_be_allocated = True
        self.initialized = False
        self.offload = context.offload
        self.simplyprint_api = context.simplyprint_api
        self._state = VersionedState(-1, ClientState.CONNECTING)
        self._pending_action_backoff = ExponentialBackoff(10, 600, 3600)
        self._pending_action_delay = timedelta.min
        self._pending_action_ts = datetime.min
        self._pending_action_log_ts = datetime.min
        self.event_bus = EventBus(event_loop_provider=self)
        self.printer = PrinterState(config=config)
        self.printer.provide_context(weakref.ref(self))
        self.logger = printer_logger(self.unique_id)
        self._register_core_handlers()

    def _register_core_handlers(self) -> None:
        """Install the protocol state machine's fixed routes.

        This table is intentionally explicit: event selection, ordering, and
        handler arity are visible here and cannot change because a method was
        renamed or received a different annotation.
        """
        on = self.event_bus.on
        on(SimplyPrintConnectionIncomingEvent, self._on_connection_incoming)
        on(SimplyPrintConnectionEstablishedEvent, self._on_connection_established)
        on(SimplyPrintConnectionLostEvent, self._on_connection_lost)
        on(ServerMsgType.ADD_CONNECTION, self._on_multi_printer_added, priority=1)
        on(
            ServerMsgType.REMOVE_CONNECTION,
            self._on_multi_printer_removed,
            priority=1,
        )
        on(ServerMsgType.CONNECTED, self._on_connected_state, priority=2)
        on(ServerMsgType.ERROR, self._on_error, priority=1)
        on(ServerMsgType.NEW_TOKEN, self._on_new_token, priority=1)
        on(ServerMsgType.CONNECTED, self._on_connected_data, priority=1)
        on(ServerMsgType.COMPLETE_SETUP, self._on_setup_complete, priority=1)
        on(ServerMsgType.INTERVAL_CHANGE, self._on_interval_change, priority=1)
        on(ServerMsgType.PONG, self._on_pong, priority=1)
        on(ServerMsgType.PRINTER_SETTINGS, self._on_printer_settings, priority=1)
        on(ServerMsgType.STREAM_RECEIVED, self._on_stream_received, priority=1)
        on(DemandMsgType.WEBCAM_SNAPSHOT, self._on_webcam_snapshot, priority=1)
        on(DemandMsgType.FILE, self._on_file_demand, priority=1)
        on(DemandMsgType.SET_MATERIAL_DATA, self._apply_material_data, priority=1)
        on(
            DemandMsgType.REFRESH_MATERIAL_DATA,
            self.on_refresh_material_data,
            priority=1,
        )
        on(
            DemandMsgType.REFRESH_PERIPHERALS,
            self.on_refresh_peripherals,
            priority=1,
        )
        on(
            DemandMsgType.RESOLVE_NOTIFICATION,
            self._on_resolve_notification,
            priority=1,
        )

    @property
    def unique_id(self) -> Union[str, int]:
        """Indicate the `identity` of a client instance related to the physical (configuration) world
        and not in-memory (instance) world, although both shall be consistent."""
        return self.printer.config.unique_id

    @property
    def config(self) -> TConfig:
        return cast(TConfig, self.printer.config)

    @property
    def active(self):
        return self._should_be_allocated

    @active.setter
    def active(self, value: bool):
        self._should_be_allocated = value
        self.signal()

    @property
    def state(self) -> ClientState:
        if self.v != self._state.v:
            return ClientState.CONNECTING

        return self._state.s

    @state.setter
    def state(self, value: ClientState):
        self._state = VersionedState(self.v, value)
        self.logger.debug(f"State changed to {self._state}")
        self.signal()

    @property
    def has_changes(self) -> bool:
        """A change was made since the last ``consume``."""
        return self.msg_id > self.last_msg_id

    @property
    def has_deferred(self) -> bool:
        """The last send left changes pending (rate limited, or not written).

        Kept apart from :attr:`has_changes` on purpose: the scheduler wakes
        for new changes at once, but retries deferred ones only on its
        tick-rate pass, so a rate-limited printer is not rebuilt on every wake.
        """
        return self._deferred

    def next_msg_id(self):
        self.msg_id += 1
        return self.msg_id

    def stamp_change(self, model: ReactiveModel, field: str) -> int:
        """The printer state's change to ``model.field``: its message id."""
        self._touched_kinds |= _affected(type(model), field)
        return self.next_msg_id()

    def state_attached(self, model: ReactiveModel) -> None:
        """A state subtree was bound to this client (``provide_context``): any
        change it already carries is now in the tree."""
        self._touched_kinds |= _affected(type(model), None)

    # coordination methods

    def is_added(self) -> bool:
        return self.state == ClientState.CONNECTED

    def is_removed(self) -> bool:
        return self.state <= ClientState.NOT_CONNECTED

    async def ensure_added(self, mode: ConnectionMode, allow_setup=False) -> bool:
        """Progress inner state based on mode protocol. Goal: Connected"""
        # For the single connection mode we do not have to perform any
        # additional actions beyond the initial connection.
        if mode == ConnectionMode.SINGLE:
            return self.state == ClientState.CONNECTED

        self._expire_pending_reply()
        if self.state == ClientState.NOT_CONNECTED and self._can_do_pending():
            self.state = ClientState.PENDING_ADDED
            await self.send(MultiPrinterAddConnectionMsg(self.config, allow_setup))
            self._do_pending()

        return self.state == ClientState.CONNECTED

    async def ensure_removed(self, mode: ConnectionMode) -> bool:
        """Ensure that the client is removed from the multi printer.
        Here the goal is different based on the connection mode.
        For single mode we can always disconnect the connection, so we just return true.
        For multi-printer mode the goal is to be removed.
        """

        # In single mode we do not need to do anything here.
        if mode == ConnectionMode.SINGLE:
            return True

        self._expire_pending_reply()
        if self.state == ClientState.CONNECTED and self._can_do_pending():
            self.state = ClientState.PENDING_REMOVED
            await self.send(MultiPrinterRemoveConnectionMsg(self.config))
            self._do_pending()

        return self.state <= ClientState.NOT_CONNECTED

    def _expire_pending_reply(self) -> None:
        """An add or remove the server never answered counts as not connected
        after :data:`PENDING_REPLY_TIMEOUT`: the next pass adds the printer
        again (paced by the pending-action backoff), and the server's
        newest-wins supersession retires whatever it still holds."""
        state = self.state
        if state not in (ClientState.PENDING_ADDED, ClientState.PENDING_REMOVED):
            return
        if datetime.now() - self._pending_action_ts < PENDING_REPLY_TIMEOUT:
            return
        self.logger.warning(
            "No answer to %s within %ss; treating the printer as not connected",
            "add_connection" if state == ClientState.PENDING_ADDED else "remove_connection",
            int(PENDING_REPLY_TIMEOUT.total_seconds()),
        )
        self.state = ClientState.NOT_CONNECTED

    def _can_do_pending(self):
        now = datetime.now()
        time_since = now - self._pending_action_ts

        can_do_pending = time_since > self._pending_action_delay

        if not can_do_pending and now - self._pending_action_log_ts > (
            self._pending_action_delay / 3
        ):
            self._pending_action_log_ts = now
            time_remaining = self._pending_action_delay - time_since
            self.logger.debug(
                "Cannot do current pending action. Time remaining: %s", time_remaining
            )

        return can_do_pending

    def _do_pending(self):
        self._pending_action_ts = datetime.now()
        self._pending_action_delay = timedelta(
            seconds=self._pending_action_backoff.delay()
        )

    def signal(self):
        if self._signals_held:
            # Inside batched_signals(): one emit when the batch ends.
            self._signal_pending = True
            return
        self.event_bus.emit_sync(ClientStateChangeEvent)

    @contextlib.contextmanager
    def batched_signals(self) -> Iterator[None]:
        """Within: state changes signal once, when the outermost batch ends.

        Every changed field signals the scheduler through the event bus; an
        integration that projects a whole report at once (many fields) needs
        the scheduler woken once for it, not once per field. Nothing reads
        the signal in between: the scheduler only runs after the batch."""
        self._signals_held += 1
        try:
            yield
        finally:
            self._signals_held -= 1
            if not self._signals_held and self._signal_pending:
                self._signal_pending = False
                self.event_bus.emit_sync(ClientStateChangeEvent)

    def consume(self) -> list:
        """Consume and return the list of pending messages.

        Each message's change markers are reset as it is built, unless
        :meth:`settling_on_write` is active: then the reset waits for
        :meth:`settle`, after the frame's write, so a change whose frame never
        left stays pending. Either way the messages built are the same.
        """
        self.last_msg_id = self.msg_id
        self._deferred = False
        self._release_held()

        msg_kinds = self._changed_kinds()

        is_pending = self.printer.config.is_pending()

        msgs = []

        # Sort by the lowest version.
        for msg_kind, (lowest, highest) in sorted(
            msg_kinds.items(), key=lambda item: item[1][0]
        ):
            # Skip over messages that are not allowed to be sent when pending.
            if is_pending and not msg_kind.msg_type().when_pending():
                continue

            data = dict(msg_kind.build(self.printer))

            if not data:
                continue

            msg = msg_kind(data)

            # Skip over messages that are not supposed to be sent.
            dispatch_mode = msg.dispatch_mode(self.printer)
            if dispatch_mode != DispatchMode.DISPATCH:
                # A rate-limited change keeps its markers. Without a new change
                # nothing would wake the scheduler, and the last value of a
                # burst would go stale; deferred, the next pass sends it.
                if dispatch_mode == DispatchMode.RATELIMIT:
                    self._deferred = True
                continue

            msgs.append(msg)
            if self._settle_on_write:
                self._hold(msg, highest)
            else:
                msg.reset_changes(self.printer, v=highest)
                self._sent_kinds.add(msg_kind)

        if self._edges_first:
            # The first send after a reconnect: edges kept from the dead link
            # go before the full-state resend, so a resent status cannot start
            # the server's job-disruption clock ahead of a kept `finished`.
            self._edges_first = False
            msgs.sort(key=lambda m: not m.carries_edge())  # stable

        return msgs

    def _changed_kinds(self) -> Dict[Type[ClientMsg], Tuple[int, int]]:
        """Each message kind with a change: its lowest and highest version.

        What the full scan (:func:`_message_changes`) finds, in its order, but
        read only for the kinds a change since the last call can have reached
        and the kinds that still had a change then (rate limited, held, built
        empty, never reset). Any other kind had none and gained none, so after
        one temperature change this reads the temperatures, not the tree.
        """
        printer = self.printer
        bits = self._touched_kinds | self._pending_kinds
        if printer is not self._scanned_printer:
            bits = _ALL_KINDS
            self._scanned_printer = printer
        self._touched_kinds = 0

        msg_kinds = {}
        pending = 0
        while bits:
            bit = bits & -bits  # the lowest: kinds stay in scan order
            bits ^= bit
            msg_kind, candidates = _MESSAGE_KINDS[bit.bit_length() - 1]
            if versions := tuple(v for v in candidates(printer) if v is not None):
                msg_kinds[msg_kind] = (min(versions), max(versions))
                pending |= bit

        # Kept until a later call finds them clear: whether this send resets
        # them is up to its build, dispatch and settle.
        self._pending_kinds = pending
        return msg_kinds

    @property
    def settles_on_write(self) -> bool:
        """Inside :meth:`settling_on_write`: whatever ``consume`` hands out is
        settled after its write. A subclass queueing its own messages commits
        them in its :meth:`settle` override then, and at once otherwise."""
        return self._settle_on_write

    @contextlib.contextmanager
    def settling_on_write(self) -> Iterator[None]:
        """Within: ``consume`` defers each reset to :meth:`settle`.

        A context rather than a ``consume`` argument, so a subclass that
        overrides ``consume()`` (and calls ``super()``) keeps working.
        """
        self._settle_on_write = True
        try:
            yield
        finally:
            self._settle_on_write = False

    def _hold(self, msg: ClientMsg, v: int) -> None:
        released = None
        # A job id admitted (FILE demand) after this message's change belongs
        # to the next job, not the one this status or end edge closes. A send
        # that stopped at a failed write builds the rest later, after such a
        # demand may have come in; the default path never interleaves.
        admitted = self._job_id_admitted_at
        if msg.releases_job_id(self.printer) and (admitted is None or v > admitted):
            # The default path clears the job id in reset_changes, before the
            # next message is built; doing it here keeps every later message
            # of this send built exactly as before.
            released = self.printer.current_job_id
            self.printer.current_job_id = None
        self._held[id(msg)] = _Held(msg, v, released)

    def _release_held(self) -> None:
        """Forget messages a previous send never settled (it was aborted).

        Their markers are still pending, so the state resends them; only a job
        id their build released has to come back.
        """
        for held in self._held.values():
            if (
                not held.taken
                and held.released_job_id is not None
                and self.printer.current_job_id is None
            ):
                self.printer.current_job_id = held.released_job_id
        self._held.clear()

    def take_held_edge(self, msg_kind: Type[ClientMsg]) -> Optional[ClientMsg]:
        """Hand over the edge of ``msg_kind`` a send is writing right now.

        For an integration about to overwrite what an in-flight edge was built
        from (a newer edge, a new job id). Its markers stay set until the write
        settles, so it looks unsent: a copy of it would go out a second time
        once the write lands, and leaving it to :meth:`settle` would lose it if
        the frame never left, the state having moved on.

        Taken, its markers are reset now (as for a written frame), the job id
        it released stays released (it travels in the frame; restoring it
        could put an old run's id on the next one), and :meth:`settle` leaves
        it to the caller: queue it, and drop it once it counts as sent.
        ``None`` if no such edge is in flight (or it was taken already).
        """
        for key, held in self._held.items():
            msg = held.msg
            if held.taken or not isinstance(msg, msg_kind) or not msg.carries_edge():
                continue
            job_id = self.printer.current_job_id
            msg.reset_changes(self.printer, v=held.v)
            self.printer.current_job_id = job_id
            self._held[key] = held._replace(taken=True)
            return msg
        return None

    def settle(self, msg: ClientMsg) -> None:
        """After the write of a message built by ``consume``: reset or keep.

        - written (or nobody reported, as for a sink that is not the protocol):
          reset, as the default path does at build time;
        - never left: keep pending, so it goes out on the next send. For an edge
          this is what makes it survive a dead socket;
        - may have left: an edge is reset (at-most-once: no resend without
          acknowledgements); a value is kept, since resending it is harmless;
        - cannot be serialized: reset, so it does not retry forever.

        Messages ``consume`` did not build (an integration's own) are left to
        the integration, and so is one handed over by :meth:`take_held_edge`.
        Any frame that does not count as sent still asks for the retry pass.
        """
        sent = msg.counts_as_sent()
        if not sent:
            # Also for messages this class did not build: the integration that
            # queued one keeps it (the v0 outbox does), and with no change in
            # the state nothing else would bring the scheduler back for it.
            self._deferred = True

        held = self._held.pop(id(msg), None)
        if held is None:
            return

        outcome = msg.write_outcome
        if held.taken:
            # Handed over (take_held_edge): reset then, resent by the taker.
            # Only the record of what the server was sent is still ours.
            if sent and outcome != WriteOutcome.UNSENDABLE:
                self._sent_kinds.add(type(msg))
            return

        if not sent:
            if held.released_job_id is not None and self.printer.current_job_id is None:
                self.printer.current_job_id = held.released_job_id
            return

        # The job id was settled when the message was built (see _hold); a
        # newer one may have been set while the frame was in flight.
        job_id = self.printer.current_job_id
        msg.reset_changes(self.printer, v=held.v)
        self.printer.current_job_id = job_id
        if outcome != WriteOutcome.UNSENDABLE:
            self._sent_kinds.add(type(msg))

    async def write(self, msg: ClientMsg) -> None:
        """The scheduler's send of one message built by ``consume``: write it,
        then :meth:`settle` it on what became of the frame."""
        try:
            await self.send(msg, skip_dispatch=True)
        except BaseException:
            if msg.write_outcome is None:
                # Failed before it reached the protocol: it never left.
                msg.record_write(WriteOutcome.NOT_SENT)
            self.settle(msg)
            # The rest of this send is not written either.
            self._deferred = True
            raise
        self.settle(msg)

    def unwritten(self, msgs: Iterable[ClientMsg]) -> None:
        """The rest of a send the scheduler stopped at a failed write: none of
        it was offered to the link. Each is settled as never sent, so it keeps
        its markers (or its integration's queue position) for the next send."""
        for msg in msgs:
            msg.record_write(WriteOutcome.NOT_SENT)
            self.settle(msg)

    # internal methods

    async def _on_connection_incoming(self, msg: ServerMsgKind, v: int):
        if self.v > v:
            self.logger.warning("Dropped incoming message %s with v: %d.", msg, v)
            return

        if self.v != v:
            previous_v = self.v
            self.v = v
            self.logger.warning(
                "Upgraded client connection version from %d to %d due to new message.",
                previous_v,
                v,
            )

        if msg.type == ServerMsgType.DEMAND:
            event = msg.data.demand
            args = (msg.data,)
        else:
            event = msg.type
            args = (msg,)

        # Preserve the protocol's single FIFO consumer. Handlers that start
        # long-running work must admit it into their own bounded/coalescing
        # subsystem and return quickly (camera and file transfer both do so).
        # Creating a task per wire message hides violations of that contract and
        # makes the application's total task count unbounded.
        await self.event_bus.emit(event, *args)

    def _on_connection_established(self, event: SimplyPrintConnectionEstablishedEvent):
        self.v = event.v

        if self.state == ClientState.CONNECTING:
            self.state = ClientState.NOT_CONNECTED
            if event.hub_size is not None:
                self.spread_add(event.hub_size)

    def spread_add(self, hub_size: int) -> None:
        """Hold this printer's next add for a random share of the hub's window.

        A hub that (re)opens would otherwise send all its adds in one burst;
        spread over ``ADD_SPREAD_PER_PRINTER_S`` per printer they stay well
        under the server's add limits. Never shortens a longer wait already
        set (a server ``Retry-After``).
        """
        delay = _jitter(ADD_SPREAD_PER_PRINTER_S * hub_size)
        if delay <= 0:
            return

        now = datetime.now()
        if self._pending_action_ts != datetime.min:
            remaining = self._pending_action_delay - (now - self._pending_action_ts)
            if remaining.total_seconds() >= delay:
                return

        self._pending_action_ts = now
        self._pending_action_delay = timedelta(seconds=delay)
        # Wake the scheduler when the hold ends, not at its next tick.
        try:
            asyncio.get_running_loop().call_later(delay, self.signal)
        except RuntimeError:
            pass  # no loop (sync tests): the scheduler's tick picks it up

    def _on_connection_lost(self, event: SimplyPrintConnectionLostEvent):
        if self.v > event.v:
            return

        # An added printer losing its link is a failure: the server may have
        # missed anything since, so the next `connected` resends the full
        # state. A deliberate removal (deallocate of an inactive printer) is
        # not: it is removed first, or inactive.
        if self.active and self.state == ClientState.CONNECTED:
            self._resend_state = True

        # handle connection lost.
        self.v = event.v
        self._pending_action_backoff.reset()
        self.state = ClientState.CONNECTING
        self.signal()

    # important functional event handling

    async def _on_multi_printer_added(self, msg: MultiPrinterAddedMsg):
        if not msg.data.status:
            retry_after = self._apply_server_retry_after(msg.data.reason)
            if retry_after is None:
                self.logger.warning("Failed to add connection. %s", msg)
            else:
                self.logger.warning(
                    "Failed to add connection; server requested retry in %ss. %s",
                    retry_after,
                    msg,
                )
            self.state = ClientState.NOT_CONNECTED
            self.signal()
            return

        # A successful addition does not require a backoff.
        self._pending_action_backoff.reset()
        self.config.id = msg.data.pid
        self.state = ClientState.CONNECTED
        self.signal()

    def _apply_server_retry_after(self, reason: Optional[str]) -> Optional[float]:
        if not reason:
            return None

        match = re.search(r"retry-after:\s*(\d+(?:\.\d+)?)s", reason, re.IGNORECASE)
        if match is None:
            return None

        retry_after = float(match.group(1))
        self._pending_action_ts = datetime.now()
        self._pending_action_delay = timedelta(seconds=retry_after)
        return retry_after

    async def _on_multi_printer_removed(self, msg: MultiPrinterRemovedMsg):
        self.logger.debug("Connection removed. %s", msg)
        self.state = ClientState.NOT_CONNECTED
        self.signal()

    async def _on_connected_state(self, _msg: ConnectedMsg):
        if self._resend_state:
            self._resend_state = False
            self._mark_full_state()
        else:
            self.printer.mark_common_fields_as_changed()
        self.state = ClientState.CONNECTED
        self.signal()

    def _mark_full_state(self) -> None:
        """A reconnect: re-mark every value-type message the server has been
        sent, so the next send carries the full current state (newest wins).

        Edges are never re-marked: one still pending never left the dead link
        and goes first (``_edges_first``); one that may have left stays sent.
        """
        for kind, mark in _RESEND_BEFORE_STATUS:
            if kind in self._sent_kinds:
                mark(self.printer)
        self.printer.mark_common_fields_as_changed()
        for kind, mark in _RESEND_AFTER_STATUS:
            if kind in self._sent_kinds:
                mark(self.printer)
        self._edges_first = True

    async def send(self, msg: ClientMsg[ClientMsgType], skip_dispatch=False):
        """External send method (applies dispatch mode)."""
        # check dispatch mode + use interval (automatically)

        if (
            not skip_dispatch
            and (dispatch_mode := msg.dispatch_mode(self.printer))
            != DispatchMode.DISPATCH
        ):
            self.logger.warning(
                "Dropped message %s due to dispatch mode %s.", msg, dispatch_mode
            )
            return

        await self.event_bus.emit(SimplyPrintConnectionOutgoingEvent, msg, self.v)

    # lifetime methods

    async def init(self):
        """Init lifecycle method. Called exactly once when the client enters
        scheduling -- before the first tick and regardless of ``active``. Arm
        device-side machinery here; it runs until :meth:`teardown`."""
        pass

    async def tick(self, delta: timedelta):
        """Tick lifecycle method. Called at the tick rate for every scheduled
        client, whether or not it is allocated/added to SimplyPrint -- guard
        SimplyPrint-side sends on :meth:`is_added` where it matters."""
        pass

    async def halt(self):
        """Halt lifecycle method: the client was deallocated from SimplyPrint
        (``active`` flipped false). SimplyPrint-side parking only -- device-side
        work keeps running (and keeps ticking) so it can reactivate the client."""
        pass

    async def teardown(self):
        """Teardown lifecycle method, final cleanup, will never be needed again."""
        pass

    # file/job bookkeeping

    @property
    def current_job_id(self):
        return self.printer.current_job_id

    @property
    def file_action_token(self):
        return self.printer.file_action_token

    async def send_ping(self) -> None:
        if not self.printer.intervals.is_ready(Interval.PING):
            return

        self.printer.latency.ping_now()
        await self.send(PingMsg())

    async def clear_bed(self, success: bool = True, rating: Optional[int] = None):
        if self.printer.have_cleared_bed:
            return

        try:
            if self.simplyprint_api is None:
                raise RuntimeError("SimplyPrint API is not configured")
            await self.simplyprint_api.clear_bed(
                self.config.id, self.file_action_token, success, rating
            )
            self.printer.have_cleared_bed = True
        except Exception as e:
            self.logger.warning("Failed to clear bed: %s", e)

    async def start_next_print(self):
        try:
            if self.simplyprint_api is None:
                raise RuntimeError("SimplyPrint API is not configured")
            await self.simplyprint_api.start_next_print(
                self.config.id, self.file_action_token
            )
        except Exception as e:
            self.logger.warning("Failed to start next print: %s", e)

    async def push_notification(self, **kwargs: Unpack[NotificationEventKwargs]):
        """
        Push unmanaged notification, no response available, no event_id available.
        Alternatively use the notification state to manage persistent notifications.
        """
        if "event_id" in kwargs:
            raise TypeError(
                "push_notification() does not accept 'event_id'; "
                "use the notification state for persistent notifications"
            )
        await self.send(NotificationMsg(data={"events": [NotificationEvent(**kwargs)]}))

    def get_current_peripheral_definitions(
        self,
    ) -> Optional[PeripheralDefinitions]:
        return None

    # Default event handling.

    def _on_error(self, msg: ErrorMsg):
        self.logger.warning("Server reported an error: %s", msg.data)

    async def _on_new_token(self, msg: NewTokenMsg):
        self.config.token = msg.data.token
        self.config.short_id = msg.data.short_id
        self.config.in_setup = bool(msg.data.short_id)

        await self.event_bus.emit(ClientConfigChangedEvent)

    async def _on_connected_data(self, msg: ConnectedMsg):
        if msg.data is None:
            # A bare `connected` frame carries nothing to apply; the
            # established-state handler still runs at its own priority.
            return

        self.config.name = msg.data.name
        self.config.in_setup = msg.data.in_setup
        self.config.short_id = msg.data.short_id

        # TODO: Reconnect token.

        if msg.data.intervals is not None:
            self.printer.intervals.update(msg.data.intervals)

        await self.event_bus.emit(ClientConfigChangedEvent)

    async def _on_setup_complete(self, msg: CompleteSetupMsg):
        try:
            self.printer.mark_common_fields_as_changed()
            self.config.id = msg.data.printer_id
            self.config.in_setup = False
            await self.event_bus.emit(ClientConfigChangedEvent)
        except Exception as e:
            self.logger.exception("Failed to complete setup: %s", e)

    def _on_interval_change(self, msg: IntervalChangeMsg):
        self.printer.intervals.update(msg.data)

    def _on_pong(self, _msg: PongMsg):
        self.printer.latency.pong_now()

    def _on_printer_settings(self, msg: PrinterSettingsMsg):
        self.printer.settings = msg.data

    def _on_stream_received(self, _msg: StreamReceivedMsg) -> None: ...

    def _on_webcam_snapshot(self, data: WebcamSnapshotDemandData):
        if data.timer is not None:
            self.printer.intervals.webcam = data.timer

    def _on_file_demand(self, data: FileDemandData):
        """Store file action_token for later use."""
        self.printer.current_job_id = data.job_id
        self._job_id_admitted_at = self.msg_id  # see _hold
        self.printer.file_action_token = data.action_token
        self.printer.have_cleared_bed = False

    def _apply_material_data(self, data: SetMaterialDataDemandData) -> None:
        for material in data.materials:
            entry = self.printer.material(material.nozzle, material.ext)

            if entry is None:
                continue

            entry.model_update(material)
            entry.model_reset_changed()

    async def on_refresh_material_data(
        self, _data: Optional[RefreshMaterialDataDemandData] = None
    ) -> None:
        await self.send(
            MaterialDataMsg(data=dict(MaterialDataMsg.build_refresh(self.printer)))
        )

    async def on_refresh_peripherals(
        self, _data: Optional[RefreshPeripheralsDemandData] = None
    ) -> None:
        definitions = self.get_current_peripheral_definitions()

        if definitions is None:
            return

        await self.send(PeripheralDefinitionsMsg(data=definitions), skip_dispatch=True)

    async def _on_resolve_notification(self, data: ResolveNotificationDemandData):
        event = self.printer.notifications.notifications.get(data.event_id)

        if event is None:
            # Already removed client-side (or never known) - nothing to resolve.
            self.logger.debug("Ignoring resolve for unknown event %s", data.event_id)
            return

        # The default action is to resolve the event client side.
        if data.action is None:
            event.resolve()

        # Give the event the response data directly, otherwise users can handle this event manually.
        event.respond(data)
