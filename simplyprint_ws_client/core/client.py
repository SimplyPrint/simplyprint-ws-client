__all__ = [
    "Client",
    "PendingMessage",
    "ClientConfigChangedEvent",
    "ClientStateChangeEvent",
    "ClientState",
    "PeripheralDefinitionEntry",
    "PeripheralDefinitions",
]

import asyncio
import logging
import weakref
from datetime import timedelta, datetime
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
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
from simplyprint_ws_client.core.job import JobTimeline
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


class PendingMessage(NamedTuple):
    """One immutable state projection and the newest version it contains."""

    message: ClientMsg
    version: int
    interval: Optional[Interval]
    order: int
    owner: "MessageOwner"


class MessageOwner(IntEnum):
    STATE = 0
    JOB_TIMELINE = 1


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
) -> int | None:
    """Latest change for one explicit state branch."""
    versions = []
    if field in model.model_self_changed_fields:
        versions.append(model.model_self_changed_fields[field])
    for child in nested:
        if changes := child.model_recursive_changeset:
            versions.append(max(changes.values()))
    return max(versions) if versions else None


def _message_changes(
    state: PrinterState,
) -> Iterable[tuple[type[ClientMsg], Iterable[int | None]]]:
    """Messages affected by the concrete branches of this state tree."""
    yield MachineDataMsg, (_field_version(state, "info", (state.info,)),)
    yield WebcamStatusMsg, (_field_version(state.webcam_info, "connected"),)
    yield (
        WebcamMsg,
        (_field_version(state, "webcam_settings", (state.webcam_settings,)),),
    )
    yield FirmwareMsg, (_field_version(state, "firmware", (state.firmware,)),)
    yield (
        FirmwareWarningMsg,
        (_field_version(state, "firmware_warning", (state.firmware_warning,)),),
    )
    yield ToolMsg, (_field_version(tool, "active_material") for tool in state.tools)
    yield (
        TemperatureMsg,
        (
            _field_version(state.bed, "temperature", (state.bed.temperature,)),
            _field_version(state.chamber, "temperature", (state.chamber.temperature,)),
            *(
                _field_version(tool, "temperature", (tool.temperature,))
                for tool in state.tools
            ),
        ),
    )
    yield AmbientTemperatureMsg, (_field_version(state.ambient_temperature, "ambient"),)
    yield StateChangeMsg, (_field_version(state, "status"),)
    yield JobInfoMsg, (_field_version(state, "job_info", (state.job_info,)),)
    yield LatencyMsg, (_field_version(state.latency, "pong"),)
    yield (
        FileProgressMsg,
        (_field_version(state, "file_progress", (state.file_progress,)),),
    )
    yield (
        FilamentSensorMsg,
        (_field_version(state, "filament_sensor", (state.filament_sensor,)),),
    )
    yield PowerControllerMsg, (_field_version(state, "psu_info", (state.psu_info,)),)
    yield CpuInfoMsg, (_field_version(state, "cpu_info", (state.cpu_info,)),)
    yield (
        MaterialDataMsg,
        (
            *(
                _field_version(tool, "materials", tool.materials)
                for tool in state.tools
            ),
            *(_field_version(tool, "size") for tool in state.tools),
            *(_field_version(tool, "type") for tool in state.tools),
            *(_field_version(tool, "volume_type") for tool in state.tools),
            _field_version(state.bed, "type"),
            _field_version(state, "mms_layout", state.mms_layout),
        ),
    )
    yield (
        NotificationMsg,
        (
            _field_version(
                state.notifications,
                "notifications",
                state.notifications.notifications.values(),
            ),
        ),
    )
    yield (
        PeripheralMsg,
        (
            _field_version(
                state.peripherals,
                "entries",
                state.peripherals.entries.values(),
            ),
        ),
    )


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
        self.job_timeline = JobTimeline()
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
        return self.printer.model_has_changed or self.job_timeline.has_pending

    def next_msg_id(self):
        self.msg_id += 1
        return self.msg_id

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

        if self.state == ClientState.CONNECTED and self._can_do_pending():
            self.state = ClientState.PENDING_REMOVED
            await self.send(MultiPrinterRemoveConnectionMsg(self.config))
            self._do_pending()

        return self.state <= ClientState.NOT_CONNECTED

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
        self.event_bus.emit_sync(ClientStateChangeEvent)

    def pending_messages(self) -> list[PendingMessage]:
        """Project pending state without acknowledging or mutating it."""
        msg_kinds = {}

        # Build a unique map of message kinds together with their highest version.
        for msg_kind, candidates in _message_changes(self.printer):
            if versions := tuple(v for v in candidates if v is not None):
                msg_kinds[msg_kind] = (min(versions), max(versions))

        is_pending = self.printer.config.is_pending()

        pending = []

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
            if msg.dispatch_mode(self.printer) != DispatchMode.DISPATCH:
                continue

            if msg_kind is JobInfoMsg and self.job_timeline.has_pending:
                continue

            pending.append(
                PendingMessage(
                    msg,
                    highest,
                    msg.dispatch_interval(self.printer),
                    lowest,
                    MessageOwner.STATE,
                )
            )

        if not is_pending or JobInfoMsg.msg_type().when_pending():
            pending.extend(
                PendingMessage(
                    item.message,
                    item.version,
                    None,
                    item.version,
                    MessageOwner.JOB_TIMELINE,
                )
                for item in self.job_timeline.pending_messages()
            )

        return sorted(pending, key=lambda item: item.order)

    def commit_message(self, pending: PendingMessage) -> None:
        """Acknowledge one projection after its socket write completed."""
        pending.message.reset_changes(self.printer, v=pending.version)
        if pending.owner == MessageOwner.JOB_TIMELINE:
            self.job_timeline.commit(pending.version)
        if pending.interval is not None:
            self.printer.intervals.use(pending.interval)

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

    def _on_connection_lost(self, event: SimplyPrintConnectionLostEvent):
        if self.v > event.v:
            return

        # handle connection lost.
        self.v = event.v
        self._pending_action_backoff.reset()
        self.state = ClientState.CONNECTING
        self.signal()

    # important functional event handling

    async def _on_multi_printer_added(self, msg: MultiPrinterAddedMsg):
        if not msg.data.status:
            self.logger.debug("Failed to add connection. %s", msg)
            self.state = ClientState.NOT_CONNECTED
            self.signal()
            return

        # A successful addition does not require a backoff.
        self._pending_action_backoff.reset()
        self.config.id = msg.data.pid
        self.state = ClientState.CONNECTED
        self.signal()

    async def _on_multi_printer_removed(self, msg: MultiPrinterRemovedMsg):
        self.logger.debug("Connection removed. %s", msg)
        self.state = ClientState.NOT_CONNECTED
        self.signal()

    async def _on_connected_state(self, _msg: ConnectedMsg):
        self.printer.mark_common_fields_as_changed()
        self.state = ClientState.CONNECTED
        self.signal()

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
        if not skip_dispatch:
            msg.mark_dispatched(self.printer)

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
