__all__ = [
    "Client",
    "ClientConfigChangedEvent",
    "ClientStateChangeEvent",
    "DefaultClient",
    "PhysicalClient",
    "ClientState",
    "configure",
]

import asyncio
import logging
import weakref
from abc import ABC
from datetime import timedelta, datetime
from enum import IntEnum
from typing import NamedTuple, Optional, Union, Generic, TypeVar, cast

try:
    from typing import Unpack, Never
except ImportError:
    from typing_extensions import Unpack, Never

from .autowire import (
    configure,
    autowire,
    AutowireClientMeta,
)
from .config import PrinterConfig
from .state import PrinterState, NotificationEvent
from .ws_protocol.connection import ConnectionMode
from .ws_protocol.events import (
    ConnectionOutgoingEvent,
    ConnectionEstablishedEvent,
    ConnectionLostEvent,
    ConnectionIncomingEvent,
)
from .ws_protocol.messages import (
    SetMaterialDataDemandData,
    MaterialDataMsg,
    MultiPrinterRemoveConnectionMsg,
    MultiPrinterRemovedMsg,
    MultiPrinterAddedMsg,
    PingMsg,
    PrinterSettingsMsg,
    IntervalChangeMsg,
    CompleteSetupMsg,
    NewTokenMsg,
    ErrorMsg,
    ConnectedMsg,
    FileDemandData,
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
from .ws_protocol.models import (
    ServerMsgType,
    ClientMsgType,
    DispatchMode,
    DemandMsgType,
)

from ..events import EventBus, Event
from ..events.event import sync_only
from ..shared.asyncio.event_loop_provider import EventLoopProvider
from ..shared.hardware.physical_machine import PhysicalMachine
from ..shared.logging import ClientName
from ..shared.sp.simplyprint_api import SimplyPrintApi
from ..shared.utils.backoff import Backoff, ExponentialBackoff


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


TConfig = TypeVar("TConfig", bound=PrinterConfig)

# Map message producers

_CLIENT_MSG_PRODUCERS = {
    MachineDataMsg: ["info"],
    WebcamStatusMsg: ["webcam_info.connected"],
    WebcamMsg: ["webcam_settings"],
    FirmwareMsg: ["firmware"],
    FirmwareWarningMsg: ["firmware_warning"],
    ToolMsg: ["tools.*.active_material"],
    TemperatureMsg: ["bed.temperature", "tools.*.temperature"],
    AmbientTemperatureMsg: ["ambient_temperature.ambient"],
    StateChangeMsg: ["status"],
    JobInfoMsg: ["job_info"],
    LatencyMsg: ["latency.pong"],
    FileProgressMsg: ["file_progress"],
    FilamentSensorMsg: ["filament_sensor"],
    PowerControllerMsg: ["psu_info"],
    CpuInfoMsg: ["cpu_info"],
    MaterialDataMsg: [
        "tools.*.materials",
        "tools.*.size",
        "tools.*.type",
        "tools.*.volume_type",
        "bed.type",
        "mms_layout",
    ],
    NotificationMsg: [
        "notifications.notifications",
    ],
}

_CLIENT_MSG_MAP = {k: v for v, keys in _CLIENT_MSG_PRODUCERS.items() for k in keys}


class Client(
    ABC,
    Generic[TConfig],
    EventLoopProvider[asyncio.AbstractEventLoop],
    metaclass=AutowireClientMeta,
):
    """A scheduling unit.

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

    v: int = -1
    msg_id: int = -1
    last_msg_id: int = -1
    printer: PrinterState
    event_bus: EventBus
    logger: logging.Logger

    _state: VersionedState
    _should_be_allocated: bool = True

    _pending_action_backoff: Backoff
    _pending_action_delay: timedelta = timedelta.min
    _pending_action_ts: datetime = datetime.min
    _pending_action_log_ts: datetime = datetime.min

    def __init__(
        self,
        config: TConfig,
        *,
        event_loop_provider: Optional[EventLoopProvider] = None,
        **kwargs,
    ):
        ABC.__init__(self)
        Generic.__init__(self)
        EventLoopProvider.__init__(self, provider=event_loop_provider)
        self._state = VersionedState(-1, ClientState.CONNECTING)
        self._pending_action_backoff = ExponentialBackoff(10, 600, 3600)
        self.event_bus = EventBus(event_loop_provider=self)
        self.printer = PrinterState(config=config)
        self.printer.provide_context(weakref.ref(self))
        self.logger = logging.getLogger(ClientName(self))
        autowire(self)

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
        return self.msg_id > self.last_msg_id

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

    def consume(self):
        """Consume the list of pending messages."""
        self.last_msg_id = self.msg_id

        changes = self.printer.model_recursive_changeset
        msg_kinds = {}

        # Build a unique map of message kinds together with their highest version.
        for k, v in changes.items():
            if k not in _CLIENT_MSG_MAP:
                continue

            msg_kind = _CLIENT_MSG_MAP.get(k)
            current = msg_kinds.get(msg_kind)

            if current is None:
                msg_kinds[msg_kind] = (v, v)
                continue

            lowest, highest = current
            msg_kinds[msg_kind] = (min(lowest, v), max(highest, v))

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
            if msg.dispatch_mode(self.printer) != DispatchMode.DISPATCH:
                continue

            msgs.append(msg)
            msg.reset_changes(self.printer, v=highest)

        return msgs, -1  # max(v for _, v in msg_kinds)

    # internal methods

    @configure(ConnectionIncomingEvent)
    async def _on_connection_incoming(self, msg: ServerMsgKind, v: int):
        if self.v > v:
            self.logger.warning("Dropped incoming message %s with v: %d.", msg, v)
            return

        if self.v != v:
            self.v = v
            self.logger.warning(
                "Upgraded client connection version from %d to %d due to new message.",
                self.v,
                v,
            )

        if msg.type == ServerMsgType.DEMAND:
            await self.event_bus.emit(msg.data.demand, msg.data)
        else:
            await self.event_bus.emit(msg.type, msg)

    @configure(ConnectionEstablishedEvent)
    def _on_connection_established(self, event: ConnectionEstablishedEvent):
        self.v = event.v

        if self.state == ClientState.CONNECTING:
            self.state = ClientState.NOT_CONNECTED

    @configure(ConnectionLostEvent)
    def _on_connection_lost(self, event: ConnectionLostEvent):
        if self.v > event.v:
            return

        # handle connection lost.
        self.v = event.v
        self._pending_action_backoff.reset()
        self.state = ClientState.CONNECTING
        self.signal()

    # important functional event handling

    @configure(ServerMsgType.ADD_CONNECTION, priority=1)
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

    @configure(ServerMsgType.REMOVE_CONNECTION, priority=1)
    async def _on_multi_printer_removed(self, msg: MultiPrinterRemovedMsg):
        self.logger.debug("Connection removed. %s", msg)
        self.state = ClientState.NOT_CONNECTED
        self.signal()

    @configure(ServerMsgType.CONNECTED, priority=2)
    async def _on_connected_state(self):
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

        await self.event_bus.emit(ConnectionOutgoingEvent, msg, self.v)

    # lifetime methods

    async def init(self):
        """Init lifecycle method. Called once per halt, and initially."""
        pass

    async def tick(self, delta: timedelta):
        """Tick lifecycle method"""
        pass

    async def halt(self):
        """Halt lifecycle method, temporarily not considered for scheduling."""
        pass

    async def teardown(self):
        """Teardown lifecycle method, final cleanup, will never be needed again."""
        pass


class DefaultClient(Client[TConfig], ABC):
    """
    Default prioritized message handling.

    Attributes:
        _have_cleared_bed: Keep track of whether we have called the API successfully to prevent many irrelevant calls.
        _file_action_token: File action token
    """

    @property
    def current_job_id(self):
        return self.printer.current_job_id

    @property
    def file_action_token(self):
        return self.printer.file_action_token

    async def send_ping(self) -> None:
        if not self.printer.intervals.is_ready("ping"):
            return

        self.printer.latency.ping_now()
        await self.send(PingMsg())

    async def clear_bed(self, success: bool = True, rating: Optional[int] = None):
        if self.printer.have_cleared_bed:
            return

        try:
            await SimplyPrintApi.clear_bed(
                self.config.id, self._file_action_token, success, rating
            )
            self.printer.have_cleared_bed = True
        except Exception as e:
            self.logger.warning("Failed to clear bed: %s", e)

    async def start_next_print(self):
        try:
            await SimplyPrintApi.start_next_print(
                self.config.id, self._file_action_token
            )
        except Exception as e:
            self.logger.warning("Failed to start next print: %s", e)

    async def push_notification(
        self, event_id: Never = ..., **kwargs: Unpack[NotificationEvent]
    ):
        """
        Push unmanaged notification, no response available, no event_id available.
        Alternatively use the notification state to manage persistent notifications.
        """
        await self.send(NotificationMsg(data={"events": [NotificationEvent(**kwargs)]}))

    # Default event handling.

    @configure(ServerMsgType.ERROR, priority=1)
    def _on_error(self, msg: ErrorMsg): ...

    @configure(ServerMsgType.NEW_TOKEN, priority=1)
    async def _on_new_token(self, msg: NewTokenMsg):
        self.config.token = msg.data.token
        self.config.short_id = msg.data.short_id
        self.config.in_setup = bool(msg.data.short_id)

        await self.event_bus.emit(ClientConfigChangedEvent)

    @configure(ServerMsgType.CONNECTED, priority=1)
    async def _on_connected_data(self, msg: ConnectedMsg):
        self.config.name = msg.data.name
        self.config.in_setup = msg.data.in_setup
        self.config.short_id = msg.data.short_id

        # TODO: Reconnect token.

        if msg.data.intervals is not None:
            self.printer.intervals.update(msg.data.intervals)

        await self.event_bus.emit(ClientConfigChangedEvent)

    @configure(ServerMsgType.COMPLETE_SETUP, priority=1)
    async def _on_setup_complete(self, msg: CompleteSetupMsg):
        try:
            self.printer.mark_common_fields_as_changed()
            self.config.id = msg.data.printer_id
            self.config.in_setup = False
            await self.event_bus.emit(ClientConfigChangedEvent)
        except Exception as e:
            self.logger.exception("Failed to complete setup: %s", e)

    @configure(ServerMsgType.INTERVAL_CHANGE, priority=1)
    def _on_interval_change(self, msg: IntervalChangeMsg):
        self.printer.intervals.update(msg.data)

    @configure(ServerMsgType.PONG, priority=1)
    def _on_pong(self):
        self.printer.latency.pong_now()

    @configure(ServerMsgType.PRINTER_SETTINGS, priority=1)
    def _on_printer_settings(self, msg: PrinterSettingsMsg):
        self.printer.settings = msg.data

    @configure(ServerMsgType.STREAM_RECEIVED, priority=1)
    def _on_stream_received(self): ...

    @configure(DemandMsgType.WEBCAM_SNAPSHOT, priority=1)
    def _on_webcam_snapshot(self, data: WebcamSnapshotDemandData):
        if data.timer is not None:
            self.printer.intervals.webcam = data.timer

    @configure(DemandMsgType.FILE, priority=1)
    def _on_file_demand(self, data: FileDemandData):
        """Store file action_token for later use."""
        self.printer.current_job_id = data.job_id
        self.printer.file_action_token = data.action_token
        self.printer.have_cleared_bed = False

    @configure(DemandMsgType.SET_MATERIAL_DATA, priority=1)
    def _on_set_material_data(self, data: SetMaterialDataDemandData):
        for material in data.materials:
            tool = self.printer.tool(material.nozzle)

            if tool is None:
                continue

            if material.ext not in tool.materials:
                continue

            tool.materials[material.ext].model_update(material)
            tool.materials[material.ext].model_reset_changed()

    @configure(DemandMsgType.REFRESH_MATERIAL_DATA, priority=1)
    async def _on_refresh_material_data(self):
        await self.send(
            MaterialDataMsg(
                data=dict(MaterialDataMsg.build(self.printer, is_refresh=True))
            )
        )

    @configure(DemandMsgType.RESOLVE_NOTIFICATION, priority=1)
    async def _on_resolve_notification(self, data: ResolveNotificationDemandData):
        event = self.printer.notifications.notifications.get(data.event_id)

        # The default action is to resolve the event client side.
        if data.action is None:
            event.resolve()

        # Give the event the response data directly, otherwise users can handle this event manually.
        event.respond(data)


class PhysicalClient(DefaultClient[TConfig], ABC):
    def __init__(self, config: PrinterConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.printer.populate_info_from_physical_machine()

    @configure(DemandMsgType.SYSTEM_RESTART, priority=1)
    def _on_system_restart(self):
        PhysicalMachine.restart()

    @configure(DemandMsgType.SYSTEM_SHUTDOWN, priority=1)
    def _on_system_shutdown(self):
        PhysicalMachine.shutdown()
