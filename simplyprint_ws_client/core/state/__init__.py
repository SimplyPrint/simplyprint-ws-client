__all__ = [
    "StateModel",
    "TemperatureState",
    "AmbientTemperatureState",
    "FileProgressState",
    "CpuInfoState",
    "PrinterInfoState",
    "PrinterFirmwareState",
    "PrinterFirmwareWarning",
    "PrinterFilamentSensorState",
    "PSUState",
    "JobInfoState",
    "PingPongState",
    "WebcamState",
    "WebcamSettings",
    "PrinterState",
    "PrinterCpuFlag",
    "PrinterStatus",
    "FileProgressStateEnum",
    "FilamentSensorEnum",
    "Intervals",
    "DisplaySettings",
    "PrinterSettings",
    "VIRTUAL_SPOOL_POSITION",
    "MaterialLayoutEntry",
    "MaterialEntry",
    "BedState",
    "ToolState",
    "MultiMaterialSolution",
    "NozzleType",
    "BedType",
    "VolumeType",
    "JobObjectEntry",
    "NotificationEventSeverity",
    "NotificationEventType",
    "NotificationEventEffect",
    "NotificationEventPayload",
    "NotificationEvent",
    "NotificationActionResponses",
]

import asyncio
import datetime
import time
import uuid
from typing import (
    Literal,
    no_type_check,
    List,
    Set,
    ClassVar,
    Dict,
    Any,
    Callable,
    Hashable,
    Optional,
    Union,
    TYPE_CHECKING,
)

from pydantic import Field, PrivateAttr, BaseModel

from .models import (
    PrinterCpuFlag,
    PrinterStatus,
    FileProgressStateEnum,
    FilamentSensorEnum,
    Intervals,
    DisplaySettings,
    PrinterSettings,
    VIRTUAL_SPOOL_POSITION,
    MultiMaterialSolution,
    BedType,
    NozzleType,
    VolumeType,
    NotificationEventType,
    NotificationEventSeverity,
    NotificationEventActions,
    NotificationEventEffect,
    NotificationActionResponses,
)
from .state_model import StateModel, Exclusive, Untracked
from .utils import _resize_state_inplace
from ..config import PrinterConfig
from ...const import VERSION
from ...shared.asyncio.event_loop_provider import EventLoopProvider
from ...shared.hardware.physical_machine import PhysicalMachine
from ...shared.sp.ambient_check import AmbientCheck

if TYPE_CHECKING:
    from ..ws_protocol.messages import ResolveNotificationDemandData

try:
    from typing import Unpack, Never
except ImportError:
    from typing_extensions import Unpack, Never


class TemperatureState(StateModel):
    actual: Optional[float] = None
    target: Optional[float] = None

    def as_rounded(self, k: Literal["actual", "target"]) -> Optional[int]:
        value: Optional[float] = getattr(self, k)

        if value is None:
            return None

        return round(value)

    def is_heating(self) -> bool:
        target = self.as_rounded("target")
        actual = self.as_rounded("actual")
        return target not in {None, 0} and target != actual

    def to_list(self):
        actual = self.as_rounded("actual")
        target = self.as_rounded("target")

        return [actual] + ([target] if target is not None else [])

    def __eq__(self, other):
        if not isinstance(other, TemperatureState):
            return False

        return self.as_rounded("target") == other.as_rounded(
            "target"
        ) and self.as_rounded("actual") == other.as_rounded("actual")


class AmbientTemperatureState(StateModel):
    ambient: int = 0

    _initial_sample: Optional[float] = None
    _update_interval: float = AmbientCheck.CHECK_INTERVAL
    _last_update: float = PrivateAttr(default_factory=lambda: time.time())

    def on_changed(self, new_ambient: float):
        self.ambient = round(new_ambient)

    def tick(self, state: "PrinterState"):
        """
        It is up to the implementation to decide when to invoke the check or respect the update_interval,
        the entire state is self-contained and requires the tool_temperatures to be passed in from the PrinterState,
        but it handles triggering the appropriate events.
        """
        now = time.time()

        if (
            self._last_update is not None
            and now - self._last_update < self._update_interval
        ):
            return

        self._last_update = now

        (self._initial_sample, self.ambient, self._update_interval) = (
            AmbientCheck.detect(
                self.on_changed,
                state.tools,
                self._initial_sample,
                self.ambient,
                state.status,
            )
        )


class FileProgressState(StateModel):
    state: Optional[FileProgressStateEnum] = None
    percent: float = 0.0
    message: Optional[str] = None

    @no_type_check
    def __setattr__(self, key, value):
        super().__setattr__(key, value)

        # Reset the progress when the state changes away from downloading.
        if key == "state" and value != FileProgressStateEnum.DOWNLOADING:
            self.percent = 0.0


class CpuInfoState(StateModel):
    usage: Optional[float] = None
    temp: Optional[float] = None
    memory: Optional[float] = None


class PrinterInfoState(StateModel):
    ui: Optional[str] = None
    ui_version: Optional[str] = None
    api: Optional[str] = None
    api_version: Optional[str] = None
    machine: Optional[str] = None
    os: Optional[str] = None
    sp_version: Optional[str] = VERSION
    python_version: Optional[str] = None
    is_ethernet: Optional[bool] = None
    ssid: Optional[str] = None
    local_ip: Optional[str] = None
    hostname: Optional[str] = None
    core_count: Optional[int] = None
    total_memory: Optional[int] = None
    mac: Optional[str] = None


class PrinterFirmwareState(StateModel):
    name: Optional[str] = None
    name_raw: Optional[str] = None
    machine: Optional[str] = None
    machine_name: Optional[str] = None
    version: Optional[str] = None
    date: Optional[str] = None
    link: Optional[str] = None


class PrinterFirmwareWarning(StateModel):
    check_name: Optional[str] = None
    warning_type: Optional[str] = None
    severity: Optional[str] = None
    url: Optional[str] = None


class PrinterFilamentSensorState(StateModel):
    state: Optional[FilamentSensorEnum] = None


class PSUState(StateModel):
    on: bool = False


class JobInfoState(StateModel, validate_assignment=True):
    progress: Optional[float] = None
    initial_estimate: Optional[float] = None
    layer: Optional[int] = None
    time: Optional[float] = None
    filament: Optional[float] = None
    filename: Exclusive[Optional[str]] = None
    delay: Optional[float] = None
    # Deprecated.
    # ai: List[int]

    # These needs to always trigger a reset.
    started: Exclusive[bool] = False
    finished: Exclusive[bool] = False
    cancelled: Exclusive[bool] = False
    failed: Exclusive[bool] = False

    # Mark a print job as a reprint of a previous (not-cleared) job from the client.
    reprint: Exclusive[Optional[int]] = None

    # Current object being printed, if known (not used currently).
    object: Exclusive[Optional[Union[int, str]]] = None
    # List of object ids that have been skipped.
    # Can be both delta or full list, when set it is sent.
    skipped_objects: Optional[List[Union[int, str]]] = None

    MUTUALLY_EXCLUSIVE_FIELDS: ClassVar[Set[str]] = {
        "started",
        "finished",
        "cancelled",
        "failed",
    }

    @no_type_check
    def __setattr__(self, key, value):
        """Only one of the 4 fields can be True at a time."""
        if key not in self.MUTUALLY_EXCLUSIVE_FIELDS:
            return super().__setattr__(key, value)

        # Set all other to false
        for field in self.MUTUALLY_EXCLUSIVE_FIELDS - {key}:
            super().__setattr__(field, False)

        return super().__setattr__(key, value)


class PingPongState(StateModel):
    ping: Optional[float] = None
    pong: Optional[float] = None

    def ping_now(self):
        self.ping = time.monotonic()

    def pong_now(self):
        self.pong = time.monotonic()

    def get_latency(self) -> Optional[float]:
        if self.ping is None or self.pong is None:
            return None

        return round((self.pong - self.ping) * 1000)


class WebcamState(StateModel):
    connected: bool = False


class WebcamSettings(StateModel):
    flipH: bool = False
    flipV: bool = False
    rotate90: bool = False


class MaterialLayoutEntry(StateModel):
    nozzle: int = 0
    mms: Optional[MultiMaterialSolution] = None
    size: Optional[int] = None
    chains: Optional[int] = None

    def get_computed_size(self) -> int:
        """Get total slot count (size * chains), or 0 for offset-based types."""
        if self.mms == MultiMaterialSolution.VIRTUAL:
            return 0  # VIRTUAL type doesn't consume sequential slots
        return self.get_size() * self.get_chains()

    def get_size(self) -> int:
        return self.size or (self.mms.default_size if self.mms else 1)

    def get_chains(self) -> int:
        if self.mms and self.mms.can_chain:
            return min(self.chains or 1, self.mms.max_chains)
        return 1

    @property
    def offset(self) -> int:
        """Starting extruder offset for this layout entry."""
        return self.mms.offset if self.mms else 0


class MaterialEntry(StateModel):
    nozzle: int
    ext: int
    type: Union[str, int, None] = None  # Material type name
    color: Optional[str] = None  # Material color name, e.g. "Red"
    hex: Optional[str] = None  # Material color hex code, e.g. "#FF0000"
    raw: Optional[dict] = None  # Vendor specific data

    @property
    def empty(self) -> bool:
        """Check if the material entry is empty."""
        return (
            self.type is None
            and self.color is None
            and self.hex is None
            and self.raw is None
        )

    def clear(self):
        self.type = None
        self.color = None
        self.hex = None
        self.raw = None


class BedState(StateModel):
    type: Optional[BedType] = None
    temperature: TemperatureState = Field(default_factory=TemperatureState)

    def is_heating(self) -> bool:
        """Returns True if the bed is currently heating."""
        return self.temperature.is_heating()


class ToolState(StateModel):
    nozzle: int
    type: Optional[NozzleType] = None
    volume_type: Optional[VolumeType] = None
    size: Optional[float] = None
    temperature: TemperatureState = Field(default_factory=TemperatureState)
    active_material: Optional[int] = None
    materials: List[MaterialEntry] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the tool state with a default material if none are provided."""
        if not self.materials:
            self.materials.append(MaterialEntry(nozzle=self.nozzle, ext=0))

    def is_heating(self):
        """Returns True if the tool is currently heating."""
        return self.temperature.is_heating()

    @property
    def material_count(self) -> int:
        """Returns the number of materials for this tool."""
        return len(self.materials)

    @material_count.setter
    def material_count(self, count: int) -> None:
        """Sets the number of materials for this tool, resizing the list if necessary."""
        if count < 1:
            raise ValueError("Material count must be at least 1")

        with self:
            _resize_state_inplace(
                self,
                self.materials,
                count,
                lambda i: MaterialEntry(nozzle=self.nozzle, ext=i),
            )


class JobObjectEntry(StateModel):
    """A skip-able object definition to share with SimplyPrint."""

    class PrintProgressPoint(BaseModel):
        layer: Optional[int] = None
        percentage: Optional[float] = None

    id: Optional[Union[int, str]] = None
    name: Optional[str] = None
    bbox: Optional[List[float]] = None
    outline: Optional[List[List[float]]] = None
    area: Optional[float] = None
    instance: Optional[int] = None
    center: Optional[List[float]] = None
    time: Optional[int] = None
    layer_height: Optional[float] = None
    filament_usage: Optional[List[float]] = None
    prints_from: Optional[PrintProgressPoint] = None
    prints_to: Optional[PrintProgressPoint] = None


class NotificationEventPayload(StateModel):
    title: Optional[str] = None
    message: Optional[str] = None
    url: Optional[str] = None
    image_url: Optional[str] = None
    effect: Optional[NotificationEventEffect] = None
    actions: Optional[Dict[str, NotificationEventActions]] = None  # action_id -> action
    data: Optional[dict] = None  # any data to attach to the event


class NotificationEvent(StateModel):
    event_id: Optional[uuid.UUID] = None
    type: NotificationEventType = NotificationEventType.GENERIC
    severity: NotificationEventSeverity = NotificationEventSeverity.INFO
    payload: NotificationEventPayload = Field(default_factory=NotificationEventPayload)
    issued_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    resolved_at: Optional[datetime.datetime] = None

    # Integrated response handler, as an alternative pattern to implementing
    # a large on_resolve_notification handler.
    _response_future: Optional[asyncio.Future] = PrivateAttr(None)

    def __await__(self):
        return self.wait_for_response().__await__()

    async def wait_for_response(
        self, timeout: Optional[float] = None
    ) -> Optional["ResolveNotificationDemandData"]:
        ctx = self.ctx()
        if not ctx or not isinstance(ctx, EventLoopProvider):
            raise RuntimeError("NotificationEvent is not bound to a Client context.")
        self._response_future = ctx.event_loop.create_future()
        try:
            return await asyncio.wait_for(self._response_future, timeout)
        except asyncio.TimeoutError:
            return None

    def resolve(self, when: Optional[datetime.datetime] = None):
        """Mark the event as resolved."""
        self.resolved_at = when or datetime.datetime.now(datetime.timezone.utc)

    def respond(self, data: Optional["ResolveNotificationDemandData"]):
        if not self._response_future or self._response_future.done():
            return
        ctx = self.ctx()
        if not ctx or not isinstance(ctx, EventLoopProvider):
            raise RuntimeError("NotificationEvent is not bound to a Client context.")
        ctx.event_loop.call_soon(self._response_future.set_result, data)


class NotificationsState(StateModel):
    """Notification and event center for printer, relayed to SimplyPrint"""

    notifications: Dict[uuid.UUID, NotificationEvent] = Field(default_factory=dict)

    # Map hashable objects to an event for persistent references.
    __idempotency_keys: Dict[Hashable, uuid.UUID] = PrivateAttr(default_factory=dict)

    def new(self, **kwargs: Unpack[NotificationEvent]) -> NotificationEvent:
        """Create a new unmanaged persistent (with event_id) notification event which can be updated and resolved async"""
        if "event_id" not in kwargs:
            kwargs["event_id"] = uuid.uuid4()
        event = NotificationEvent(**kwargs)
        event.provide_context(self)
        event.model_set_changed(
            "issued_at"
        )  # mark as changed when we create the event.
        self.notifications[event.event_id] = event
        return event

    def keyed(
        self,
        idempotency_key: Hashable,
        event_id: Never = ...,
        **kwargs: Unpack[NotificationEvent],
    ) -> NotificationEvent:
        """Create new managed persistent reference to event, which is tied to some external object state (args)"""
        with self:
            if (
                event_id := self.__idempotency_keys.get(idempotency_key)
            ) and event_id in self.notifications:
                event = self.notifications[event_id]
                if event.resolved_at is None:
                    return event

            event = self.new(**kwargs)
            self.__idempotency_keys[idempotency_key] = event.event_id
            return event

    def rekey(self, idempotency_key: Hashable, event_id: uuid.UUID):
        if event_id not in self.notifications:
            return
        with self:
            self.__idempotency_keys[idempotency_key] = event_id

    def keys(self) -> List[Hashable]:
        """Return a (copied) list of all keys currently in use."""
        with self:
            return list(self.__idempotency_keys.keys())

    def retain(self, *uuids: uuid.UUID, remove_immediately=False):
        """Clean up all events except the ones provided."""
        for key in list(self.notifications.keys()):
            if key in uuids:
                continue
            if remove_immediately:
                self.remove(key)
                continue
            if key in self.notifications:
                self.notifications[key].resolve()

    def retain_keys(self, *keys: Hashable, remove_immediately=False):
        """
        Retain only the notifications with the given keys, remove all others.
        Useful when downstream APIs mark resolved events as no longer appearing.
        """
        for key in self.keys():
            if key in keys:
                continue
            if remove_immediately:
                self.remove(key)
                continue
            with self:
                uuid_key = self.__idempotency_keys.get(key)
            if uuid_key and uuid_key in self.notifications:
                self.notifications[uuid_key].resolve()

    def filter_retain_keys(
        self,
        func: Callable[[Hashable], bool],
        *keys: Hashable,
        remove_immediately=False,
    ):
        """
        Retain all keys, except for the subset of keys where func(key) is True - your kept keys.
        Useful when having multiple basis types for keys and wanting to retain only a subset.
        """
        all_keys = self.keys()
        all_keys_without_subset = set(all_keys) - set(k for k in all_keys if func(k))
        all_keys_without_subset = all_keys_without_subset.union(
            set(keys)
        )  # Add the explicit keys to retain
        self.retain_keys(
            *all_keys_without_subset, remove_immediately=remove_immediately
        )

    def clear(self, remove_immediately=False):
        """Reset all notifications"""
        self.retain(remove_immediately=remove_immediately)

    def get(self, key: Union[Hashable, uuid.UUID]) -> Optional[NotificationEvent]:
        """Get the notification with the given key."""
        if not isinstance(key, uuid.UUID):
            with self:
                key = self.__idempotency_keys.get(key)

        return self.notifications.get(key)

    def remove(self, key: Union[Hashable, uuid.UUID]):
        """Remove the notification with the given key."""
        if not isinstance(key, uuid.UUID):
            with self:
                key = self.__idempotency_keys.pop(key, None)
        else:
            # If we are removing an uuid, also remove all keys pointing to it.
            with self:
                for k, v in list(self.__idempotency_keys.items()):
                    if v != key:
                        continue
                    self.__idempotency_keys.pop(k, None)

        self.notifications.pop(key, None)

    def __contains__(self, item) -> bool:
        if isinstance(item, NotificationEvent):
            return item in self.notifications.values()

        if isinstance(item, bytes):
            item = item.hex()

        return item in self.notifications.keys()


class PrinterState(StateModel):
    # Non-tracked configuration/state.
    config: Untracked[PrinterConfig]
    have_cleared_bed: Untracked[bool] = False
    current_job_id: Untracked[Optional[int]] = None
    file_action_token: Untracked[Optional[str]] = None

    # General state.
    status: Optional[PrinterStatus] = None
    bed: BedState = Field(default_factory=BedState)
    tools: List[ToolState] = Field(default_factory=lambda: [ToolState(nozzle=0)])
    mms_layout: List[MaterialLayoutEntry] = Field(default_factory=list)

    # Job state.
    job_info: JobInfoState = Field(default_factory=JobInfoState)
    file_progress: FileProgressState = Field(default_factory=FileProgressState)

    # Misc.
    intervals: Intervals = Field(default_factory=Intervals)
    settings: PrinterSettings = Field(default_factory=PrinterSettings)
    latency: PingPongState = Field(default_factory=PingPongState)
    notifications: NotificationsState = Field(default_factory=NotificationsState)

    # Static information
    info: PrinterInfoState = Field(default_factory=PrinterInfoState)
    firmware: PrinterFirmwareState = Field(default_factory=PrinterFirmwareState)
    firmware_warning: PrinterFirmwareWarning = Field(
        default_factory=PrinterFirmwareWarning
    )

    # Sensors
    cpu_info: CpuInfoState = Field(default_factory=CpuInfoState)
    psu_info: PSUState = Field(default_factory=PSUState)
    filament_sensor: PrinterFilamentSensorState = Field(
        default_factory=PrinterFilamentSensorState
    )
    ambient_temperature: AmbientTemperatureState = Field(
        default_factory=AmbientTemperatureState
    )

    # Webcam
    webcam_info: WebcamState = Field(default_factory=WebcamState)
    webcam_settings: WebcamSettings = Field(default_factory=WebcamSettings)

    def set_info(self, name, version="0.0.1"):
        """Set the same info for all fields, both for UI / API and the client."""
        self.set_api_info(name, version)
        self.set_ui_info(name, version)

    def set_api_info(self, api: str, api_version: str):
        self.info.api = api
        self.info.api_version = api_version

    def set_ui_info(self, ui: str, ui_version: str):
        self.info.ui = ui
        self.info.ui_version = ui_version

    @property
    def tool0(self) -> ToolState:
        """Convenience property to access the first tool."""
        return self.tools[0]

    @property
    def material0(self) -> MaterialEntry:
        tool0 = self.tool0
        return tool0.materials[0]

    @property
    def materials0(self) -> List[MaterialEntry]:
        """Convenience property to access the materials of the first tool."""
        return self.tool0.materials

    @property
    def tool_count(self) -> int:
        return len(self.tools)

    @tool_count.setter
    def tool_count(self, count: int) -> None:
        if count < 1:
            raise ValueError("Nozzle count must be at least 1")

        with self:
            _resize_state_inplace(
                self, self.tools, count, lambda i: ToolState(nozzle=i)
            )

    def tool(self, nozzle: int = 0) -> Optional[ToolState]:
        """Safe getter for the tool temperature at the given nozzle index."""
        if nozzle < 0:
            return None

        return self.tools[nozzle] if nozzle < len(self.tools) else None

    def material(self, nozzle: int = 0, ext: int = 0) -> Optional[MaterialEntry]:
        """Safe getter for the material at the given nozzle index and ext."""
        if tool := self.tool(nozzle):
            if ext < 0 or ext >= tool.material_count:
                return None

            return tool.materials[ext]

        return None

    def update_mms_layout(self, mms_layout: List[MaterialLayoutEntry]):
        """Helper function to set nozzles and materials based on a provided MMS layout.
        It does not change the tool count, this needs to be done separately.
        """

        with self:
            # compare the new layout with the current one
            if len(self.mms_layout) == len(mms_layout):
                for a, b in zip(self.mms_layout, mms_layout):
                    if a == b:
                        continue
                    break
                else:
                    # If all entries are the same, no need to update
                    return

            self.mms_layout = mms_layout

            layout_per_nozzle: Dict[int, List[MaterialLayoutEntry]] = {}

            for entry in mms_layout:
                if entry.nozzle not in layout_per_nozzle:
                    layout_per_nozzle[entry.nozzle] = []
                layout_per_nozzle[entry.nozzle].append(entry)

            material_count_per_nozzle: Dict[int, int] = {}

            for nozzle, entries in layout_per_nozzle.items():
                material_count_per_nozzle[nozzle] = 0

                for entry in entries:
                    material_count_per_nozzle[nozzle] += entry.get_computed_size()

            for i, tool in enumerate(self.tools):
                tool.material_count = max(material_count_per_nozzle.get(i, 1), 1)

    def is_printing(self, *status) -> bool:
        """If any of the statuses are printing, return True. Default behavior is to check own status."""
        if len(status) == 0:
            status = (self.status,)

        return PrinterStatus.is_printing(*status)

    def is_heating(self) -> bool:
        return any([h.is_heating() for h in (self.bed.temperature, *self.tools)])

    def populate_info_from_physical_machine(self, *skip: str):
        """Set information about the physical machine the client is running on."""
        for k, v in PhysicalMachine.get_info().items():
            if k in skip:
                continue

            setattr(self.info, k, v)

    def mark_common_fields_as_changed(self):
        # Mark non-default fields as changed so they will be sent to the client.
        # In theory, we could store this information, but this is easier.
        self.model_set_changed("status")
        self.info.model_set_changed("sp_version")
        self.firmware.model_set_changed("name")
