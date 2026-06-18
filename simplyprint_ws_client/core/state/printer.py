"""The reactive printer state model tree (:class:`PrinterState` and its parts)."""

import time
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Union,
    no_type_check,
)

from pydantic import BaseModel, Field, PrivateAttr

from simplyprint_ws_client.core.state.models import (
    BedType,
    FilamentSensorEnum,
    FileProgressStateEnum,
    Intervals,
    MultiMaterialSolution,
    NozzleType,
    PrinterSettings,
    PrinterStatus,
    VolumeType,
)
from simplyprint_ws_client.core.state.notifications import NotificationsState
from simplyprint_ws_client.common.model.reactive import ReactiveModel
from simplyprint_ws_client.common.model.annotations import Exclusive, Untracked
from simplyprint_ws_client.core.state.utils import _resize_state_inplace
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.const import VERSION
from simplyprint_ws_client.common.hardware.physical_machine import PhysicalMachine
from simplyprint_ws_client.core.api.ambient_check import AmbientCheck


class TemperatureState(ReactiveModel):
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


class AmbientTemperatureState(ReactiveModel):
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


class FileProgressState(ReactiveModel):
    state: Optional[FileProgressStateEnum] = None
    percent: float = 0.0
    message: Optional[str] = None

    @no_type_check
    def __setattr__(self, key, value):
        super().__setattr__(key, value)

        # Reset the progress when the state changes away from downloading.
        if key == "state" and value != FileProgressStateEnum.DOWNLOADING:
            self.percent = 0.0


class CpuInfoState(ReactiveModel):
    usage: Optional[float] = None
    temp: Optional[float] = None
    memory: Optional[float] = None


class PrinterInfoState(ReactiveModel):
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


class PrinterFirmwareState(ReactiveModel):
    name: Optional[str] = None
    name_raw: Optional[str] = None
    machine: Optional[str] = None
    machine_name: Optional[str] = None
    version: Optional[str] = None
    date: Optional[str] = None
    link: Optional[str] = None


class PrinterFirmwareWarning(ReactiveModel):
    check_name: Optional[str] = None
    warning_type: Optional[str] = None
    severity: Optional[str] = None
    url: Optional[str] = None


class PrinterFilamentSensorState(ReactiveModel):
    state: Optional[FilamentSensorEnum] = None


class PSUState(ReactiveModel):
    on: bool = False


_PERIPHERAL_UNSET = object()


class PeripheralHandle:
    def __init__(self, peripherals: "PeripheralsState", peripheral_id: str):
        self.peripherals = peripherals
        self.peripheral_id = peripheral_id

    @property
    def state(self) -> Optional["PeripheralState"]:
        return self.peripherals.get(self.peripheral_id)

    @property
    def value(self) -> Optional[Any]:
        return self.state.value if self.state else None

    @property
    def available(self) -> Optional[bool]:
        return self.state.available if self.state else None

    @property
    def updated(self) -> Optional[int]:
        return self.state.updated if self.state else None

    def set(
        self,
        value: Any,
        *,
        available: Optional[bool] = _PERIPHERAL_UNSET,
        updated: Optional[int] = _PERIPHERAL_UNSET,
    ) -> "PeripheralState":
        return self.peripherals.update(
            self.peripheral_id,
            value=value,
            available=available,
            updated=updated,
        )

    def update(
        self,
        *,
        value: Any = _PERIPHERAL_UNSET,
        available: Optional[bool] = _PERIPHERAL_UNSET,
        updated: Optional[int] = _PERIPHERAL_UNSET,
    ) -> "PeripheralState":
        return self.peripherals.update(
            self.peripheral_id,
            value=value,
            available=available,
            updated=updated,
        )


class PeripheralState(ReactiveModel):
    id: str
    value: Optional[Any] = Field(None, alias="v")
    available: Optional[bool] = Field(None, alias="a")
    updated: Optional[int] = Field(None, alias="u")


class PeripheralsState(ReactiveModel):
    entries: Dict[str, PeripheralState] = Field(default_factory=dict)

    def get(self, peripheral_id: str) -> Optional[PeripheralState]:
        return self.entries.get(peripheral_id)

    def peripheral(self, peripheral_id: str) -> PeripheralHandle:
        return PeripheralHandle(self, peripheral_id)

    def light(self, name: Union[str, int]) -> PeripheralHandle:
        return self.peripheral(f"light:{name}")

    def fan(self, name: Union[str, int]) -> PeripheralHandle:
        return self.peripheral(f"fan:{name}")

    def door(self, name: Union[str, int]) -> PeripheralHandle:
        return self.peripheral(f"door:{name}")

    def power(self, name: Union[str, int] = "psu") -> PeripheralHandle:
        return self.peripheral(f"power:{name}")

    def update(
        self,
        peripheral_id: str,
        *,
        value: Any = _PERIPHERAL_UNSET,
        available: Optional[bool] = _PERIPHERAL_UNSET,
        updated: Optional[int] = _PERIPHERAL_UNSET,
    ) -> PeripheralState:
        entry = self.entries.get(peripheral_id)

        if entry is None:
            entry = PeripheralState(id=peripheral_id)
            entry.provide_context(self)
            self.entries[peripheral_id] = entry
            self.model_set_changed("entries")
            entry.model_set_changed("id")

        if value is not _PERIPHERAL_UNSET:
            entry.value = value

        if available is not _PERIPHERAL_UNSET:
            entry.available = available

        if updated is not _PERIPHERAL_UNSET:
            entry.updated = updated

        return entry


class JobInfoState(ReactiveModel, validate_assignment=True):
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


class PingPongState(ReactiveModel):
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


class WebcamState(ReactiveModel):
    connected: bool = False


class WebcamSettings(ReactiveModel):
    flipH: bool = False
    flipV: bool = False
    rotate90: bool = False


class MaterialLayoutEntry(ReactiveModel):
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


class MaterialEntry(ReactiveModel):
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


class BedState(ReactiveModel):
    type: Optional[BedType] = None
    temperature: TemperatureState = Field(default_factory=TemperatureState)

    def is_heating(self) -> bool:
        """Returns True if the bed is currently heating."""
        return self.temperature.is_heating()


class ChamberState(ReactiveModel):
    temperature: TemperatureState = Field(default_factory=TemperatureState)

    def is_heating(self) -> bool:
        return self.temperature.is_heating()


class ToolState(ReactiveModel):
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


class JobObjectEntry(ReactiveModel):
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


class PrinterState(ReactiveModel):
    # Non-tracked configuration/state.
    config: Untracked[PrinterConfig]
    have_cleared_bed: Untracked[bool] = False
    current_job_id: Untracked[Optional[int]] = None
    file_action_token: Untracked[Optional[str]] = None

    # General state.
    status: Optional[PrinterStatus] = None
    bed: BedState = Field(default_factory=BedState)
    chamber: ChamberState = Field(default_factory=ChamberState)
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
    peripherals: PeripheralsState = Field(default_factory=PeripheralsState)
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

    def peripheral(self, peripheral_id: str) -> PeripheralHandle:
        return self.peripherals.peripheral(peripheral_id)

    def light(self, name: Union[str, int]) -> PeripheralHandle:
        return self.peripherals.light(name)

    def fan(self, name: Union[str, int]) -> PeripheralHandle:
        return self.peripherals.fan(name)

    def door(self, name: Union[str, int]) -> PeripheralHandle:
        return self.peripherals.door(name)

    def power(self, name: Union[str, int] = "psu") -> PeripheralHandle:
        return self.peripherals.power(name)

    def is_printing(self, *status) -> bool:
        """If any of the statuses are printing, return True. Default behavior is to check own status."""
        if len(status) == 0:
            status = (self.status,)

        return PrinterStatus.is_printing(*status)

    def is_heating(self) -> bool:
        return any([h.is_heating() for h in (self.bed, self.chamber, *self.tools)])

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
