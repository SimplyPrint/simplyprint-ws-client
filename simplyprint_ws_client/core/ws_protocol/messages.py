__all__ = [
    "Msg",
    "ErrorMsgData",
    "ErrorMsg",
    "NewTokenMsgData",
    "NewTokenMsg",
    "ConnectedMsgData",
    "ConnectedMsg",
    "CompleteSetupMsgData",
    "CompleteSetupMsg",
    "IntervalChangeMsg",
    "PongMsg",
    "StreamReceivedMsg",
    "PrinterSettingsMsg",
    "MultiPrinterAddedMsgData",
    "MultiPrinterAddedMsg",
    "MultiPrinterRemovedMsgData",
    "MultiPrinterRemovedMsg",
    "PauseDemandData",
    "ResumeDemandData",
    "CancelDemandData",
    "TerminalDemandData",
    "GcodeDemandData",
    "WebcamTestDemandData",
    "WebcamSnapshotDemandData",
    "FileDemandData",
    "StartPrintDemandData",
    "ConnectPrinterDemandData",
    "DisconnectPrinterDemandData",
    "SystemRestartDemandData",
    "SystemShutdownDemandData",
    "ApiRestartDemandData",
    "ApiShutdownDemandData",
    "UpdateDemandData",
    "PluginInstallDemandData",
    "PluginUninstallDemandData",
    "WebcamSettingsUpdatedDemandData",
    "StreamOnDemandData",
    "StreamOffDemandData",
    "SetPrinterProfileDemandData",
    "SetMaterialDataDemandData",
    "SkipObjectsDemandData",
    "ResolveNotificationDemandData",
    "GetGcodeScriptBackupsDemandData",
    "HasGcodeChangesDemandData",
    "PsuKeepaliveDemandData",
    "PsuOnDemandData",
    "PsuOffDemandData",
    "DisableWebsocketsDemandData",
    "GotoWsProdDemandData",
    "GotoWsTestDemandData",
    "SendLogsDemandData",
    "DemandMsgKind",
    "DemandMsg",
    "ServerMsgKind",
    "ServerMsg",
    "TClientMsgDataGenerator",
    "ClientMsg",
    "MultiPrinterAddConnectionMsg",
    "MultiPrinterRemoveConnectionMsg",
    "GcodeScriptsMsg",
    "MachineDataMsg",
    "WebcamStatusMsg",
    "WebcamMsg",
    "InstalledPluginsMsg",
    "SoftwareUpdatesMsg",
    "FirmwareMsg",
    "FirmwareWarningMsg",
    "ToolMsg",
    "TemperatureMsg",
    "AmbientTemperatureMsg",
    "ConnectionMsg",
    "StateChangeMsg",
    "JobInfoMsg",
    "AiRespMsg",
    "PrinterErrorMsg",
    "ShutdownMsg",
    "StreamMsg",
    "PingMsg",
    "LatencyMsg",
    "FileProgressMsg",
    "FilamentSensorMsg",
    "PowerControllerMsg",
    "CpuInfoMsg",
    "MeshDataMsg",
    "LogsSentMsg",
    "MaterialDataMsg",
    "NotificationMsg",
    "ObjectsMsg",
]

import uuid
from itertools import chain
from typing import (
    Generic,
    TypeVar,
    Union,
    Literal,
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Generator,
    Annotated,
    get_args,
    ClassVar,
    Set,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    RootModel,
    model_serializer,
    model_validator,
)

from .models import (
    ServerMsgType,
    ClientMsgType,
    DemandMsgType,
    DispatchMode,
)
from ..config import PrinterConfig
from ..state import (
    Intervals,
    PrinterSettings,
    FileProgressStateEnum,
    PrinterState,
    JobInfoState,
    MaterialEntry,
    JobObjectEntry,
    PrinterStatus,
    NotificationActionResponses,
)

try:
    from types import NoneType
except ImportError:
    NoneType = type(None)

TMsgType = TypeVar("TMsgType", bound=str)
TMsgData = TypeVar("TMsgData")


class Msg(BaseModel, Generic[TMsgType, TMsgData], populate_by_name=True):
    type: TMsgType
    data: TMsgData
    for_client: Union[int, str, None] = Field(None, alias="for")

    @model_serializer(mode="plain", when_used="json")
    def model_serializer(self):
        """Waiting for https://github.com/pydantic/pydantic-core/pull/1535"""
        serialized = {"type": self.type}

        if self.data is not None and (
            not isinstance(self.data, (dict, list)) or len(self.data) > 0
        ):
            serialized["data"] = self.data

        if self.for_client is not None:
            serialized["for"] = self.for_client

        return serialized


class ErrorMsgData(BaseModel):
    msg: str


class ErrorMsg(Msg[Literal[ServerMsgType.ERROR], ErrorMsgData]): ...


class NewTokenMsgData(BaseModel):
    short_id: str
    token: str
    no_exist: Optional[bool] = None


class NewTokenMsg(Msg[Literal[ServerMsgType.NEW_TOKEN], NewTokenMsgData]): ...


class ConnectedMsgData(BaseModel):
    in_setup: bool = False
    intervals: Optional[Intervals] = None
    printer_settings: Optional[PrinterSettings] = None
    short_id: Optional[str] = None
    reconnect_token: Optional[str] = None
    name: Optional[str] = None
    region: str
    is_locked: Optional[bool] = None


class ConnectedMsg(Msg[Literal[ServerMsgType.CONNECTED], Optional[ConnectedMsgData]]):
    data: Optional[ConnectedMsgData] = None


class CompleteSetupMsgData(BaseModel):
    printer_id: int


class CompleteSetupMsg(
    Msg[Literal[ServerMsgType.COMPLETE_SETUP], CompleteSetupMsgData]
): ...


class IntervalChangeMsg(Msg[Literal[ServerMsgType.INTERVAL_CHANGE], Intervals]): ...


class PongMsg(Msg[Literal[ServerMsgType.PONG], NoneType]):
    data: NoneType = None


class StreamReceivedMsg(Msg[Literal[ServerMsgType.STREAM_RECEIVED], NoneType]):
    data: NoneType = None


class PrinterSettingsMsg(
    Msg[Literal[ServerMsgType.PRINTER_SETTINGS], PrinterSettings]
): ...


class MultiPrinterAddedMsgData(BaseModel):
    pid: Optional[int] = None
    unique_id: Optional[str] = None
    status: bool
    reason: Optional[str] = None


class MultiPrinterAddedMsg(
    Msg[Literal[ServerMsgType.ADD_CONNECTION], MultiPrinterAddedMsgData]
): ...


class MultiPrinterRemovedMsgData(BaseModel):
    pid: Optional[int] = None
    unique_id: Optional[str] = None
    deleted: bool
    # Emulated ws status codes
    code: Optional[int] = None
    reason: Optional[str] = None


class MultiPrinterRemovedMsg(
    Msg[Literal[ServerMsgType.REMOVE_CONNECTION], MultiPrinterRemovedMsgData]
): ...


class PauseDemandData(BaseModel):
    demand: Literal[DemandMsgType.PAUSE] = DemandMsgType.PAUSE


class ResumeDemandData(BaseModel):
    demand: Literal[DemandMsgType.RESUME] = DemandMsgType.RESUME


class CancelDemandData(BaseModel):
    demand: Literal[DemandMsgType.CANCEL] = DemandMsgType.CANCEL


class TerminalDemandData(BaseModel):
    demand: Literal[DemandMsgType.TERMINAL] = DemandMsgType.TERMINAL
    enabled: bool = False


class GcodeDemandData(BaseModel):
    demand: Literal[DemandMsgType.GCODE] = DemandMsgType.GCODE
    list: List[str] = Field(default_factory=list)


class WebcamTestDemandData(BaseModel):
    demand: Literal[DemandMsgType.TEST_WEBCAM] = DemandMsgType.TEST_WEBCAM


class WebcamSnapshotDemandData(BaseModel):
    demand: Literal[DemandMsgType.WEBCAM_SNAPSHOT] = DemandMsgType.WEBCAM_SNAPSHOT
    id: Optional[str] = None
    timer: Optional[int] = None
    endpoint: Optional[str] = None


class MMSMapEntry(BaseModel):
    nozzle: int
    ext: Optional[int]

    @model_validator(mode="before")
    @classmethod
    def _validate_mms_map_entry(
        cls, v: Union[int, dict, None]
    ) -> Dict[str, Union[int, None]]:
        """Coerce the MMS map entry to a dict with 'nozzle' and 'ext' keys."""
        if isinstance(v, dict):
            return v
        elif isinstance(v, int):
            return {"nozzle": 0, "ext": v}
        else:
            return {"nozzle": 0, "ext": None}


class FileDemandData(BaseModel):
    demand: Literal[DemandMsgType.FILE] = DemandMsgType.FILE
    job_id: Optional[int] = None
    url: Optional[str] = None
    cdn_url: Optional[str] = None
    auto_start: bool = False
    file_name: Optional[str] = None
    file_id: Optional[str] = None
    file_size: Optional[int] = None
    start_options: Dict[str, Union[bool, str]] = Field(default_factory=dict)
    mms_map: Optional[List[MMSMapEntry]] = None
    zip_printable: Optional[str] = None
    skip_objects: Optional[List[Union[str, int]]] = None
    action_token: Optional[str] = None


class StartPrintDemandData(BaseModel):
    demand: Literal[DemandMsgType.START_PRINT] = DemandMsgType.START_PRINT


class ConnectPrinterDemandData(BaseModel):
    demand: Literal[DemandMsgType.CONNECT_PRINTER] = DemandMsgType.CONNECT_PRINTER


class DisconnectPrinterDemandData(BaseModel):
    demand: Literal[DemandMsgType.DISCONNECT_PRINTER] = DemandMsgType.DISCONNECT_PRINTER


class SystemRestartDemandData(BaseModel):
    demand: Literal[DemandMsgType.SYSTEM_RESTART] = DemandMsgType.SYSTEM_RESTART


class SystemShutdownDemandData(BaseModel):
    demand: Literal[DemandMsgType.SYSTEM_SHUTDOWN] = DemandMsgType.SYSTEM_SHUTDOWN


class ApiRestartDemandData(BaseModel):
    demand: Literal[DemandMsgType.API_RESTART] = DemandMsgType.API_RESTART


class ApiShutdownDemandData(BaseModel):
    demand: Literal[DemandMsgType.API_SHUTDOWN] = DemandMsgType.API_SHUTDOWN


class UpdateDemandData(BaseModel):
    demand: Literal[DemandMsgType.UPDATE] = DemandMsgType.UPDATE


class PluginInstallDemandData(BaseModel):
    demand: Literal[DemandMsgType.PLUGIN_INSTALL] = DemandMsgType.PLUGIN_INSTALL
    plugins: Optional[List[Dict[str, Any]]] = None


class PluginUninstallDemandData(BaseModel):
    demand: Literal[DemandMsgType.PLUGIN_UNINSTALL] = DemandMsgType.PLUGIN_UNINSTALL
    plugins: Optional[List[Any]] = None


class WebcamSettingsUpdatedDemandData(BaseModel):
    demand: Literal[DemandMsgType.WEBCAM_SETTINGS_UPDATED] = (
        DemandMsgType.WEBCAM_SETTINGS_UPDATED
    )
    settings: Optional[Dict[str, Any]] = None


class StreamOnDemandData(BaseModel):
    demand: Literal[DemandMsgType.STREAM_ON] = DemandMsgType.STREAM_ON
    interval: Optional[float] = None

    # Default to 300 and always divide by 1000
    @field_validator("interval", mode="before")
    @classmethod
    def convert_interval(cls, v):
        return v / 1000 if v is not None else 300 / 1000


class StreamOffDemandData(BaseModel):
    demand: Literal[DemandMsgType.STREAM_OFF] = DemandMsgType.STREAM_OFF


class SetPrinterProfileDemandData(BaseModel):
    demand: Literal[DemandMsgType.SET_PRINTER_PROFILE] = (
        DemandMsgType.SET_PRINTER_PROFILE
    )
    printer_profile: Optional[Any] = None


class SetMaterialDataDemandData(BaseModel):
    demand: Literal[DemandMsgType.SET_MATERIAL_DATA] = DemandMsgType.SET_MATERIAL_DATA
    materials: List[MaterialEntry] = Field(default_factory=list)


class RefreshMaterialDataDemandData(BaseModel):
    demand: Literal[DemandMsgType.REFRESH_MATERIAL_DATA] = (
        DemandMsgType.REFRESH_MATERIAL_DATA
    )


class SkipObjectsDemandData(BaseModel):
    demand: Literal[DemandMsgType.SKIP_OBJECTS] = DemandMsgType.SKIP_OBJECTS
    objects: List[Union[str, int]] = Field(default_factory=list)


class ResolveNotificationDemandData(BaseModel):
    demand: Literal[DemandMsgType.RESOLVE_NOTIFICATION] = (
        DemandMsgType.RESOLVE_NOTIFICATION
    )
    event_id: uuid.UUID
    action: Optional[str]  # dynamic custom name of action or predefined resolve.
    payload: Optional[NotificationActionResponses] = None


class GetGcodeScriptBackupsDemandData(BaseModel):
    demand: Literal[DemandMsgType.GET_GCODE_SCRIPT_BACKUPS] = (
        DemandMsgType.GET_GCODE_SCRIPT_BACKUPS
    )
    force: bool = False


class HasGcodeChangesDemandData(BaseModel):
    demand: Literal[DemandMsgType.HAS_GCODE_CHANGES] = DemandMsgType.HAS_GCODE_CHANGES
    scripts: Optional[Any] = None


class PsuKeepaliveDemandData(BaseModel):
    demand: Literal[DemandMsgType.PSU_KEEPALIVE] = DemandMsgType.PSU_KEEPALIVE


class PsuOnDemandData(BaseModel):
    demand: Literal[DemandMsgType.PSU_ON] = DemandMsgType.PSU_ON


class PsuOffDemandData(BaseModel):
    demand: Literal[DemandMsgType.PSU_OFF] = DemandMsgType.PSU_OFF


class DisableWebsocketsDemandData(BaseModel):
    demand: Literal[DemandMsgType.DISABLE_WEBSOCKETS] = DemandMsgType.DISABLE_WEBSOCKETS
    websocket_ready: bool = False


class GotoWsProdDemandData(BaseModel):
    demand: Literal[DemandMsgType.GOTO_WS_PROD] = DemandMsgType.GOTO_WS_PROD


class GotoWsTestDemandData(BaseModel):
    demand: Literal[DemandMsgType.GOTO_WS_TEST] = DemandMsgType.GOTO_WS_TEST


class SendLogsDemandData(BaseModel):
    demand: Literal[DemandMsgType.SEND_LOGS] = DemandMsgType.SEND_LOGS
    token: str
    logs: List[str]
    max_body: int = 100000000

    @property
    def send_main(self):
        return "main" in self.logs

    @property
    def send_plugin(self):
        return "plugin" in self.logs

    @property
    def send_serial(self):
        return "serial" in self.logs


DemandMsgKind = Union[
    PauseDemandData,
    ResumeDemandData,
    CancelDemandData,
    TerminalDemandData,
    GcodeDemandData,
    WebcamTestDemandData,
    WebcamSnapshotDemandData,
    FileDemandData,
    StartPrintDemandData,
    ConnectPrinterDemandData,
    DisconnectPrinterDemandData,
    SystemRestartDemandData,
    SystemShutdownDemandData,
    ApiRestartDemandData,
    ApiShutdownDemandData,
    UpdateDemandData,
    PluginInstallDemandData,
    PluginUninstallDemandData,
    WebcamSettingsUpdatedDemandData,
    StreamOnDemandData,
    StreamOffDemandData,
    SetPrinterProfileDemandData,
    SetMaterialDataDemandData,
    RefreshMaterialDataDemandData,
    SkipObjectsDemandData,
    ResolveNotificationDemandData,
    GetGcodeScriptBackupsDemandData,
    HasGcodeChangesDemandData,
    PsuKeepaliveDemandData,
    PsuOnDemandData,
    PsuOffDemandData,
    DisableWebsocketsDemandData,
    GotoWsProdDemandData,
    GotoWsTestDemandData,
    SendLogsDemandData,
]


class DemandMsg(
    Msg[
        Literal[ServerMsgType.DEMAND],
        Annotated[DemandMsgKind, Field(discriminator="demand")],
    ]
): ...


ServerMsgKind = Union[
    ErrorMsg,
    NewTokenMsg,
    ConnectedMsg,
    CompleteSetupMsg,
    IntervalChangeMsg,
    PongMsg,
    StreamReceivedMsg,
    PrinterSettingsMsg,
    MultiPrinterAddedMsg,
    MultiPrinterRemovedMsg,
    DemandMsg,
]


# Incoming Msgs from the server
class ServerMsg(RootModel):
    root: Annotated[ServerMsgKind, Field(discriminator="type")]


TClientMsgDataGenerator = Generator[Tuple[str, Any], None, None]


# Outgoing Msgs from the client
class ClientMsg(Msg[TMsgType, Optional[dict]]):
    def __init__(self, data: Optional[dict] = None, **kwargs):
        super().__init__(type=self.msg_type(), data=data, **kwargs)

    @classmethod
    def msg_type(cls) -> TMsgType:
        """Return a literal type of the message"""
        return get_args(cls.model_fields["type"].annotation)[0]

    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        """Construct a dict with data based on the current state"""
        raise NotImplementedError()

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        """
        Custom reset logic for the message, typically used to
        fully reset some state, but optionally comparing and only
        resetting state when it has not been changed further.
        """
        pass

    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        """
        Based on the Msg and client state, determine if
        this message needs to be rate-limited, always sent or treated
        as normal. This takes intervals into consideration.
        """
        _ = self
        return DispatchMode.DISPATCH


class MultiPrinterAddConnectionMsg(ClientMsg[Literal[ClientMsgType.ADD_CONNECTION]]):
    def __init__(self, config: PrinterConfig, allow_setup: bool = False):
        super().__init__(
            data={
                "pid": int(config.id if not config.in_setup else 0),
                "token": config.token,
                "unique_id": config.unique_id,
                "allow_setup": bool(allow_setup or False),
                "client_ip": config.public_ip,
            }
        )


class MultiPrinterRemoveConnectionMsg(
    ClientMsg[Literal[ClientMsgType.REMOVE_CONNECTION]]
):
    def __init__(self, config: PrinterConfig):
        super().__init__(
            data={
                "pid": int(config.id if not config.in_setup else 0),
                "unique_id": config.unique_id,
            }
        )


class GcodeScriptsMsg(ClientMsg[Literal[ClientMsgType.GCODE_SCRIPTS]]): ...


class MachineDataMsg(ClientMsg[Literal[ClientMsgType.INFO]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        return state.info.model_dump(mode="json", exclude_none=True)

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.info.model_reset_changed()


class WebcamStatusMsg(ClientMsg[Literal[ClientMsgType.WEBCAM_STATUS]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        yield "connected", state.webcam_info.connected

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.webcam_info.model_reset_changed(v=v)


class WebcamMsg(ClientMsg[Literal[ClientMsgType.WEBCAM]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        for key in state.webcam_settings.model_changed_fields:
            yield key, getattr(state.webcam_settings, key)

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.webcam_settings.model_reset_changed()


class InstalledPluginsMsg(ClientMsg[Literal[ClientMsgType.INSTALLED_PLUGINS]]): ...


class SoftwareUpdatesMsg(ClientMsg[Literal[ClientMsgType.SOFTWARE_UPDATES]]): ...


class FirmwareMsg(ClientMsg[Literal[ClientMsgType.FIRMWARE]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        """
        Construct a dict "fw" with all fields prefixed with firmware,
        except for name which is just supposed to be firmware.
        """
        yield (
            "fw",
            (
                {
                    (f"firmware_{key}" if key != "name" else "firmware"): value
                    for key in state.firmware.__class__.model_fields
                    if (value := getattr(state.firmware, key)) is not None
                }
            ),
        )

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.firmware.model_reset_changed()


class FirmwareWarningMsg(ClientMsg[Literal[ClientMsgType.FIRMWARE_WARNING]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        for key in state.firmware_warning.model_changed_fields:
            yield key, getattr(state.firmware_warning, key)

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.firmware_warning.model_reset_changed()


class ToolMsg(ClientMsg[Literal[ClientMsgType.TOOL]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        # Derive active_tools from each tool's active_material
        yield (
            "active_tools",
            {tool.nozzle: tool.active_material for tool in state.tools},
        )

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        for tool in state.tools:
            tool.model_reset_changed("active_material", v=v)


class TemperatureMsg(ClientMsg[Literal[ClientMsgType.TEMPERATURES]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        if state.bed.temperature.model_has_changed:
            yield "bed", state.bed.temperature.to_list()

        for i, tool in enumerate(state.tools):
            if not tool.temperature.model_has_changed:
                continue

            yield f"tool{i}", tool.temperature.to_list()

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.bed.model_reset_changed("temperature")
        state.bed.temperature.model_reset_changed()

        for tool in state.tools:
            tool.model_reset_changed("temperature")
            tool.temperature.model_reset_changed()

    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        # If any target has been set, we need to send the message.
        if any(
            "target" in x.model_self_changed_fields
            for x in chain(
                (state.bed.temperature,), map(lambda t: t.temperature, state.tools)
            )
        ):
            return DispatchMode.DISPATCH

        # For normal temperature changes, report more often if we are heating up or down.
        # TODO: StrEnum
        interval_t: Literal["temps_target", "temps"] = (
            "temps_target" if state.is_heating() else "temps"
        )
        return state.intervals.dispatch_mode(interval_t)


class AmbientTemperatureMsg(ClientMsg[Literal[ClientMsgType.AMBIENT]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        yield "new", state.ambient_temperature.ambient

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.ambient_temperature.model_reset_changed()


class ConnectionMsg(ClientMsg[Literal[ClientMsgType.CONNECTION]]): ...


class StateChangeMsg(ClientMsg[Literal[ClientMsgType.STATUS]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        if state.status is None:
            return

        yield "new", state.status

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        # When the printer goes operational, we can clear the current job id.
        if state.status == PrinterStatus.OPERATIONAL:
            state.current_job_id = None

        state.model_reset_changed("status")


class JobInfoMsg(ClientMsg[Literal[ClientMsgType.JOB_INFO]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        if state.current_job_id is not None:
            yield "job_id", state.current_job_id

        for key in state.job_info.model_changed_fields:
            value = getattr(state.job_info, key)

            # Progress is a float, but for simplicity we'll round it here.
            # TODO: For diff checking it might be smart to keep it as an int always.
            if key == "progress" and value is not None:
                value = round(value)

            # XXX: job_info does not really support "null" values.
            if value is None:
                continue

            # These values are not allowed to be sent as anything but true.
            if key in JobInfoState.MUTUALLY_EXCLUSIVE_FIELDS and not value:
                continue

            yield key, value

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        # When an ended field has been changed to true, we can clear the current job id.
        ended_fields = JobInfoState.MUTUALLY_EXCLUSIVE_FIELDS - {"started"}
        if state.job_info.model_has_changes(*ended_fields) and any(
            getattr(state.job_info, f) for f in ended_fields
        ):
            state.current_job_id = None

        state.job_info.model_reset_changed()

    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        # Check if there is an intersection between the changed fields and the boolean fields
        if JobInfoState.MUTUALLY_EXCLUSIVE_FIELDS.intersection(
            state.job_info.model_changed_fields
        ):
            return DispatchMode.DISPATCH

        return state.intervals.dispatch_mode("job")


# Deprecated.
class AiRespMsg(ClientMsg[Literal[ClientMsgType.AI_RESP]]): ...


class PrinterErrorMsg(ClientMsg[Literal[ClientMsgType.PRINTER_ERROR]]): ...


class ShutdownMsg(ClientMsg[Literal[ClientMsgType.SHUTDOWN]]): ...


class StreamMsg(ClientMsg[Literal[ClientMsgType.STREAM]]):
    def __init__(self, base64jpg: str):
        super().__init__(data={"base": base64jpg})

    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        return state.intervals.dispatch_mode("webcam")


class PingMsg(ClientMsg[Literal[ClientMsgType.PING]]):
    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        return state.intervals.dispatch_mode("ping")


class LatencyMsg(ClientMsg[Literal[ClientMsgType.LATENCY]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        yield "ms", state.latency.get_latency()

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.latency.model_reset_changed()


class FileProgressMsg(ClientMsg[Literal[ClientMsgType.FILE_PROGRESS]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        if state.file_progress.state is None:
            return

        yield "state", state.file_progress.state

        if state.current_job_id is not None:
            yield "job_id", state.current_job_id

        if state.file_progress.state == FileProgressStateEnum.ERROR:
            yield "message", state.file_progress.message or "Unknown error"
            return

        # Always send progress when we are downloading.
        if state.file_progress.state in (
            FileProgressStateEnum.DOWNLOADING,
            FileProgressStateEnum.STARTED,
        ):
            yield "percent", state.file_progress.percent

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.file_progress.model_reset_changed(v=v)


class FilamentSensorMsg(ClientMsg[Literal[ClientMsgType.FILAMENT_SENSOR]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        yield "state", state.filament_sensor.state

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.filament_sensor.model_reset_changed()


class PowerControllerMsg(ClientMsg[Literal[ClientMsgType.PSU]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        yield "on", state.psu_info

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.psu_info.model_reset_changed()


class CpuInfoMsg(ClientMsg[Literal[ClientMsgType.CPU_INFO]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        for key in state.cpu_info.model_changed_fields:
            yield key, getattr(state.cpu_info, key)

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        state.cpu_info.model_reset_changed()

    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        return state.intervals.dispatch_mode("cpu")


class MeshDataMsg(ClientMsg[Literal[ClientMsgType.MESH_DATA]]): ...


class LogsSentMsg(ClientMsg[Literal[ClientMsgType.LOGS_SENT]]): ...


class MaterialDataMsg(ClientMsg[Literal[ClientMsgType.MATERIAL_DATA]]):
    _BED_FIELDS: ClassVar[Set[str]] = {"type"}
    _TOOL_FIELDS: ClassVar[Set[str]] = {"nozzle", "type", "volume_type", "size"}

    @classmethod
    def build(cls, state: PrinterState, is_refresh=False) -> TClientMsgDataGenerator:
        if is_refresh:
            yield "refresh", True

        if is_refresh or state.bed.model_has_changes(*cls._BED_FIELDS):
            yield (
                "bed",
                state.bed.model_dump(
                    exclude_none=True, include=cls._BED_FIELDS, mode="json"
                ),
            )

        if is_refresh or any(
            m.model_has_changes(*cls._TOOL_FIELDS) for m in state.tools
        ):
            yield (
                "nozzles",
                [
                    m.model_dump(
                        exclude_none=True, include=cls._TOOL_FIELDS, mode="json"
                    )
                    for m in state.tools
                    if m.model_has_changes(*cls._TOOL_FIELDS) or is_refresh
                ],
            )

        if (
            is_refresh
            or "mms_layout" in state.model_self_changed_fields
            or any(m.model_has_changed for m in state.mms_layout)
        ):
            yield (
                "layout",
                [
                    m.model_dump(exclude_none=True, mode="json")
                    for m in state.mms_layout
                ],
            )

        materials = list(chain.from_iterable(map(lambda t: t.materials, state.tools)))

        if is_refresh or any(m.model_has_changed for m in materials):
            yield (
                "materials",
                [
                    m.model_dump(exclude_none=True, mode="json")
                    for m in materials
                    if m.model_has_changed or is_refresh
                ],
            )

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        # Mirror of producer side.
        state.model_reset_changed("mms_layout", v=v)
        state.bed.model_reset_changed(*self._BED_FIELDS, v=v)
        for t in state.tools:
            for m in t.materials:
                m.model_reset_changed(v=v)
            t.model_reset_changed("materials", *self._TOOL_FIELDS, v=v)


class NotificationMsg(ClientMsg[Literal[ClientMsgType.NOTIFICATION]]):
    @classmethod
    def build(cls, state: PrinterState) -> TClientMsgDataGenerator:
        events = []

        # Find changed notification events, and either:
        # - Send the entire event the first time around
        # - Send only resolved_at when it has been resolved
        # - Send only changed fields otherwise (updates)
        for event_id, event in list(state.notifications.notifications.items()):
            if not event.model_has_changed:
                continue

            # Only send delta.
            if event.model_has_changes("resolved_at") and event.resolved_at is not None:
                data = event.model_dump(
                    include={"event_id", "resolved_at", *event.model_changed_fields}
                    - {"issued_at"},
                    mode="json",
                )
            elif event.model_has_changes("issued_at"):
                data = event.model_dump(exclude_none=True, mode="json")
            else:
                data = event.model_dump(
                    include={"event_id", *event.model_changed_fields}
                    - {"issued_at", "resolved_at"},
                    mode="json",
                )

            if not data:
                continue

            events.append(data)

        if not events:
            return

        yield "events", events

    def dispatch_mode(self, state: PrinterState) -> DispatchMode:
        return state.intervals.dispatch_mode("notification")

    def reset_changes(self, state: PrinterState, v: Optional[int] = None) -> None:
        for event_id, notification in list(state.notifications.notifications.items()):
            if notification.resolved_at is not None:
                state.notifications.remove(event_id)
                continue

            notification.model_reset_changed(v=v)


class ObjectsData(BaseModel):
    objects: List[JobObjectEntry]


class ObjectsMsg(ClientMsg[Literal[ClientMsgType.OBJECTS]]):
    """Raw objects data support"""

    def __init__(self, data: Union[dict, ObjectsData]):
        super().__init__(data=data)
