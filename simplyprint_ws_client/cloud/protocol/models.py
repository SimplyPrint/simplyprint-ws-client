__all__ = [
    "ServerMsgType",
    "DemandMsgType",
    "ClientMsgType",
    "DispatchMode",
]

from enum import IntEnum, StrEnum


class ServerMsgType(StrEnum):
    DEMAND = "demand"
    ERROR = "error"
    NEW_TOKEN = "new_token"
    CONNECTED = "connected"
    COMPLETE_SETUP = "complete_setup"
    INTERVAL_CHANGE = "interval_change"
    PONG = "pong"
    STREAM_RECEIVED = "stream_received"
    PRINTER_SETTINGS = "printer_settings"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"


class DemandMsgType(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    TERMINAL = "terminal"
    GCODE = "gcode"
    TEST_WEBCAM = "test_webcam"
    WEBCAM_SNAPSHOT = "webcam_snapshot"
    FILE = "file"
    START_PRINT = "start_print"
    CONNECT_PRINTER = "connect_printer"
    DISCONNECT_PRINTER = "disconnect_printer"
    SYSTEM_RESTART = "system_restart"
    SYSTEM_SHUTDOWN = "system_shutdown"
    API_RESTART = "api_restart"
    API_SHUTDOWN = "api_shutdown"
    UPDATE = "update"
    PLUGIN_INSTALL = "plugin_install"
    PLUGIN_UNINSTALL = "plugin_uninstall"
    WEBCAM_SETTINGS_UPDATED = "webcam_settings_updated"
    STREAM_ON = "stream_on"
    STREAM_OFF = "stream_off"
    SET_PRINTER_PROFILE = "set_printer_profile"
    SET_MATERIAL_DATA = "set_material_data"
    REFRESH_MATERIAL_DATA = "refresh_material_data"
    SKIP_OBJECTS = "skip_objects"
    GET_GCODE_SCRIPT_BACKUPS = "get_gcode_script_backups"
    HAS_GCODE_CHANGES = "has_gcode_changes"
    PSU_KEEPALIVE = "psu_keepalive"
    PSU_ON = "psu_on"
    PSU_OFF = "psu_off"
    DISABLE_WEBSOCKETS = "disable_websockets"
    GOTO_WS_PROD = "goto_ws_prod"
    GOTO_WS_TEST = "goto_ws_test"
    SEND_LOGS = "send_logs"
    RESOLVE_NOTIFICATION = "resolve_notification"


class ClientMsgType(StrEnum):
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    PING = "ping"
    KEEPALIVE = "keepalive"
    LATENCY = "latency"
    TOOL = "tool"
    STATUS = "state_change"
    AMBIENT = "ambient"
    TEMPERATURES = "temps"
    SHUTDOWN = "shutdown"
    CONNECTION = "connection"
    CAMERA_SETTINGS = "camera_settings"
    JOB_INFO = "job_info"
    FILE_PROGRESS = "file_progress"
    PSU_CHANGE = "psu_change"
    CPU_INFO = "cpu_info"
    PSU = "psu"
    STREAM = "stream"
    PRINTER_ERROR = "printer_error"
    MESH_DATA = "mesh_data"
    INFO = "machine_data"
    INPUT_REQUIRED = "input_required"
    UPDATE_STARTED = "update_started"
    FIRMWARE = "firmware"
    WEBCAM = "webcam"
    WEBCAM_STATUS = "webcam_status"
    UNSAFE_FIRMWARE = "unsafe_firmware"
    FILAMENT_ANALYSIS = "filament_analysis"
    OCTOPRINT_PLUGINS = "octoprint_plugins"
    GCODE_SCRIPTS = "gcode_scripts"
    INSTALLED_PLUGINS = "installed_plugins"
    SOFTWARE_UPDATES = "software_updates"
    FIRMWARE_WARNING = "firmware_warning"
    AI_RESP = "ai_resp"
    LOGS_SENT = "logs_sent"
    FILAMENT_SENSOR = "filament_sensor"
    MATERIAL_DATA = "material_data"
    NOTIFICATION = "notification"
    OBJECTS = "objects"

    def when_pending(self) -> bool:
        # Allowed messages for a pending printer connection
        return self in {
            ClientMsgType.PING,
            ClientMsgType.KEEPALIVE,
            ClientMsgType.CONNECTION,
            ClientMsgType.STATUS,
            ClientMsgType.SHUTDOWN,
            ClientMsgType.INFO,
            ClientMsgType.FIRMWARE,
            ClientMsgType.FIRMWARE_WARNING,
            ClientMsgType.INSTALLED_PLUGINS,
        }


class DispatchMode(IntEnum):
    """Either do nothing, or provide a reason
    for not sending the message.
    """

    DISPATCH = 0
    RATELIMIT = 1
    CANCEL = 2
