import asyncio
import base64
import math
import random
import time
from typing import Optional

from yarl import URL

from simplyprint_ws_client import (
    PrinterConfig,
    DefaultClient,
    GcodeDemandData,
    PrinterStatus,
    FileDemandData,
    FileProgressStateEnum,
    MaterialDataMsg,
    MeshDataMsg,
)
from simplyprint_ws_client.shared.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)
from simplyprint_ws_client.shared.camera.mixin import ClientCameraMixin


def expt_smooth(target, actual, alpha, dt) -> float:
    return actual + (target - actual) * (1.0 - math.exp(-alpha * dt))


class VirtualConfig(PrinterConfig):
    """Define extra fields that will be persisted in a config entry"""

    ...


_TEST_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAeAAAAFoCAIAAAAAVb93AAAFMklEQVR4nOzWQQkCYRhFUZFpYAVzWcAAGsAONnFvJ3FtAZc/fJfhnARvdXnb8/Q4wF68bp/pCbDMcXoAAP8JNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRG3f+3l6AyxzeV+nJ8AyHjRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNECUQANECTRAlEADRAk0QJRAA0QJNEDULwAA//+/JAlAlWQCTwAAAABJRU5ErkJggg=="


def random_float(a, b):
    return a + random.random() * (b - a)


def _generate_fake_mesh_data():
    return {
        "mesh_matrix": [[random_float(0, 1) for _ in range(4)] for _ in range(4)],
        "mesh_max": [random_float(200, 220) for _ in range(3)],
        "mesh_min": [random_float(0, 20) for _ in range(3)],
    }


class VirtualCamera(BaseCameraProtocol):
    polling_mode = CameraProtocolPollingMode.ON_DEMAND
    is_async = False

    @staticmethod
    def test(uri: URL) -> bool:
        if uri.scheme != "virtual":
            return False

        return True

    def read(self):
        while True:
            time.sleep(0.5)
            yield base64.b64decode(_TEST_IMAGE)


class VirtualClient(DefaultClient[VirtualConfig], ClientCameraMixin):
    job_progress_alpha: float = 0.1
    pending_job: Optional[FileDemandData] = None

    def __init__(self, *args, **kwargs):
        DefaultClient.__init__(self, *args, **kwargs)
        self.initialize_camera_mixin(**kwargs)

        self.printer.firmware.name = "Virtual Printer Firmware"
        self.printer.firmware.version = "1.0.0"

        self.printer.set_info("Virtual Printer", "0.0.1")
        self.printer.tool_count = 1
        self.printer.tool().material_count = 4

        self.camera_uri = URL("virtual://localhost")
        self.printer.webcam_info.connected = True

        for i, mat in enumerate(self.printer.materials0):
            mat.type = "PLA" if i in (0, 1) else "PETG"
            mat.color = "Black"
            mat.hex = "#000000"

    async def on_connected(self):
        _ = self
        self.logger.info("Yay i am connected :) :) :)")

    async def on_gcode(self, data: GcodeDemandData):
        self.logger.info("Gcode: %s", data.list)

        # event = self.printer.notifications.new(
        #    type=NotificationEventType.GENERIC,
        #    severity=NotificationEventSeverity.ERROR,
        #    payload=NotificationEventPayload(
        #        title="Hey! Are you sure about this?",
        #        message=f"Bout to execute very dangerous gcode commands {'; '.join(data.list)}",
        #        actions={
        #            "cancel": NotificationEventButtonAction(
        #                label="Cancel gcode command before it's too late!"
        #            )
        #        },
        #    ),
        # )
        #
        # response = await event.wait_for_response(timeout=5)
        # if response is not None and response.action == "cancel":
        #    self.logger.info("Gcode command cancelled by user.")
        #    return

        if any("BEDLEVELVISUALIZER" in gcode for gcode in data.list):
            await asyncio.sleep(2)
            self.logger.info("Generated fake mesh data for BEDLEVELVISUALIZER")
            await self.send(MeshDataMsg(data=_generate_fake_mesh_data()))
            return

        for gcode in data.list:
            if gcode[:4] == "M104":
                target = float(gcode[6:])

                self.logger.info(f"Setting tool temperature to {target}")

                if target > 0.0:
                    self.printer.tool0.temperature.target = target
                else:
                    self.printer.tool0.temperature.target = 0.0

            if gcode[:4] == "M140":
                target = float(gcode[6:])

                self.logger.info(f"Setting bed temperature to {target}")

                if target > 0.0:
                    self.printer.bed.temperature.target = target
                else:
                    self.printer.bed.temperature.target = 0.0

    async def on_file(self, data: FileDemandData):
        self.printer.status = PrinterStatus.DOWNLOADING

        # fake self.printer.file_progress.percent using event.file_size
        self.printer.file_progress.state = FileProgressStateEnum.DOWNLOADING
        self.printer.file_progress.percent = 0.0

        alpha = random.uniform(0.1, 0.5)

        while self.printer.file_progress.percent < 100.0:
            self.printer.file_progress.percent = max(
                100.0,
                expt_smooth(
                    105.0,
                    self.printer.file_progress.percent,
                    alpha,
                    0.1,
                ),
            )
            await asyncio.sleep(0.1)

        self.pending_job = data
        self.printer.file_progress.state = FileProgressStateEnum.READY

        if data.auto_start:
            await self.on_start_print(data)
        else:
            self.printer.status = PrinterStatus.OPERATIONAL

    async def on_start_print(self, _):
        if not self.pending_job:
            return

        self.pending_job = None

        # self.job_progress_alpha = random.uniform(0.05, 0.1)

        self.printer.status = PrinterStatus.PRINTING
        self.printer.job_info.started = True
        self.printer.job_info.progress = 0.0
        # Calculate the time to finish the print using the progress rate
        self.printer.job_info.time = round(100.0 / self.job_progress_alpha)

        self.printer.bed.temperature.target = 60.0
        self.printer.tool0.temperature.target = 225.0

    async def on_cancel(self, _):
        self.printer.status = PrinterStatus.CANCELLING
        self.printer.job_info.cancelled = True
        await asyncio.sleep(2)
        self.printer.status = PrinterStatus.OPERATIONAL

        self.printer.bed.temperature.target = 0.0
        self.printer.tool0.temperature.target = 0.0

    async def on_stream_off(self):
        self.printer.material0.raw = {
            "tray_uuid": "C5D095A34DF246E8A9B99C1D6AD667BE",
            "tag_uid": "2496010000000100",
            "tray_color": "5898DDFF",
            "tray_type": "TPU-AMS",
            "tray_id_name": "U02-B0",
            "tray_info_idx": "GFU02",
            "tray_sub_brands": "TPU for AMS",
            "tray_weight": "1000",
            "tray_diameter": "1.75",
            "cols": ["5898DDFF"],
        }

        await self.send(
            MaterialDataMsg(
                data=dict(MaterialDataMsg.build(self.printer, is_refresh=True))
            )
        )

    async def init(self):
        self.printer.bed.temperature.actual = 20.0
        self.printer.bed.temperature.target = 0.0
        self.printer.tool0.temperature.actual = 20.0
        self.printer.tool0.temperature.target = 0.0
        self.printer.status = PrinterStatus.OPERATIONAL

    async def tick(self, _):
        await self.send_ping()

        # Update temperatures, printer status and so on with smoothing function
        if self.printer.bed.temperature.target:
            target = self.printer.bed.temperature.target

            self.printer.bed.temperature.actual = expt_smooth(
                target,
                self.printer.bed.temperature.actual,
                1,
                0.1,
            )

        else:
            self.printer.bed.temperature.actual = 20.0

        if self.printer.tool0.temperature.target:
            target = self.printer.tool0.temperature.target

            self.printer.tool0.temperature.actual = expt_smooth(
                target,
                self.printer.tool0.temperature.actual,
                1,
                0.1,
            )

        else:
            self.printer.tool0.temperature.actual = 20.0

        self.printer.ambient_temperature.tick(self.printer)

        if (
            self.printer.status == PrinterStatus.PRINTING
            and not self.printer.is_heating()
        ):
            self.printer.job_info.progress = expt_smooth(
                100.0,
                self.printer.job_info.progress,
                self.job_progress_alpha,
                0.1,
            )

            self.printer.job_info.time = round(100.0 / self.job_progress_alpha)

            if round(self.printer.job_info.progress) >= 100.0:
                self.printer.job_info.finished = True
                self.printer.job_info.progress = 100
                self.printer.status = PrinterStatus.OPERATIONAL

                self.printer.bed.temperature.target = 0.0
                self.printer.tool0.temperature.target = 0.0

    async def halt(self):
        pass

    async def teardown(self):
        pass
