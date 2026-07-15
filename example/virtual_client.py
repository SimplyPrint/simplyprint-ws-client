import asyncio
import math
import random
import time
from pathlib import Path
from typing import Optional

from yarl import URL

from simplyprint_ws_client import (
    ConnectedMsg,
    ClientContext,
    FileDemandData,
    FileProgressStateEnum,
    GcodeDemandData,
    MeshDataMsg,
    PrinterConfig,
    PrinterClient,
    PrinterStatus,
)
from simplyprint_ws_client.integration.camera.base import (
    BaseCameraProtocol,
    CameraProtocolPollingMode,
)


def expt_smooth(target, actual, alpha, dt) -> float:
    return actual + (target - actual) * (1.0 - math.exp(-alpha * dt))


class VirtualConfig(PrinterConfig):
    """Define extra fields that will be persisted in a config entry"""

    ...


def random_float(a, b):
    return a + random.random() * (b - a)


def _generate_fake_mesh_data():
    return {
        "mesh_matrix": [[random_float(0, 1) for _ in range(4)] for _ in range(4)],
        "mesh_max": [random_float(200, 220) for _ in range(3)],
        "mesh_min": [random_float(0, 20) for _ in range(3)],
    }


def _random_test_image():
    path = Path(__file__).parent / "images"
    images = list(path.glob("*.jpg"))
    random.shuffle(images)
    return images[0].read_bytes()


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
            yield _random_test_image()


class VirtualClient(PrinterClient[VirtualConfig]):
    job_progress_alpha: float = 2.0
    pending_job: Optional[FileDemandData] = None

    def __init__(
        self,
        config: VirtualConfig,
        *,
        context: ClientContext,
    ) -> None:
        super().__init__(config, context=context)

        self.printer.firmware.machine_name = "Creality K2"
        self.printer.firmware.name = "Creality K2"
        self.printer.firmware.version = "1.0.0"

        self.printer.set_info("Virtual Printer", "0.0.1")
        self.printer.info.api = "Bambu"
        self.printer.tool_count = 1

        self.camera.set_uri(URL("virtual://localhost"))

        self.printer.material0.color = "#BC0900"
        self.printer.material0.type = "PETG"

    async def on_connected(self, _msg: ConnectedMsg) -> None:
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

        # ~10s download: alpha=0.3, dt=0.1
        alpha = 0.3
        raw_percent = 0.0

        while self.printer.file_progress.percent < 100:
            raw_percent = expt_smooth(105.0, raw_percent, alpha, 0.1)
            self.printer.file_progress.percent = min(100, round(raw_percent))
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

    async def on_cancel(self, _):
        self.printer.status = PrinterStatus.CANCELLING
        self.printer.job_info.cancelled = True
        await asyncio.sleep(2)
        self.printer.status = PrinterStatus.OPERATIONAL

        self.printer.bed.temperature.target = 0.0
        self.printer.tool0.temperature.target = 0.0

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
