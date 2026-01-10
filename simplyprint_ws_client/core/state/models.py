__all__ = [
    "PrinterCpuFlag",
    "PrinterStatus",
    "FileProgressStateEnum",
    "FilamentSensorEnum",
    "Intervals",
    "DisplaySettings",
    "PrinterSettings",
    "VIRTUAL_SPOOL_POSITION",
    "MultiMaterialSolution",
    "BedType",
    "NozzleType",
    "VolumeType",
    "NotificationEventSeverity",
    "NotificationEventType",
    "NotificationEventActionType",
    "NotificationEventActions",
    "NotificationEventEffect",
    "NotificationActionResponses",
    "NotificationEventButtonAction",
]

import asyncio
import time
from enum import IntEnum, StrEnum, Enum
from typing import Optional, Dict, Any, Literal, Generic, TypeVar, Union, Annotated

from pydantic import BaseModel, PrivateAttr, Field

from ..ws_protocol.models import DispatchMode

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class PrinterCpuFlag(IntEnum):
    NONE = 0
    THROTTLED = 1


class PrinterStatus(StrEnum):
    OPERATIONAL = "operational"
    PRINTING = "printing"
    OFFLINE = "offline"
    PAUSED = "paused"
    PAUSING = "pausing"
    CANCELLING = "cancelling"
    RESUMING = "resuming"
    DOWNLOADING = "downloading"
    ERROR = "error"
    NOT_READY = "not_ready"

    @staticmethod
    def is_printing(*status: Optional["PrinterStatus"]) -> bool:
        return (
            len(
                set(status).intersection(
                    {
                        PrinterStatus.PRINTING,
                        PrinterStatus.PAUSED,
                        PrinterStatus.PAUSING,
                        PrinterStatus.RESUMING,
                        PrinterStatus.CANCELLING,
                    }
                )
            )
            > 0
        )


class FileProgressStateEnum(StrEnum):
    DOWNLOADING = "downloading"
    ERROR = "error"
    PENDING = "pending"
    STARTED = "started"
    READY = "ready"


class FilamentSensorEnum(StrEnum):
    LOADED = "loaded"
    RUNOUT = "runout"


IntervalT = Literal[
    "ai",
    "job",
    "temps",
    "temps_target",
    "cpu",
    "reconnect",
    "ready_message",
    "ping",
    "webcam",
    "notification",
]


class Intervals(BaseModel):
    ai: int = 30000
    job: int = 5000
    temps: int = 5000
    temps_target: int = 2500
    cpu: int = 30000
    reconnect: int = 1000
    ready_message: int = 60000
    ping: int = 20000
    webcam: int = 1000
    notification: int = 1000

    _usages: Dict[str, Optional[int]] = PrivateAttr(...)

    def model_post_init(self, __context: Any) -> None:
        self._usages = {key: None for key in self.__class__.model_fields.keys()}

    @staticmethod
    def now() -> int:
        return int(time.monotonic_ns() / 1_000_000)

    def time_until_ready(self, t: IntervalT) -> int:
        if self._usages[t] is None:
            return 0

        return getattr(self, t) - (self.now() - self._usages[t])

    def is_ready(self, t: IntervalT):
        if self._usages[t] is None:
            return True

        return self.now() - self._usages[t] >= getattr(self, t)

    def dispatch_mode(self, t: IntervalT):
        if not self.use(t):
            return DispatchMode.RATELIMIT

        return DispatchMode.DISPATCH

    def use(self, t: IntervalT) -> bool:
        if not self.is_ready(t):
            return False

        self._usages[t] = self.now()

        return True

    async def wait_for(self, t: IntervalT):
        while True:
            time_remaining = self.time_until_ready(t)

            if time_remaining <= 0:
                break

            await asyncio.sleep(time_remaining / 1000.0)

    def set(self, t: IntervalT, value: int):
        setattr(self, t, value)

    def update(self, other: Self):
        for key in other.model_fields.keys():
            setattr(self, key, getattr(other, key))


class DisplaySettings(BaseModel):
    enabled: bool = True
    branding: bool = True
    while_printing_type: int = 0
    show_status: bool = True


class PrinterSettings(BaseModel):
    has_psu: bool = False
    has_filament_settings: bool = False
    display: Optional[DisplaySettings] = None


# Virtual spool position constant - used for VIRTUAL MMS type (external spools)
VIRTUAL_SPOOL_POSITION: int = 256


class MultiMaterialSolution(Enum):
    BAMBU_AMS = "bambu_ams"
    BAMBU_AMS_2_PRO = "bambu_ams_2_pro"
    BAMBU_AMS_HT = "bambu_ams_ht"
    BAMBU_AMS_LITE = "bambu_ams_lite"
    PRUSA_MMU3 = "prusa_mmu3"
    PRUSA_MMU2 = "prusa_mmu2"
    MOSAICPALETTE2S = "mosaic_palette2s"
    MOSAICPALETTE2 = "mosaic_palette2"
    BOXTURTLE = "boxturtle"
    CREALITY_CFS = "creality_cfs"
    ANYCUBIC_ACE_PRO = "anycubic_ace_pro"
    VIRTUAL = "virtual"
    CUSTOM = "custom"

    @property
    def can_chain(self) -> bool:
        return self in {
            self.CREALITY_CFS,
            self.ANYCUBIC_ACE_PRO,
            self.BAMBU_AMS,
            self.BAMBU_AMS_2_PRO,
            self.BAMBU_AMS_HT,
            self.BOXTURTLE,
        }

    @property
    def max_chains(self) -> Optional[int]:
        return {
            self.BAMBU_AMS_HT: 4,
            self.BAMBU_AMS: 4,
            self.BAMBU_AMS_2_PRO: 4,
            self.BAMBU_AMS_LITE: 1,
            self.CREALITY_CFS: 4,
        }.get(self)

    @property
    def default_size(self) -> int:
        return {
            self.PRUSA_MMU2: 5,
            self.PRUSA_MMU3: 5,
            self.BAMBU_AMS_HT: 1,
            self.VIRTUAL: 1,
        }.get(self, 4)

    @property
    def offset(self) -> int:
        """Starting extruder offset for this MMS type.

        - Standard MMS types: 0 (sequential slots starting from current position)
        - VIRTUAL type: 256 (external spool position)
        """
        if self == self.VIRTUAL:
            return VIRTUAL_SPOOL_POSITION
        return 0


class BedType(Enum):
    PRUSA_SMOOTH_SHEET = "prusa_smooth_sheet"
    PRUSA_TEXTURED_SHEET = "prusa_textured_sheet"
    PRUSA_SATIN_SHEET = "prusa_satin_sheet"
    PRUSA_NYLON_SHEET = "prusa_nylon_sheet"
    PRUSA_PP_SHEET = "prusa_pp_sheet"

    BAMBU_3D_EFFECT_PLATE = "bambu_3d_effect_plate"
    BAMBU_COOL_PLATE = "bambu_cool_plate"
    BAMBU_COOL_PLATE_SUPERTACK = "bambu_cool_plate_supertack"
    BAMBU_ENGINEERING_PLATE = "bambu_engineering_plate"
    BAMBU_GALAXY_SURFACE_PLATE = "bambu_galaxy_surface_plate"
    BAMBU_SMOOTH_PEI_PLATE = "bambu_smooth_pei_plate"
    BAMBU_STARRY_SURFACE_PLATE = "bambu_starry_surface_plate"
    BAMBU_TEXTURED_PEI_PLATE = "bambu_textured_pei_plate"
    BAMBU_DIAMOND_EFFECT_PLATE = "bambu_diamond_effect_plate"
    BAMBU_CARBON_FIBER_EFFECT_PLATE = "bambu_carbon_fiber_effect_plate"

    BIQU_PANDA_DESIGNER_HONEYCOMB = "biqu_panda_designer_honeycomb"
    BIQU_PANDA_BUILD_PLATE_DESIGNER_HOUNDSTOOTH = (
        "biqu_panda_build_plate_designer_houndstooth"
    )
    BIQU_PANDA_CRYOGRIP_FROSTBITE = "biqu_panda_cryogrip_frostbite"
    BIQU_PANDA_CRYOGRIP_GLACIER = "biqu_panda_cryogrip_glacier"
    BIQU_PANDA_CRYOGRIP_TEXTURED_STEEL_SHEET = (
        "biqu_panda_cryogrip_textured_steel_sheet"
    )

    ELEGOO_TEXTURED_PEI_SURFACE = "elegoo_textured_pei_surface"
    ELEGOO_SMOOTH_PEI_SURFACE = "elegoo_smooth_pei_surface"
    ELEGOO_STARRY_SURFACE = "elegoo_starry_surface"
    ELEGOO_DIAMOND_SURFACE = "elegoo_diamond_surface"
    ELEGOO_CARBON_FIBER_SURFACE = "elegoo_carbon_fiber_surface"
    ELEGOO_COOL_PLATE_SURFACE = "elegoo_cool_plate_surface"

    GENERIC_FLEXIBLE_PEI_SHEET = "generic_flexible_pei_sheet"
    GENERIC_GAROLITE_G10_PLATE = "generic_garolite_g10_plate"
    GENERIC_GLASS_PLATE = "generic_glass_plate"
    GENERIC_PP_SHEET = "generic_pp_sheet"

    AFTERMARKET_PEI_SHEET = "aftermarket_pei_sheet"
    CUSTOM = "custom"


class NozzleType(Enum):
    STANDARD = "standard"  # brass
    PLATED_BRASS = "plated_brass"  # plated brass
    HARDENED_STEEL = "hardened_steel"
    STAINLESS_STEEL = "stainless_steel"
    TUNGSTEN_CARBIDE = "tungsten_carbide"
    RUBY_TIPPED = "ruby_tipped"
    HEMISPHERICAL = "hemispherical"
    CUSTOM = "custom"


class VolumeType(Enum):
    STANDARD = "standard"
    HIGH_FLOW = "high_flow"


class NotificationEventSeverity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


class NotificationEventType(StrEnum):
    GENERIC = "generic"
    STACKED = "stacked"


class NotificationEventEffect(StrEnum):
    PRINT_PAUSE = "pause"
    PRINT_CANCEL = "cancel"


class NotificationEventActionType(StrEnum):
    BUTTON = "button"


_ActionType = TypeVar("_ActionType", bound=NotificationEventActionType)


class _NotificationEventAction(BaseModel, Generic[_ActionType]):
    type: _ActionType


class NotificationEventButtonAction(
    _NotificationEventAction[Literal[NotificationEventActionType.BUTTON]]
):
    type: Literal[NotificationEventActionType.BUTTON] = (
        NotificationEventActionType.BUTTON
    )
    label: str


NotificationEventActions = Annotated[
    Union[NotificationEventButtonAction], Field(discriminator="type")
]


class _NotificationActionResponse(BaseModel, Generic[_ActionType]):
    type: _ActionType


class NotificationEventButtonActionResponse(
    _NotificationActionResponse[Literal[NotificationEventActionType.BUTTON]]
):
    type: Literal[NotificationEventActionType.BUTTON] = (
        NotificationEventActionType.BUTTON
    )


NotificationActionResponses = Annotated[
    Union[NotificationEventButtonActionResponse], Field(discriminator="type")
]
