from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from simplyprint_ws_client.core.state import PrinterStatus

if TYPE_CHECKING:
    from simplyprint_ws_client.core.state import ToolState


class AmbientCheck:
    AMBIENT_CHECK_TIME = 5.0 * 60.0
    SAMPLE_CHECK_TIME = 20.0
    CHECK_INTERVAL = 5.0

    @staticmethod
    def detect(
        on_changed: Callable[[int], None],
        tools: List["ToolState"],
        initial_sample: Optional[float] = None,
        ambient: float = 0,
        status: Optional[PrinterStatus] = None,
    ) -> Tuple[Optional[float], int, float]:
        """
        By observing the change in temperature of the first tool
        the ambient temperature is chosen when the average drift of the
        temperature is less than 2 degrees.
        """

        if len(tools) == 0:
            return None, round(ambient), AmbientCheck.CHECK_INTERVAL

        tool0 = tools[0].temperature

        # Do not detect any ambient temperature if we are printing or heating.
        # When we are cooling down we rely on the printer cooling down more than 2 degrees per 5 min + 20 sec window.
        if PrinterStatus.is_printing(status) or tool0.is_heating():
            return None, round(ambient), AmbientCheck.AMBIENT_CHECK_TIME

        # Wait until we can collect a sample (20-seconds)
        if initial_sample is None:
            return tool0.actual, round(ambient), AmbientCheck.SAMPLE_CHECK_TIME

        diff = abs(tool0.actual - initial_sample)

        if diff <= 2.0:
            new_ambient = (tool0.actual + initial_sample) / 2

            if ambient != new_ambient:
                on_changed(round(new_ambient))

            return None, round(new_ambient), AmbientCheck.AMBIENT_CHECK_TIME

        return tool0.actual, round(ambient), AmbientCheck.SAMPLE_CHECK_TIME
