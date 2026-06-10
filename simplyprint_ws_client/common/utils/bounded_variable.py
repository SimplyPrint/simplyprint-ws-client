from typing import TYPE_CHECKING, TypeVar, Union, Generic, Optional

if TYPE_CHECKING:
    pass

TIntervalValue = TypeVar("TIntervalValue", bound=Union[float, int])


class BoundedInterval(Generic[TIntervalValue]):
    max: TIntervalValue
    step: TIntervalValue
    default: Optional[TIntervalValue] = None

    def __init__(
        self,
        max_value: TIntervalValue,
        step: TIntervalValue,
        default: Optional[TIntervalValue] = None,
    ):
        self.max = max_value
        self.step = step
        self.default = default

    def create_variable(
        self, value: Optional[TIntervalValue] = None
    ) -> "BoundedVariable":
        default = self.default if value is None else value

        if default is None:
            raise ValueError(
                f"Cannot create bounded variable {self.__class__.__name__} with default value None"
            )

        return BoundedVariable(default, self)


class BoundedVariable(Generic[TIntervalValue]):
    interval: BoundedInterval[TIntervalValue]

    _starting_value: TIntervalValue
    _current_value: TIntervalValue

    def __init__(
        self, value: TIntervalValue, interval: BoundedInterval[TIntervalValue]
    ):
        self._starting_value = value
        self._current_value = value
        self.interval = interval

    @property
    def value(self) -> TIntervalValue:
        return self._current_value

    @property
    def default(self) -> TIntervalValue:
        return self._starting_value

    def increment(self, step: Optional[TIntervalValue] = None) -> TIntervalValue:
        step = step or self.interval.step
        self._current_value = min(self._current_value + step, self.interval.max)
        return self._current_value

    def exponential_increment(
        self, factor: Optional[TIntervalValue] = None
    ) -> TIntervalValue:
        factor = factor or self.interval.step
        self._current_value = min(self._current_value * factor, self.interval.max)
        return self._current_value

    def is_at_bound(self) -> bool:
        return self._current_value >= self.interval.max

    def guard_until_bound(self, increment_if_false=True, reset_if_true=True) -> bool:
        is_at_bound = self.is_at_bound()

        if is_at_bound and reset_if_true:
            self.reset()

        if not is_at_bound and increment_if_false:
            self.increment()

        return is_at_bound

    def reset(self):
        self._current_value = self._starting_value

    def __str__(self):
        return f"{self.value}/{self.interval}"
