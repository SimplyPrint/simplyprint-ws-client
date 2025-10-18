__all__ = ["StateModel", "Exclusive", "Untracked"]

import datetime
import decimal
import enum
import functools
import weakref
from itertools import repeat
from typing import (
    Optional,
    Dict,
    Any,
    TypeVar,
    TYPE_CHECKING,
    ClassVar,
    Type,
    get_origin,
    List,
    Set,
    Tuple,
    Mapping,
    get_args,
    Union,
    no_type_check,
    cast,
    Annotated,
)

from pydantic import BaseModel, PrivateAttr
from pydantic.fields import FieldInfo

from simplyprint_ws_client.shared.utils.synchronized import Synchronized

_ExclusiveSentinel = object()
_TExclusive = TypeVar("_TExclusive", bound=Any)
Exclusive = Annotated[_TExclusive, _ExclusiveSentinel]


@functools.lru_cache
def _is_exclusive(field_info: FieldInfo) -> bool:
    return any(v is _ExclusiveSentinel for v in field_info.metadata)


_UntrackedSentinel = object()
_TUntracked = TypeVar("_TUntracked", bound=Any)
Untracked = Annotated[_TUntracked, _UntrackedSentinel]


@functools.lru_cache
def _is_untracked(field_info: FieldInfo) -> bool:
    return any(v is _UntrackedSentinel for v in field_info.metadata)


TStateModel = TypeVar("TStateModel", bound="StateModel")


class StateModel(BaseModel, Synchronized):
    """
    Adapted from pydantic-changetracker (ChangeDetectionMixin).
    MIT License
    Copyright (c) TEAM23 GmbH 2022.
    """

    COMPARABLE_TYPES: ClassVar[tuple] = (
        str,
        int,
        float,
        bool,
        enum.Enum,
        decimal.Decimal,
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
        BaseModel,
    )

    @classmethod
    @functools.lru_cache
    def is_pydantic_change_detect_annotation(
        cls, annotation: Type[Any] = None, field_info: Optional[FieldInfo] = None
    ) -> bool:
        """
        Return True if the given annotation is a ChangeDetectionMixin annotation.
        """
        if field_info:
            if _is_untracked(field_info):
                return False

            annotation = field_info.annotation

        # if annotation is an ChangeDetectionMixin everything is easy
        if (
            get_origin(annotation)
            is None  # If the origin is None, it's likely a concrete class
            and isinstance(annotation, type)
            and issubclass(annotation, StateModel)
        ):
            return True

        # Otherwise we may need to handle typing arguments
        origin = get_origin(annotation)
        if (
            origin is List
            or origin is list
            or origin is Set
            or origin is set
            or origin is Tuple
            or origin is tuple
        ):
            return cls.is_pydantic_change_detect_annotation(get_args(annotation)[0])
        elif origin is Dict or origin is dict or origin is Mapping:
            return cls.is_pydantic_change_detect_annotation(get_args(annotation)[1])
        elif origin is Union:
            # Note: This includes Optional, as Optional[...] is just Union[..., None]
            return any(
                cls.is_pydantic_change_detect_annotation(arg)
                for arg in get_args(annotation)
            )

        # If we did not detect an annotation, return False
        return False

    if TYPE_CHECKING:
        model_self_changed_fields: Dict[str, int] = PrivateAttr(...)
        ctx: weakref.ref = PrivateAttr(...)

    __slots__ = ("model_self_changed_fields", "ctx")

    def __init__(self, ctx=lambda: None, **kwargs: Any) -> None:
        BaseModel.__init__(self, **kwargs)
        Synchronized.__init__(self)
        # Default to static no-context to prevent unnecessary errors
        object.__setattr__(self, "ctx", ctx)
        self.model_reset_changed()

    def provide_context(self, ctx: Union["StateModel", weakref.ref]) -> None:
        """Give tree a reference"""
        if isinstance(ctx, StateModel):
            object.__setattr__(self, "ctx", ctx.ctx)
        else:
            object.__setattr__(self, "ctx", ctx)

        for field in self.__class__.model_fields:
            value = getattr(self, field)

            if isinstance(value, StateModel):
                value.provide_context(self)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, StateModel):
                        item.provide_context(self)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, StateModel):
                        item.provide_context(self)

    def model_reset_changed(self, *keys: str, v: Optional[int] = None) -> None:
        """
        Reset the changed state, this will clear model_self_changed_fields.
        """

        if not keys and v is None:
            object.__setattr__(self, "model_self_changed_fields", {})
            return

        if not keys and v is not None:
            object.__setattr__(
                self,
                "model_self_changed_fields",
                {k: v2 for k, v2 in self.model_self_changed_fields.items() if v2 > v},
            )
            return

        for key in keys:
            if v is not None and self.model_self_changed_fields.get(key, -1) > v:
                continue

            self.model_self_changed_fields.pop(key, None)

    @property
    def model_changed_fields(self) -> set:
        """
        Return a dictionary of all changed fields.
        """

        changed_fields = set(self.model_self_changed_fields.keys())

        for field_name, model_field in self.__class__.model_fields.items():
            field_value = self.__dict__[field_name]

            if isinstance(field_value, StateModel) and field_value.model_has_changed:
                changed_fields.add(field_name)

            if field_value and self.is_pydantic_change_detect_annotation(
                field_info=model_field
            ):
                if isinstance(field_value, (list, tuple)):
                    field_value_list = list(enumerate(field_value))
                elif isinstance(field_value, dict):
                    field_value_list = list(field_value.items())
                else:
                    continue

                for inner_field_index, inner_field_value in field_value_list:
                    if (
                        isinstance(inner_field_value, StateModel)
                        and inner_field_value.model_has_changed
                    ):
                        changed_fields.add(field_name)
                        break

        return changed_fields

    @property
    def model_recursive_changeset(self) -> Dict[str, int]:
        changed_fields = self.model_self_changed_fields.copy()

        def set_changed_field(field: str, change_id: int) -> None:
            nonlocal changed_fields
            changed_fields[field] = max(changed_fields.get(field, -1), change_id)

        for field_name, model_field in self.__class__.model_fields.items():
            field_value = self.__dict__[field_name]

            if not field_value or not self.is_pydantic_change_detect_annotation(
                field_info=model_field
            ):
                continue

            # Handle direct nested state models.
            if isinstance(field_value, StateModel) and (
                field_value_changes := field_value.model_recursive_changeset
            ):
                for key, value in field_value_changes.items():
                    set_changed_field(f"{field_name}.{key}", value)

                set_changed_field(field_name, max(field_value_changes.values()))
                continue

            # Handle collection of nested state models.
            if isinstance(field_value, list):
                # Generate a matchable key for list changes
                field_values = zip(repeat("*"), field_value)
            elif isinstance(field_value, tuple):
                field_values = enumerate(field_value)  # Tuples have fixed indices
            elif isinstance(field_value, dict):
                field_values = field_value.items()
            else:
                continue

            for inner_field_index, inner_field_value in field_values:
                if isinstance(inner_field_value, StateModel) and (
                    inner_field_value_changes
                    := inner_field_value.model_recursive_changeset
                ):
                    for key, value in inner_field_value_changes.items():
                        set_changed_field(
                            f"{field_name}.{inner_field_index}.{key}", value
                        )

                    x = max(inner_field_value_changes.values())
                    set_changed_field(f"{field_name}.{inner_field_index}", x)
                    set_changed_field(field_name, x)

        return changed_fields

    @property
    def model_has_changed(self) -> bool:
        """Return True, when some field was changed."""

        if self.model_self_changed_fields:
            return True

        return bool(self.model_changed_fields)

    def model_has_changes(self, *fields: str) -> bool:
        """
        Check if the given fields have changed.
        If no fields are given, check if any field has changed.
        """

        if not fields:
            return self.model_has_changed

        # Ensure all fields exists
        for name in fields:
            if name not in self.__class__.model_fields:
                raise AttributeError(f"Field {name} not available in this model")

        return any(name in self.model_self_changed_fields for name in fields)

    def model_set_changed(self, *fields: str) -> None:
        """Set fields as changed."""

        # Ensure all fields exists
        for name in fields:
            if name not in self.__class__.model_fields:
                raise AttributeError(f"Field {name} not available in this model")

        ctx = self.ctx()

        if not ctx:
            return

        # Mark fields as changed
        for name in fields:
            self.model_self_changed_fields[name] = ctx.next_msg_id()

        ctx.signal()

    def _model_value_is_comparable_type(self, value: Any) -> bool:
        if isinstance(value, (list, set, tuple)):
            return all(self._model_value_is_comparable_type(i) for i in value)
        elif isinstance(value, dict):
            return all(
                (
                    self._model_value_is_comparable_type(k)
                    and self._model_value_is_comparable_type(v)
                )
                for k, v in value.items()
            )

        return value is None or isinstance(value, self.COMPARABLE_TYPES)

    @staticmethod
    def _model_value_is_actually_unchanged(value1: Any, value2: Any) -> bool:
        return value1 == value2

    @no_type_check
    def __setattr__(self, name, value) -> None:  # noqa: ANN001
        field_info = self.__class__.model_fields.get(name)

        # Private attributes do not need to be handled
        # Same for non-model fields, or untracked fields.
        if (
            field_info is None
            or _is_untracked(field_info)
            or (
                self.__private_attributes__ and name in self.__private_attributes__
            )  # can be None
        ):
            super().__setattr__(name, value)
            return

        # Get original value
        original_value = self.__dict__[name]

        # Store changed value using pydantic
        super().__setattr__(name, value)

        # Check if value has actually been changed
        has_changed = True

        # Fetch original from original_update so we don't have to check everything again
        # Don't use value parameter directly, as pydantic validation might have changed it
        # (when validate_assignment == True)
        current_value = self.__dict__[name]

        if (
            not _is_exclusive(field_info)
            and self._model_value_is_comparable_type(original_value)
            and self._model_value_is_comparable_type(current_value)
            and self._model_value_is_actually_unchanged(original_value, current_value)
        ):
            has_changed = False

        # Store changed state
        if has_changed and (ctx := self.ctx()):
            self.model_self_changed_fields[name] = ctx.next_msg_id()
            ctx.signal()

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["model_self_changed_fields"] = self.model_self_changed_fields.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)

        if "model_self_changed_fields" in state:
            object.__setattr__(
                self, "model_self_changed_fields", state["model_self_changed_fields"]
            )
        else:
            object.__setattr__(self, "model_self_changed_fields", {})

    @classmethod
    def model_construct(
        cls: Type[TStateModel], *args: Any, **kwargs: Any
    ) -> TStateModel:
        """Construct an unvalidated instance"""

        m = cast(TStateModel, super().model_construct(*args, **kwargs))
        m.model_reset_changed()
        return m

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.model_reset_changed()

    def model_update(
        self, other: TStateModel, exclude: Optional[Set[str]] = None
    ) -> None:
        for field_name, model_field in self.__class__.model_fields.items():
            if exclude and field_name in exclude:
                continue

            if field_name in other.__class__.model_fields:
                field_value = getattr(other, field_name)

                if isinstance(field_value, StateModel):
                    field_value.provide_context(self)

                setattr(self, field_name, field_value)

    def __copy__(self: TStateModel) -> TStateModel:
        clone = cast(
            TStateModel,
            super().__copy__(),
        )
        object.__setattr__(
            clone, "model_self_changed_fields", self.model_self_changed_fields.copy()
        )
        return clone

    def __deepcopy__(
        self: TStateModel, memo: Optional[Dict[int, Any]] = None
    ) -> TStateModel:
        clone = cast(
            TStateModel,
            super().__deepcopy__(memo=memo),
        )
        object.__setattr__(
            clone, "model_self_changed_fields", self.model_self_changed_fields.copy()
        )
        return clone
